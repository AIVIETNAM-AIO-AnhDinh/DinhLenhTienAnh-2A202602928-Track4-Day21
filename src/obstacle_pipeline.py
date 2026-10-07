"""Topic D: phát hiện vật cản từ LiDAR, không dùng deep learning.

Pipeline:  crop ROI -> voxel downsample -> RANSAC ground plane -> DBSCAN -> box mỗi cluster

    python -m src.obstacle_pipeline --data-root data/kitti_mini --frame 000011
    python -m src.obstacle_pipeline --data-root data/nuscenes_mini_subset --frame scene-0103_010
    python -m src.obstacle_pipeline --help

Chạy được cho cả KITTI (x trước, y trái, z lên) và nuScenes (x phải, y trước, z lên):
pipeline chỉ dùng z-up và khoảng cách ngang sqrt(x²+y²), nên không phụ thuộc trục x/y.

Tham khảo API Open3D: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html
(voxel_down_sample, segment_plane, cluster_dbscan).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from starter.datasets import dataset_type, load_frame  # noqa: E402
from starter.kitti_io import KittiCalib, KittiObject  # noqa: E402
from starter.projection import box3d_corners_cam, cam_to_image, velo_to_cam  # noqa: E402

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Class được coi là "vật cản có nhãn" khi so khớp cluster với GT.
OBSTACLE_CLASSES = {
    # KITTI
    "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc",
    # nuScenes (đã map trong starter/nuscenes_io.py)
    "Bus", "Trailer", "ConstructionVehicle", "Bicycle", "Motorcycle", "Barrier", "TrafficCone",
}


@dataclass
class PipelineConfig:
    voxel_size: float = 0.10          # m, cạnh voxel khi downsample
    distance_threshold: float = 0.20  # m, điểm cách mặt phẳng <= ngưỡng này bị coi là mặt đất
    ransac_n: int = 3
    num_iterations: int = 500
    eps: float = 0.50                 # m, bán kính láng giềng của DBSCAN
    min_points: int = 5               # số điểm tối thiểu để là core point của DBSCAN
    min_cluster_points: int = 5       # cluster nhỏ hơn bị bỏ (nhiễu)
    min_range: float = 2.5            # m, bỏ điểm trên thân xe ego (nuScenes có điểm trên nóc xe)
    max_range: float = 40.0           # m, chỉ xét vật cản trong bán kính này
    roi: str = "camera"               # "camera": chỉ giữ điểm trong FOV camera (vùng có label); "all": 360°
    seed: int = 0
    ransac_impl: str = "numpy"        # "numpy" (có seed, tái lập được) hoặc "open3d" (segment_plane)
    ground_method: str = "plane"      # "plane": 1 mặt phẳng RANSAC; "zones": mỗi vành khoảng cách 1 mặt phẳng
    zone_edges: tuple = (0.0, 10.0, 20.0, 1e9)  # m, ranh giới các vành cho ground_method="zones"
    max_tilt_deg: float = 10.0        # zones: bỏ mặt phẳng ứng viên nghiêng hơn góc này (tường, sườn dốc)


# ---------------------------------------------------------------- các bước pipeline

def roi_mask(points: np.ndarray, calib: KittiCalib, image_shape, cfg: PipelineConfig) -> np.ndarray:
    """Bỏ NaN/Inf, bỏ điểm trên thân xe ego, giới hạn bán kính, (tuỳ chọn) chỉ giữ FOV camera."""
    finite = np.isfinite(points[:, :3]).all(axis=1)
    xyz = np.where(finite[:, None], points[:, :3], 0.0)
    r = np.linalg.norm(xyz[:, :2], axis=1)
    mask = finite & (r > cfg.min_range) & (r < cfg.max_range)
    if cfg.roi == "camera":
        _, _, in_img = cam_to_image(velo_to_cam(xyz, calib), calib.P2, image_shape)
        mask &= in_img
    return mask


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0:
        return xyz
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    return np.asarray(pcd.voxel_down_sample(voxel_size).points)


def ransac_plane_numpy(xyz: np.ndarray, distance_threshold: float, num_iterations: int,
                       seed: int, max_tilt_deg: float = 90.0) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC mặt phẳng bằng numpy, có seed -> chạy lại ra đúng cùng kết quả.
    (open3d segment_plane chạy song song nên kết quả đổi giữa các lần chạy dù đã đặt seed.)
    max_tilt_deg < 90: chỉ nhận mặt phẳng ứng viên gần nằm ngang."""
    rng = np.random.default_rng(seed)
    tri = xyz[rng.integers(0, len(xyz), size=(num_iterations, 3))]       # (I, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])            # (I, 3)
    norm = np.linalg.norm(n, axis=1)
    ok = norm > 1e-9
    n, p0 = n[ok] / norm[ok, None], tri[ok, 0]
    flat = np.abs(n[:, 2]) >= np.cos(np.deg2rad(max_tilt_deg))
    if not flat.any():
        return np.array([0.0, 0.0, 1.0, np.inf]), np.zeros(len(xyz), dtype=bool)
    n, p0 = n[flat], p0[flat]
    d = -(n * p0).sum(1)
    best, best_count = 0, -1
    for s in range(0, len(n), 64):  # chia batch để không tạo ma trận (N, I) quá lớn
        dist = np.abs(xyz @ n[s:s + 64].T + d[s:s + 64])
        counts = (dist <= distance_threshold).sum(0)
        if counts.max() > best_count:
            best, best_count = s + int(counts.argmax()), int(counts.max())
    plane = np.r_[n[best], d[best]]
    inliers = np.abs(xyz @ plane[:3] + plane[3]) <= distance_threshold
    return plane, inliers


def _upward(plane: np.ndarray) -> np.ndarray:
    plane = plane / np.linalg.norm(plane[:3])
    return -plane if plane[2] < 0 else plane  # pháp tuyến hướng lên -> "độ cao trên mặt đất" dương


def segment_ground(xyz: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tách mặt đất. Trả về (plane toàn cục [a,b,c,d], mask ground, độ cao từng điểm so với mặt đất cục bộ)."""
    plane, ground = _segment_plane(xyz, cfg)
    height = xyz @ plane[:3] + plane[3]
    if cfg.ground_method == "zones":
        ground, height = _segment_zones(xyz, cfg, plane)
    return plane, ground, height


def _segment_zones(xyz: np.ndarray, cfg: PipelineConfig, fallback: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mỗi vành khoảng cách [r_i, r_i+1) fit 1 mặt phẳng gần nằm ngang riêng (ý tưởng concentric zone
    của Patchwork, Lim et al., RA-L 2021, rút gọn). Đường dốc/đỉnh dốc được xấp xỉ bằng nhiều mặt phẳng.
    Vành quá ít điểm hoặc không có mặt phẳng hợp lệ thì dùng mặt phẳng của vành trước."""
    r = np.linalg.norm(xyz[:, :2], axis=1)
    ground = np.zeros(len(xyz), dtype=bool)
    height = xyz @ fallback[:3] + fallback[3]
    prev = fallback
    for lo, hi in zip(cfg.zone_edges[:-1], cfg.zone_edges[1:]):
        idx = np.flatnonzero((r >= lo) & (r < hi))
        if len(idx) == 0:
            continue
        plane = prev
        if len(idx) >= 50:
            cand, inl = ransac_plane_numpy(xyz[idx], cfg.distance_threshold, cfg.num_iterations, cfg.seed,
                                           cfg.max_tilt_deg)
            if inl.sum() >= 30:
                plane = _upward(cand)
        h = xyz[idx] @ plane[:3] + plane[3]
        ground[idx] = np.abs(h) <= cfg.distance_threshold
        height[idx] = h
        prev = plane
    return ground, height


def _segment_plane(xyz: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC một mặt phẳng cho toàn bộ ROI (baseline của đề: open3d segment_plane)."""
    if cfg.ransac_impl == "open3d":
        o3d.utility.random.seed(cfg.seed)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        plane, inliers = pcd.segment_plane(cfg.distance_threshold, cfg.ransac_n, cfg.num_iterations)
        plane = np.asarray(plane, dtype=float)
        ground = np.zeros(len(xyz), dtype=bool)
        ground[np.asarray(inliers, dtype=int)] = True
    else:
        plane, ground = ransac_plane_numpy(xyz, cfg.distance_threshold, cfg.num_iterations, cfg.seed)
    return _upward(plane), ground


def cluster_dbscan(xyz: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Nhãn cluster (N,), -1 = nhiễu. Cluster < min_cluster_points cũng gán -1."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=int)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    labels = np.asarray(pcd.cluster_dbscan(eps=cfg.eps, min_points=cfg.min_points, print_progress=False))
    if labels.max(initial=-1) < 0:
        return labels
    counts = np.bincount(labels[labels >= 0])
    small = np.flatnonzero(counts < cfg.min_cluster_points)
    labels[np.isin(labels, small)] = -1
    # đánh lại số liên tục 0..K-1, sắp theo khoảng cách gần nhất tăng dần
    ids = np.unique(labels[labels >= 0])
    if len(ids) == 0:
        return labels
    r = np.linalg.norm(xyz[:, :2], axis=1)
    order = sorted(ids, key=lambda k: r[labels == k].min())
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap.get(l, -1) for l in labels])


def cluster_boxes(xyz: np.ndarray, labels: np.ndarray, height: np.ndarray) -> list[dict]:
    """Box axis-aligned mỗi cluster + các đại lượng robot cần: khoảng cách gần nhất, độ cao trên mặt đất."""
    boxes = []
    for k in range(labels.max(initial=-1) + 1):
        p = xyz[labels == k]
        h = height[labels == k]
        boxes.append({
            "id": k, "n_points": len(p),
            "min": p.min(0), "max": p.max(0), "center": p.mean(0),
            "extent": p.max(0) - p.min(0),
            "nearest_m": float(np.linalg.norm(p[:, :2], axis=1).min()),
            "height_top_m": float(h.max()),
            # cluster dẹt (cao < 0.25 m) nhưng rộng > 2 m: nghi là mặt đất bị lọt vào tập vật cản
            "flat": bool((p[:, 2].max() - p[:, 2].min()) < 0.25 and (p.max(0) - p.min(0))[:2].max() > 2.0),
        })
    return boxes


def run_pipeline(points: np.ndarray, calib: KittiCalib, image_shape, cfg: PipelineConfig) -> dict:
    """Chạy toàn bộ pipeline trên 1 frame, kèm thời gian từng bước (ms)."""
    t = [time.perf_counter()]
    roi = roi_mask(points, calib, image_shape, cfg)
    xyz_roi = points[roi, :3].astype(np.float64)
    t.append(time.perf_counter())
    xyz_ds = voxel_downsample(xyz_roi, cfg.voxel_size)
    t.append(time.perf_counter())
    plane, ground, height = segment_ground(xyz_ds, cfg)
    t.append(time.perf_counter())
    obj_xyz = xyz_ds[~ground]
    labels = cluster_dbscan(obj_xyz, cfg)
    t.append(time.perf_counter())
    boxes = cluster_boxes(obj_xyz, labels, height[~ground])
    t.append(time.perf_counter())
    dt = np.diff(t) * 1e3
    return {
        "xyz_roi": xyz_roi, "xyz_ds": xyz_ds, "plane": plane, "ground": ground,
        "height": height, "obj_xyz": obj_xyz, "obj_height": height[~ground], "labels": labels, "boxes": boxes,
        "ms": dict(zip(["roi", "voxel", "ground", "dbscan", "boxes"], dt)) | {"total": float(dt.sum())},
    }


# ---------------------------------------------------------------- so khớp với GT (label_2)

def points_in_gt_box(xyz_velo: np.ndarray, obj: KittiObject, calib: KittiCalib, margin: float = 0.0) -> np.ndarray:
    """Mask điểm (velodyne frame) nằm trong 3D box GT (camera frame, KITTI convention), nới thêm `margin` mét."""
    p = velo_to_cam(xyz_velo, calib) - obj.location
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    # box3d_corners_cam dùng R_y(ry) @ local; đảo lại: local = R_y(ry)^T @ p
    lx = c * p[:, 0] - s * p[:, 2]
    lz = s * p[:, 0] + c * p[:, 2]
    ly = p[:, 1]
    h, w, l = obj.dimensions
    return ((np.abs(lx) <= l / 2 + margin) & (np.abs(lz) <= w / 2 + margin)
            & (ly <= margin) & (ly >= -h - margin))


def gt_corners_velo(obj: KittiObject, calib: KittiCalib) -> np.ndarray:
    """8 góc box GT trong velodyne frame (để vẽ BEV)."""
    corners = np.hstack([box3d_corners_cam(obj), np.ones((8, 1))])
    return (corners @ np.linalg.inv(calib.T_cam_velo).T)[:, :3]


def match_clusters_to_gt(res: dict, labels_gt: list[KittiObject], calib: KittiCalib,
                         min_gt_points: int = 5, hit_points: int = 3, margin: float = 0.25) -> dict:
    """GT được 'phát hiện' nếu có cluster có >= hit_points điểm trong box GT (nới margin).
    Cluster là TP nếu >= 50% điểm của nó nằm trong một box GT.
    Chỉ tính GT thuộc OBSTACLE_CLASSES và có >= min_gt_points điểm LiDAR (ROI, chưa downsample) trong box:
    vật không có điểm LiDAR nào thì không pipeline LiDAR nào thấy được, tính vào recall là đo sai."""
    obj_xyz, labels = res["obj_xyz"], res["labels"]
    n_clusters = labels.max(initial=-1) + 1
    gts = []
    for obj in labels_gt:
        if obj.type not in OBSTACLE_CLASSES:
            continue
        n_raw = int(points_in_gt_box(res["xyz_roi"], obj, calib).sum())
        if n_raw < min_gt_points:
            continue
        gts.append((obj, n_raw))

    inside = np.zeros((len(gts), n_clusters), dtype=int)  # số điểm của cluster k trong GT g
    for g, (obj, _) in enumerate(gts):
        m = points_in_gt_box(obj_xyz, obj, calib, margin) & (labels >= 0)
        if m.any():
            inside[g] += np.bincount(labels[m], minlength=n_clusters)
    sizes = np.bincount(labels[labels >= 0], minlength=n_clusters) if n_clusters else np.zeros(0)

    gt_hit = (inside >= hit_points).any(axis=1) if n_clusters else np.zeros(len(gts), bool)
    cl_tp = (inside >= 0.5 * sizes).any(axis=0) & (sizes > 0) if len(gts) else np.zeros(n_clusters, bool)
    # under-segmentation: 1 cluster phủ >= 2 GT. over-segmentation: 1 GT bị chia ra >= 2 cluster.
    merged = int(((inside >= hit_points).sum(axis=0) >= 2).sum()) if len(gts) else 0
    split = int(((inside >= hit_points).sum(axis=1) >= 2).sum()) if n_clusters else 0
    per_gt = [{
        "type": obj.type, "n_raw_points": n_raw, "hit": bool(gt_hit[g]),
        "range_m": float(np.linalg.norm(gt_corners_velo(obj, calib).mean(0)[:2])),
        "height_m": float(obj.dimensions[0]),
    } for g, (obj, n_raw) in enumerate(gts)]
    return {
        "n_gt": len(gts), "n_gt_hit": int(gt_hit.sum()),
        "n_clusters": int(n_clusters), "n_cluster_tp": int(cl_tp.sum()),
        "merged": merged, "split": split, "per_gt": per_gt,
    }


# ---------------------------------------------------------------- BEV occupancy grid

def bev_occupancy(obj_xyz: np.ndarray, height: np.ndarray, cell: float = 0.2, extent: float = 40.0,
                  h_min: float = 0.05, h_max: float = 2.5) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Lưới chiếm chỗ 2D: ô bị chiếm nếu có điểm non-ground cao h_min..h_max (m) so với mặt đất.
    Điểm cao hơn h_max (tán cây, biển báo trên cao) không chặn đường robot."""
    p = obj_xyz[(height >= h_min) & (height <= h_max)]
    n = int(2 * extent / cell)
    grid = np.zeros((n, n), dtype=np.uint8)
    ij = np.floor((p[:, :2] + extent) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    grid[ij[ok, 1], ij[ok, 0]] = 1  # hàng = y, cột = x
    return grid, (-extent, extent, -extent, extent)


# ---------------------------------------------------------------- hình minh hoạ

def forward_axes(dtype: str) -> tuple[int, int, int]:
    """(index trục 'phía trước', index trục 'bên trái', dấu của trục trái) để vẽ BEV xe hướng lên trên."""
    return (0, 1, 1) if dtype == "kitti" else (1, 0, -1)  # nuScenes: y trước, x phải -> trái = -x


def _bev(ax, xyz, c, dtype, title, s=0.3, cmap=None, lim=40.0):
    f, l, sgn = forward_axes(dtype)
    ax.scatter(sgn * xyz[:, l], xyz[:, f], c=c, s=s, cmap=cmap, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(lim * 0.75, -lim * 0.75)  # trái của xe nằm bên trái hình
    ax.set_ylim(0, lim)
    ax.set_xlabel("trái  <-  (m)  ->  phải", fontsize=7)
    ax.tick_params(labelsize=7)


def _draw_gt_bev(ax, labels_gt, calib, dtype, color="lime"):
    f, l, sgn = forward_axes(dtype)
    for obj in labels_gt:
        if obj.type not in OBSTACLE_CLASSES:
            continue
        c = gt_corners_velo(obj, calib)[[0, 1, 2, 3, 0]]
        ax.plot(sgn * c[:, l], c[:, f], color=color, lw=0.8)


def _cluster_colors(labels: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("tab20")
    col = np.array([cmap(k % 20) for k in np.maximum(labels, 0)])
    col[labels < 0] = (0.6, 0.6, 0.6, 1.0)
    return col


def plot_steps(res: dict, labels_gt, calib, dtype: str, title: str, out: Path) -> None:
    """Ảnh 4 panel BEV: ROI -> downsample -> ground/non-ground -> cluster + box (GT viền xanh lá)."""
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
    _bev(axs[0], res["xyz_roi"], "k", dtype, f"1. ROI: {len(res['xyz_roi'])} điểm", s=0.1)
    _bev(axs[1], res["xyz_ds"], "k", dtype, f"2. Voxel downsample: {len(res['xyz_ds'])} điểm", s=0.2)
    g = res["ground"]
    _bev(axs[2], res["xyz_ds"], np.where(g, "#c8c8c8", "#d62728"), dtype,
         f"3. RANSAC: ground {g.sum()} (xám) / non-ground {(~g).sum()} (đỏ)", s=0.2)
    lab = res["labels"]
    _bev(axs[3], res["obj_xyz"], _cluster_colors(lab), dtype,
         f"4. DBSCAN: {lab.max(initial=-1) + 1} cluster (GT: viền xanh lá)", s=0.6)
    f, l, sgn = forward_axes(dtype)
    for b in res["boxes"]:
        x0, x1 = sorted([sgn * b["min"][l], sgn * b["max"][l]])
        axs[3].add_patch(plt.Rectangle((x0, b["min"][f]), x1 - x0, b["max"][f] - b["min"][f],
                                       fill=False, ec="k", lw=0.6))
    _draw_gt_bev(axs[3], labels_gt, calib, dtype)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_camera_overlay(image: np.ndarray, res: dict, calib: KittiCalib, out: Path,
                        labels_gt=None) -> np.ndarray:
    """Chiếu điểm của từng cluster lên ảnh (màu theo cluster) + 2D box mỗi cluster + khoảng cách gần nhất."""
    vis = (image * 0.6).astype(np.uint8)
    uv_g, _, _ = cam_to_image(velo_to_cam(res["xyz_ds"][res["ground"]], calib), calib.P2, image.shape)
    for u, v in uv_g.astype(int):
        cv2.circle(vis, (u, v), 1, (120, 120, 120), -1)
    lab = res["labels"]
    cols = (_cluster_colors(lab)[:, :3][:, ::-1] * 255).astype(int)
    uv, _, m = cam_to_image(velo_to_cam(res["obj_xyz"], calib), calib.P2, image.shape)
    lab_in, col_in = lab[m], cols[m]
    for (u, v), k, c in zip(uv.astype(int), lab_in, col_in):
        if k >= 0:
            cv2.circle(vis, (u, v), 2, tuple(int(x) for x in c), -1)
    for b in res["boxes"]:
        sel = lab_in == b["id"]
        if sel.sum() < 3:
            continue
        (x1, y1), (x2, y2) = uv[sel].min(0).astype(int), uv[sel].max(0).astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(vis, f"{b['nearest_m']:.1f}m", (x1, max(10, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for obj in labels_gt or []:
        if obj.type in OBSTACLE_CLASSES:
            x1, y1, x2, y2 = obj.bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), vis)
    return vis


def plot_occupancy(res: dict, dtype: str, labels_gt, calib, out: Path, cell: float = 0.2) -> None:
    grid, _ = bev_occupancy(res["obj_xyz"], res["obj_height"], cell=cell)
    n = grid.shape[0]
    xs = (np.arange(n) + 0.5) * cell - n * cell / 2
    occ = np.argwhere(grid)
    pts = np.c_[xs[occ[:, 1]], xs[occ[:, 0]], np.zeros(len(occ))]  # (x, y) velodyne frame
    fig, ax = plt.subplots(figsize=(6, 6))
    f, l, sgn = forward_axes(dtype)
    ax.scatter(sgn * pts[:, l], pts[:, f], s=1.5, marker="s", c="k", linewidths=0)
    _draw_gt_bev(ax, labels_gt, calib, dtype)
    ax.plot(0, 0, marker="^", color="r", ms=8)
    ax.set_aspect("equal")
    ax.set_xlim(30, -30)
    ax.set_ylim(-2, 40)
    ax.set_title(f"BEV occupancy {cell} m/ô: {int(grid.sum())} ô bị chiếm (GT viền xanh lá)", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def add_config_args(ap: argparse.ArgumentParser) -> None:
    d = PipelineConfig()
    ap.add_argument("--voxel-size", type=float, default=d.voxel_size, help="cạnh voxel (m), 0 = không downsample")
    ap.add_argument("--distance-threshold", type=float, default=d.distance_threshold,
                    help="ngưỡng RANSAC (m): điểm cách mặt phẳng <= ngưỡng bị coi là mặt đất")
    ap.add_argument("--num-iterations", type=int, default=d.num_iterations, help="số vòng RANSAC")
    ap.add_argument("--eps", type=float, default=d.eps, help="bán kính láng giềng DBSCAN (m)")
    ap.add_argument("--min-points", type=int, default=d.min_points, help="min_points của DBSCAN")
    ap.add_argument("--min-cluster-points", type=int, default=d.min_cluster_points, help="bỏ cluster nhỏ hơn")
    ap.add_argument("--min-range", type=float, default=d.min_range, help="bỏ điểm gần hơn (m), lọc thân xe ego")
    ap.add_argument("--max-range", type=float, default=d.max_range, help="bỏ điểm xa hơn (m)")
    ap.add_argument("--roi", choices=["camera", "all"], default=d.roi, help="camera = chỉ FOV camera; all = 360°")
    ap.add_argument("--seed", type=int, default=d.seed, help="seed cho RANSAC")
    ap.add_argument("--ground-method", choices=["plane", "zones"], default=d.ground_method,
                    help="plane = 1 mặt phẳng RANSAC; zones = mỗi vành 0-10/10-20/20+ m một mặt phẳng")
    ap.add_argument("--ransac-impl", choices=["numpy", "open3d"], default=d.ransac_impl,
                    help="numpy = RANSAC có seed, tái lập được; open3d = segment_plane (không tái lập được)")


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(**{k: getattr(args, k) for k in asdict(PipelineConfig()) if hasattr(args, k)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Topic D: voxel downsample -> RANSAC ground -> DBSCAN -> box. "
                                             "Lưu ảnh từng bước, overlay camera và BEV occupancy.")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--out-dir", default="results/figures")
    add_config_args(ap)
    args = ap.parse_args()
    cfg = config_from_args(args)

    dtype = dataset_type(args.data_root)
    fr = load_frame(args.data_root, args.frame)
    res = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, cfg)
    m = match_clusters_to_gt(res, fr["labels"], fr["calib"])

    out_dir = Path(args.out_dir)
    tag = f"{args.frame}_v{cfg.voxel_size}_d{cfg.distance_threshold}_e{cfg.eps}_{cfg.ground_method}"
    title = (f"{args.data_root} / {args.frame}  voxel={cfg.voxel_size} dist_thr={cfg.distance_threshold} "
             f"eps={cfg.eps} min_pts={cfg.min_points} ground={cfg.ground_method}  |  GT hit {m['n_gt_hit']}/{m['n_gt']}")
    plot_steps(res, fr["labels"], fr["calib"], dtype, title, out_dir / f"d_steps_{tag}.png")
    plot_camera_overlay(fr["image"], res, fr["calib"], out_dir / f"d_camera_{tag}.png", fr["labels"])
    plot_occupancy(res, dtype, fr["labels"], fr["calib"], out_dir / f"d_occupancy_{tag}.png")

    a, b, c, d = res["plane"]
    near = min((bx["nearest_m"] for bx in res["boxes"]), default=float("nan"))
    print(f"frame={args.frame} roi={len(res['xyz_roi'])} downsampled={len(res['xyz_ds'])} "
          f"ground={int(res['ground'].sum())} non_ground={len(res['obj_xyz'])}")
    print(f"plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0  (sensor cao {d:.2f} m so với mặt đất)")
    print(f"clusters={m['n_clusters']} nearest_obstacle={near:.2f} m  "
          f"GT hit={m['n_gt_hit']}/{m['n_gt']} cluster_TP={m['n_cluster_tp']}/{m['n_clusters']} "
          f"merged={m['merged']} split={m['split']}")
    print("ms: " + " ".join(f"{k}={v:.1f}" for k, v in res["ms"].items()))
    print(f"-> {out_dir}/d_steps_{tag}.png, d_camera_{tag}.png, d_occupancy_{tag}.png")


if __name__ == "__main__":
    main()
