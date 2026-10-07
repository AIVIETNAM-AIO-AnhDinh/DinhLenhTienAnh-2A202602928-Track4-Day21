"""Topic D: vật thấp sát đất (pallet, xe đẩy, người ngồi) có bị ground removal "nuốt" không?

Chèn một vật hình hộp (mặc định 0.8 x 1.2 m, cao h) vào point cloud KITTI thật, đặt trên mặt đất đã fit,
ở khoảng cách D phía trước. Điểm của vật được sinh bằng ray casting theo mẫu tia gần đúng của
Velodyne HDL-64E (sensor của KITTI), có che khuất (điểm thật phía sau vật bị xoá) và nhiễu range 2 cm.
Sau đó chạy pipeline với nhiều distance_threshold và kiểm tra vật có thành cluster không.

    python -m src.low_obstacle                       # sweep mặc định, ghi results/low_obstacle_injection.csv
    python -m src.low_obstacle --frame 000012 --heights 0.1 0.2 0.3 --distances 5 10 20

Đây là dữ liệu BÁN TỔNG HỢP (real background + vật giả lập): dùng để so sánh tương đối giữa các tham số,
không thay cho đo trên vật thật.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.obstacle_pipeline import PipelineConfig, run_pipeline  # noqa: E402
from starter.datasets import load_frame  # noqa: E402

# HDL-64E: 32 tia trên từ +2.0° xuống -8.33° (~1/3°), 32 tia dưới từ -8.83° xuống -24.33° (~1/2°).
# Độ phân giải ngang ~0.17° ở 10 Hz. Nguồn: Velodyne HDL-64E S2 datasheet (giá trị gần đúng).
HDL64_ELEV_DEG = np.r_[np.linspace(2.0, -8.33, 32), np.linspace(-8.83, -24.33, 32)]
HDL64_AZ_STEP_DEG = 0.17


def ray_box(dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Giao tia từ gốc toạ độ theo hướng dirs (N, 3) với hộp AABB (slab method). Trả về (hit, t_near)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1, t2 = bmin * inv, bmax * inv
    tn = np.nanmax(np.minimum(t1, t2), axis=1)
    tf = np.nanmin(np.maximum(t1, t2), axis=1)
    return (tf >= tn) & (tn > 0), tn


def inject_box(points: np.ndarray, plane: np.ndarray, dist: float, y0: float, height: float,
               size_xy=(0.8, 1.2), noise: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về (point cloud mới, mask điểm của vật, (bmin, bmax) của vật)."""
    a, b, c, d = plane
    z_ground = -(a * dist + b * y0 + d) / c
    bmin = np.array([dist - size_xy[0] / 2, y0 - size_xy[1] / 2, z_ground])
    bmax = np.array([dist + size_xy[0] / 2, y0 + size_xy[1] / 2, z_ground + height])

    # che khuất: bỏ điểm thật mà tia tới nó đi xuyên qua vật
    xyz = points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    hit, tn = ray_box(xyz / np.maximum(r, 1e-9)[:, None], bmin, bmax)
    kept = points[~(hit & (tn < r))]

    # sinh điểm của vật bằng ray casting
    corners = np.array([[x, y] for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1])])
    az = np.degrees(np.arctan2(corners[:, 1], corners[:, 0]))
    az_grid = np.arange(np.floor(az.min() / HDL64_AZ_STEP_DEG), np.ceil(az.max() / HDL64_AZ_STEP_DEG) + 1)
    az_grid = np.deg2rad(az_grid * HDL64_AZ_STEP_DEG)
    el, azm = np.meshgrid(np.deg2rad(HDL64_ELEV_DEG), az_grid, indexing="ij")
    dirs = np.stack([np.cos(el) * np.cos(azm), np.cos(el) * np.sin(azm), np.sin(el)], -1).reshape(-1, 3)
    hit, tn = ray_box(dirs, bmin, bmax)
    rng = np.random.default_rng(seed)
    t = tn[hit] + rng.normal(0, noise, hit.sum())
    obj = np.c_[dirs[hit] * t[:, None], np.full(hit.sum(), 0.3)].astype(np.float32)

    out = np.vstack([kept, obj])
    is_obj = np.r_[np.zeros(len(kept), bool), np.ones(len(obj), bool)]
    return out, is_obj, np.stack([bmin, bmax])


def evaluate_injection(fr: dict, cfg: PipelineConfig, plane: np.ndarray, dist: float, y0: float, height: float,
                       seed: int, hit_points: int = 3) -> dict:
    pts, is_obj, (bmin, bmax) = inject_box(fr["points"], plane, dist, y0, height, seed=seed)
    res = run_pipeline(pts, fr["calib"], fr["image"].shape, cfg)
    m = 0.1
    in_box_ds = ((res["xyz_ds"] >= bmin - m) & (res["xyz_ds"] <= bmax + m)).all(1)
    in_box_obj = ((res["obj_xyz"] >= bmin - m) & (res["obj_xyz"] <= bmax + m)).all(1)
    lab = res["labels"][in_box_obj]
    best = int(np.bincount(lab[lab >= 0]).max()) if (lab >= 0).any() else 0
    return {
        "n_object_points_raw": int(is_obj.sum()),
        "n_object_points_downsampled": int(in_box_ds.sum()),
        "n_kept_after_ground": int(in_box_obj.sum()),
        "best_cluster_points": best,
        "detected": best >= hit_points,
        "_res": res, "_box": (bmin, bmax),
    }


def pick_free_lateral(fr: dict, cfg: PipelineConfig, distances, candidates=(0.0, -1.5, 1.5, -3.0, 3.0)) -> float:
    """Chọn vị trí ngang y0 sao cho hành lang phía trước (tới khoảng cách xa nhất) không có vật thật nào."""
    res = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, cfg)
    obj = res["obj_xyz"]
    for y0 in candidates:
        corridor = (np.abs(obj[:, 1] - y0) < 1.5) & (obj[:, 0] > 0) & (obj[:, 0] < max(distances) + 2)
        if not corridor.any():
            return y0
    raise SystemExit("Không tìm được hành lang trống phía trước; thử --frame khác")


def plot_detection(df: pd.DataFrame, out: Path) -> None:
    dts = sorted(df.distance_threshold.unique())
    fig, axs = plt.subplots(1, len(dts), figsize=(3.6 * len(dts), 3.6), sharey=True)
    for ax, dt in zip(np.atleast_1d(axs), dts):
        s = df[df.distance_threshold == dt]
        piv = s.pivot(index="height_m", columns="distance_m", values="best_cluster_points")
        det = s.pivot(index="height_m", columns="distance_m", values="detected")
        ax.imshow(det.values.astype(float), cmap="RdYlGn", vmin=0, vmax=1, origin="lower", aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, int(piv.values[i, j]), ha="center", va="center", fontsize=8)
        ax.set_xticks(range(piv.shape[1]), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{r:g}" for r in piv.index])
        ax.set_xlabel("khoảng cách (m)")
        ax.set_title(f"distance_threshold = {dt} m", fontsize=9)
    np.atleast_1d(axs)[0].set_ylabel("chiều cao vật (m)")
    fig.suptitle("Vật thấp 0.8x1.2 m chèn vào KITTI: xanh = thành cluster (>=3 điểm), đỏ = bị mất. "
                 "Số = điểm của cluster lớn nhất trên vật", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_side_view(res: dict, box, title: str, out: Path) -> None:
    """Mặt cắt dọc (x-z) quanh vật: điểm bị gán ground (xám), điểm còn lại sau tách mặt đất (đỏ)."""
    bmin, bmax = box
    sel = lambda p: (np.abs(p[:, 1] - (bmin[1] + bmax[1]) / 2) < 1.0) & (p[:, 0] > bmin[0] - 4) & (p[:, 0] < bmax[0] + 4)
    g = res["xyz_ds"][res["ground"]]
    o = res["obj_xyz"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.scatter(g[sel(g), 0], g[sel(g), 2], s=6, c="0.6", label="bị gán ground")
    ax.scatter(o[sel(o), 0], o[sel(o), 2], s=10, c="tab:red", label="non-ground (vào DBSCAN)")
    ax.add_patch(plt.Rectangle((bmin[0], bmin[2]), bmax[0] - bmin[0], bmax[2] - bmin[2], fill=False, ec="b", lw=1.2,
                               label="vật chèn vào"))
    ax.set_xlabel("x phía trước (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Chèn vật thấp vào point cloud KITTI, đo distance_threshold nào làm mất vật")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000012", help="nên chọn frame có đường trống phía trước")
    ap.add_argument("--heights", type=float, nargs="+", default=[0.1, 0.15, 0.2, 0.3, 0.5])
    ap.add_argument("--distances", type=float, nargs="+", default=[5, 10, 15, 20, 30])
    ap.add_argument("--dist-thresholds", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--voxel-size", type=float, default=PipelineConfig.voxel_size)
    ap.add_argument("--eps", type=float, default=PipelineConfig.eps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/low_obstacle_injection.csv"))
    args = ap.parse_args()

    fr = load_frame(args.data_root, args.frame)
    # roi="all": FOV camera KITTI chỉ thấy từ ~0.36 m trên mặt đất ở 5 m, nên ROI camera sẽ tự che vật thấp ở gần.
    # Dùng 360° để cô lập ảnh hưởng của ground removal.
    base = PipelineConfig(voxel_size=args.voxel_size, eps=args.eps, seed=args.seed, roi="all")
    plane = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, base)["plane"]
    y0 = pick_free_lateral(fr, base, args.distances)
    print(f"frame={args.frame} ground plane={np.round(plane, 3)} y0={y0} m")

    rows = []
    for dt in args.dist_thresholds:
        cfg = replace(base, distance_threshold=dt)
        for h in args.heights:
            for D in args.distances:
                r = evaluate_injection(fr, cfg, plane, D, y0, h, args.seed)
                rows.append({"frame": args.frame, "distance_threshold": dt, "voxel_size": cfg.voxel_size,
                             "eps": cfg.eps, "height_m": h, "distance_m": D, "lateral_y_m": y0}
                            | {k: v for k, v in r.items() if not k.startswith("_")})
        s = pd.DataFrame(rows)
        s = s[s.distance_threshold == dt]
        print(f"dist_thr={dt}: phát hiện {int(s.detected.sum())}/{len(s)}; "
              f"vật thấp nhất luôn thấy ở mọi khoảng cách: "
              f"{s.groupby('height_m').detected.all().pipe(lambda x: x[x].index.min())} m")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    fig_dir = args.out.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_detection(df, fig_dir / "low_obstacle_detection.png")
    print(f"-> {args.out}, {fig_dir / 'low_obstacle_detection.png'}")

    # ảnh failure: vật 0.15 m ở 10 m, so sánh distance_threshold 0.1 và 0.2
    for i, dt in enumerate((0.1, 0.2)):
        r = evaluate_injection(fr, replace(base, distance_threshold=dt), plane, 10.0, y0, 0.15, args.seed)
        verdict = "THẤY" if r["detected"] else "MẤT"
        plot_side_view(r["_res"], r["_box"],
                       f"Vật cao 0.15 m ở 10 m, distance_threshold={dt} m -> {verdict} "
                       f"({r['n_kept_after_ground']}/{r['n_object_points_downsampled']} điểm sống sót)",
                       fig_dir / f"low_obstacle_side_dt{dt}.png")


if __name__ == "__main__":
    main()
