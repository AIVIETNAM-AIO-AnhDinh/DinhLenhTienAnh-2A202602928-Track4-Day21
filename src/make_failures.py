"""Topic D, CP4: tạo ảnh failure case results/figures/fail_*.png.

    python -m src.make_failures

fail_01: 1 mặt phẳng RANSAC trên đường dốc/đỉnh dốc (KITTI 000061) -> mặt đường thành "vật cản" ở 6.4 m.
fail_02: vật thấp 0.15 m ở 10 m bị ground removal nuốt khi distance_threshold = 0.2 m (vật chèn, bán tổng hợp).
fail_03: eps DBSCAN lớn (0.8) gộp người đi bộ đứng sát tường vào cluster tường (KITTI 000011).
fail_04: nuScenes 32 beam: người đi bộ ở ~26-28 m chỉ có 5-6 điểm -> DBSCAN coi là nhiễu, bị bỏ sót.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.low_obstacle import evaluate_injection, plot_side_view  # noqa: E402
from src.obstacle_pipeline import (OBSTACLE_CLASSES, PipelineConfig, _cluster_colors, forward_axes,  # noqa: E402
                                   gt_corners_velo, match_clusters_to_gt, plot_camera_overlay,
                                   points_in_gt_box, run_pipeline)
from starter.datasets import list_frames, load_frame  # noqa: E402

OUT = Path("results/figures")
SCRATCH = OUT / "_tmp_overlay.png"


def _camera(fr, res) -> np.ndarray:
    vis = plot_camera_overlay(fr["image"], res, fr["calib"], SCRATCH, fr["labels"])
    SCRATCH.unlink(missing_ok=True)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def fail_01_slope() -> None:
    fr = load_frame("data/kitti_mini", "000061")
    panels = []
    for gm in ("plane", "zones"):
        res = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, PipelineConfig(ground_method=gm))
        near = min(res["boxes"], key=lambda b: b["nearest_m"])
        panels.append((gm, res, near))
    fig, axs = plt.subplots(3, 1, figsize=(12, 10.5), gridspec_kw={"height_ratios": [1, 1, 0.9]})
    for ax, (gm, res, near) in zip(axs[:2], panels):
        ax.imshow(_camera(fr, res))
        ax.set_title(f"ground_method={gm}: vật cản gần nhất = {near['nearest_m']:.1f} m "
                     f"({near['n_points']} điểm, cao {near['extent'][2]:.2f} m, rộng {near['extent'][1]:.1f} m)"
                     + ("  <- MẶT ĐƯỜNG bị coi là vật cản" if near["flat"] else ""), fontsize=10)
        ax.axis("off")
    # mặt cắt dọc theo hướng xe chạy: độ cao điểm so với mặt phẳng toàn cục
    res = panels[0][1]
    xyz = res["xyz_ds"]
    lane = np.abs(xyz[:, 1]) < 2.0
    h = xyz @ res["plane"][:3] + res["plane"][3]
    ax = axs[2]
    ax.scatter(xyz[lane & res["ground"], 0], h[lane & res["ground"]], s=3, c="0.6", label="ground (1 mặt phẳng)")
    ax.scatter(xyz[lane & ~res["ground"], 0], h[lane & ~res["ground"]], s=3, c="tab:red", label="non-ground")
    ax.axhspan(-0.2, 0.2, color="tab:green", alpha=0.15, label="±distance_threshold 0.2 m")
    ax.set_xlim(0, 20)
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel("x phía trước (m), làn |y| < 2 m")
    ax.set_ylabel("độ cao so với mặt phẳng RANSAC (m)")
    ax.set_title("Mặt cắt dọc làn xe: mặt đường ở 6–7 m cao hơn mặt phẳng RANSAC ~0.2 m, ở 12–15 m thấp hơn 0.2–0.45 m "
                 "-> mặt đường không phẳng, 1 mặt phẳng không ôm được", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.suptitle("fail_01 — KITTI 000061: giả định mặt đất phẳng sai trên đường dốc (lớp Geometry/Preprocess)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fail_01_single_plane_slope_000061.png", dpi=100)
    plt.close(fig)


def fail_02_low_obstacle() -> None:
    fr = load_frame("data/kitti_mini", "000012")
    base = PipelineConfig(roi="all")
    plane = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, base)["plane"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.4))
    tmp = []
    for dt in (0.1, 0.2):
        r = evaluate_injection(fr, replace(base, distance_threshold=dt), plane, 10.0, 0.0, 0.15, seed=0)
        p = OUT / f"_tmp_side_{dt}.png"
        verdict = "THẤY" if r["detected"] else "MẤT"
        plot_side_view(r["_res"], r["_box"], f"distance_threshold={dt} m -> {verdict} "
                       f"({r['n_kept_after_ground']}/{r['n_object_points_downsampled']} điểm của vật còn lại)", p)
        tmp.append(p)
    for ax, p in zip(axs, tmp):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        p.unlink()
    fig.suptitle("fail_02 — vật cao 0.15 m (pallet) chèn ở 10 m trên KITTI 000012: ngưỡng RANSAC 0.2 m nuốt cả vật "
                 "(lớp Preprocess)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fail_02_low_obstacle_swallowed.png", dpi=110)
    plt.close(fig)


def _bev_zoom(ax, fr, res, center, half, title):
    f, l, sgn = forward_axes(dataset_type_of(fr))
    o, lab = res["obj_xyz"], res["labels"]
    ax.scatter(sgn * o[:, l], o[:, f], c=_cluster_colors(lab), s=6, linewidths=0)
    for obj in fr["labels"]:
        if obj.type in OBSTACLE_CLASSES:
            c = gt_corners_velo(obj, fr["calib"])[[0, 1, 2, 3, 0]]
            ax.plot(sgn * c[:, l], c[:, f], color="lime" if obj.type != "Pedestrian" else "k", lw=1)
    ax.set_xlim(center[0] + half, center[0] - half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


def dataset_type_of(fr) -> str:
    return "nuscenes" if "timestamp_lidar_us" in fr else "kitti"


def fail_03_merge() -> None:
    """eps lớn: người đi bộ đứng sát tường bị gộp vào cluster tường (KITTI 000011).
    Tìm người có cluster 'sạch' ở eps=0.5 nhưng bị nuốt vào cluster lớn ở eps=0.8."""
    f, eps_big = "000011", 0.8
    fr = load_frame("data/kitti_mini", f)
    runs = {e: run_pipeline(fr["points"], fr["calib"], fr["image"].shape, PipelineConfig(eps=e)) for e in (0.5, eps_big)}

    def owner(res, obj):
        m = points_in_gt_box(res["obj_xyz"], obj, fr["calib"], 0.25) & (res["labels"] >= 0)
        if m.sum() < 3:
            return None
        k = int(np.bincount(res["labels"][m]).argmax())
        return k, float(m[res["labels"] == k].sum() / (res["labels"] == k).sum())

    target = None
    for obj in fr["labels"]:
        if obj.type != "Pedestrian":
            continue
        a, b = owner(runs[0.5], obj), owner(runs[eps_big], obj)
        if a and b and a[1] > 0.6 and b[1] < 0.3:
            target = (obj, a, b)
            break
    if target is None:
        print("fail_03: không tìm thấy trường hợp, bỏ qua")
        return
    obj, (k5, p5), (kb, pb) = target
    c = gt_corners_velo(obj, fr["calib"]).mean(0)
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    for i, (e, k, pur) in enumerate(((0.5, k5, p5), (eps_big, kb, pb))):
        res = runs[e]
        b = res["boxes"][k]
        ax = fig.add_subplot(gs[0, i])
        _bev_zoom(ax, fr, res, (c[1], c[0]), 9.0,
                  f"eps={e} m: cluster chứa người đi bộ có {b['n_points']} điểm, dài {b['extent'][:2].max():.1f} m, "
                  f"{100 * pur:.0f}% điểm thuộc người (viền đen)")
        sel = res["labels"] == k
        ax.scatter(res["obj_xyz"][sel, 1], res["obj_xyz"][sel, 0], s=14, facecolors="none", edgecolors="r", lw=0.4)
    ax = fig.add_subplot(gs[1, :])
    vis = _camera(fr, runs[eps_big])
    x1, y1, x2, y2 = obj.bbox.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
    cv2.putText(vis, "Pedestrian -> bi gop vao tuong", (x1 - 60, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)
    ax.imshow(vis)
    ax.axis("off")
    ax.set_title(f"camera, eps={eps_big}: người (khung đỏ) cùng màu với cluster tường bên trái", fontsize=9)
    fig.suptitle(f"fail_03 — KITTI {f}: eps={eps_big} nối người đứng sát tường thành 1 cluster tường "
                 f"(lớp Preprocess; recall theo GT vẫn tính là 'hit' -> lớp Metric)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"fail_03_pedestrian_merged_into_wall_eps{eps_big}_{f}.png", dpi=100)
    plt.close(fig)


def fail_04_sparse() -> None:
    """nuScenes: GT người đi bộ xa có ít điểm bị bỏ sót. Chọn frame có nhiều GT bị miss nhất."""
    root = "data/nuscenes_mini_subset"
    cfg = PipelineConfig()
    best = None
    for f in list_frames(root)[:40]:
        fr = load_frame(root, f)
        res = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, cfg)
        m = match_clusters_to_gt(res, fr["labels"], fr["calib"])
        miss = [g for g in m["per_gt"] if not g["hit"] and g["type"] == "Pedestrian"]
        if best is None or len(miss) > len(best[2]):
            best = (f, fr, miss, res, m)
    f, fr, miss, res, m = best
    vis = _camera(fr, res)
    # tô đỏ 2D box của GT người bị miss
    gts = [o for o in fr["labels"] if o.type in OBSTACLE_CLASSES
           and points_in_gt_box(res["xyz_roi"], o, fr["calib"]).sum() >= 5]
    for o, g in zip(gts, m["per_gt"]):
        if not g["hit"]:
            x1, y1, x2, y2 = o.bbox.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(vis, f"MISS {g['type']} {g['range_m']:.0f}m {g['n_raw_points']}pts", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(vis)
    ax.axis("off")
    pts = [g["n_raw_points"] for g in miss]
    rngs = [g["range_m"] for g in miss]
    ax.set_title(f"fail_04 — nuScenes {f} (32 beam): {len(miss)} người đi bộ ở {min(rngs):.0f}–{max(rngs):.0f} m "
                 f"chỉ có {min(pts)}–{max(pts)} điểm LiDAR "
                 f"-> DBSCAN (eps={cfg.eps}, min_points={cfg.min_points}) coi là nhiễu (khung đỏ). "
                 f"GT hit {m['n_gt_hit']}/{m['n_gt']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"fail_04_sparse_far_pedestrian_{f}.png", dpi=100)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fail_01_slope, fail_02_low_obstacle, fail_03_merge, fail_04_sparse):
        fn()
        print(f"done {fn.__name__}")
    print(sorted(p.name for p in OUT.glob("fail_*.png")))


if __name__ == "__main__":
    main()
