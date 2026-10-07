"""Topic D, CP3: sweep tham số của pipeline vật cản, ghi CSV + vẽ biểu đồ.

    python -m src.sweep --experiment voxel_dist      # voxel_size x distance_threshold trên 20 frame KITTI
    python -m src.sweep --experiment eps             # eps của DBSCAN
    python -m src.sweep --experiment ground          # 1 mặt phẳng vs zones, numpy vs open3d (B1)
    python -m src.sweep --experiment dataset         # cùng cấu hình trên KITTI và nuScenes (B5)
    python -m src.sweep --experiment all

Mỗi cấu hình: 1 lần chạy warmup bị bỏ, sau đó lặp --repeats lần trên mọi frame để đo latency p50/p95.
Metric chất lượng lấy từ lần lặp đầu (pipeline tất định với seed cố định, các lần lặp cho cùng kết quả).
"""
from __future__ import annotations

import argparse
import platform
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.obstacle_pipeline import PipelineConfig, match_clusters_to_gt, points_in_gt_box, run_pipeline  # noqa: E402
from starter.datasets import list_frames, load_frame  # noqa: E402

SMALL_CLASSES = {"Pedestrian", "Person_sitting", "Cyclist", "Bicycle", "Motorcycle", "TrafficCone", "Barrier"}
RANGE_BINS = [(0, 10), (10, 20), (20, 30), (30, 40)]


def load_all(data_root: str, frames: list[str]) -> list[dict]:
    return [load_frame(data_root, f) for f in frames]


def low_part_kept(res: dict, fr: dict, cut: float = 0.3) -> tuple[int, int]:
    """Đếm điểm (đã downsample) nằm trong box GT người/xe đạp VÀ thấp hơn `cut` m so với mặt đất,
    và bao nhiêu điểm trong số đó còn sống sau khi tách mặt đất.
    Đây là proxy cho 'vật thấp sát đất' (chân người, bánh xe đạp, vật ngồi)."""
    xyz, ground, h = res["xyz_ds"], res["ground"], res["height"]
    total = kept = 0
    for obj in fr["labels"]:
        if obj.type not in SMALL_CLASSES:
            continue
        m = points_in_gt_box(xyz, obj, fr["calib"]) & (h < cut) & (h > 0.02)
        total += int(m.sum())
        kept += int((m & ~ground).sum())
    return kept, total


def evaluate(cfg: PipelineConfig, frames: list[dict], repeats: int) -> tuple[dict, list[dict]]:
    """Chạy cfg trên mọi frame. Trả về (1 dòng tổng hợp, danh sách GT từng frame)."""
    run_pipeline(frames[0]["points"], frames[0]["calib"], frames[0]["image"].shape, cfg)  # warmup, bỏ
    lat, core = [], []
    agg = dict(n_gt=0, n_gt_hit=0, n_clusters=0, n_cluster_tp=0, merged=0, split=0, low_kept=0, low_total=0,
               flat=0, nearest_flat=0)
    per_gt, nearest, extents = [], [], []
    for rep in range(repeats):
        for fr in frames:
            t0 = time.perf_counter()
            res = run_pipeline(fr["points"], fr["calib"], fr["image"].shape, cfg)
            lat.append((time.perf_counter() - t0) * 1e3)
            core.append(res["ms"]["voxel"] + res["ms"]["ground"] + res["ms"]["dbscan"])
            if rep > 0:
                continue
            m = match_clusters_to_gt(res, fr["labels"], fr["calib"])
            for k in ("n_gt", "n_gt_hit", "n_clusters", "n_cluster_tp", "merged", "split"):
                agg[k] += m[k]
            kept, total = low_part_kept(res, fr)
            agg["low_kept"] += kept
            agg["low_total"] += total
            per_gt += [g | {"frame": fr["frame_id"]} for g in m["per_gt"]]
            nearest.append(min((b["nearest_m"] for b in res["boxes"]), default=np.nan))
            agg["flat"] += sum(b["flat"] for b in res["boxes"])
            agg["nearest_flat"] += bool(res["boxes"]) and min(res["boxes"], key=lambda b: b["nearest_m"])["flat"]
            extents += [float(np.prod(b["extent"][:2])) for b in res["boxes"]]
    small = [g for g in per_gt if g["type"] in SMALL_CLASSES]
    big = [g for g in per_gt if g["type"] not in SMALL_CLASSES]
    n = len(frames)
    row = {
        "frames": n,
        "clusters_per_frame": agg["n_clusters"] / n,
        "median_cluster_bev_area_m2": float(np.median(extents)) if extents else np.nan,
        "nearest_obstacle_median_m": float(np.nanmedian(nearest)),
        "gt_total": agg["n_gt"],
        "recall_all": agg["n_gt_hit"] / max(agg["n_gt"], 1),
        "recall_small": sum(g["hit"] for g in small) / max(len(small), 1),
        "recall_vehicle": sum(g["hit"] for g in big) / max(len(big), 1),
        "cluster_precision_vs_gt": agg["n_cluster_tp"] / max(agg["n_clusters"], 1),
        "merged_clusters": agg["merged"],
        "split_gt": agg["split"],
        "low_pts_kept_pct": 100 * agg["low_kept"] / max(agg["low_total"], 1),
        "flat_clusters_per_frame": agg["flat"] / n,
        "frames_nearest_is_flat": int(agg["nearest_flat"]),
        "latency_total_p50_ms": float(np.percentile(lat, 50)),
        "latency_total_p95_ms": float(np.percentile(lat, 95)),
        "latency_core_p50_ms": float(np.percentile(core, 50)),
        "latency_core_p95_ms": float(np.percentile(core, 95)),
        "latency_samples": len(lat),
    }
    return row, per_gt


def run_grid(name: str, cfgs: list[PipelineConfig], frames: list[dict], repeats: int, out_dir: Path,
             extra_cols: dict | None = None) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(cfgs):
        row, _ = evaluate(cfg, frames, repeats)
        c = asdict(cfg)
        rows.append({k: c[k] for k in ("voxel_size", "distance_threshold", "eps", "min_points",
                                       "ground_method", "ransac_impl")}
                    | (extra_cols[i] if extra_cols else {}) | row)
        r = rows[-1]
        print(f"[{name}] {i + 1}/{len(cfgs)} voxel={cfg.voxel_size} dist={cfg.distance_threshold} eps={cfg.eps} "
              f"ground={cfg.ground_method}/{cfg.ransac_impl}: clusters/frame={r['clusters_per_frame']:.1f} "
              f"recall={r['recall_all']:.3f} small={r['recall_small']:.3f} low_kept={r['low_pts_kept_pct']:.1f}% "
              f"p50={r['latency_total_p50_ms']:.1f}ms")
    df = pd.DataFrame(rows)
    out = out_dir / f"{name}.csv"
    df.round(4).to_csv(out, index=False)
    print(f"-> {out}")
    return df


# ---------------------------------------------------------------- các thí nghiệm

def exp_voxel_dist(args, base: PipelineConfig) -> None:
    frames = load_all(args.kitti, list_frames(args.kitti))
    voxels, dists = [0.05, 0.1, 0.2, 0.3], [0.05, 0.1, 0.2, 0.3, 0.5]
    cfgs = [replace(base, voxel_size=v, distance_threshold=d) for v in voxels for d in dists]
    df = run_grid("sweep_voxel_dist", cfgs, frames, args.repeats, args.out_dir)

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
    for v in voxels:
        s = df[df.voxel_size == v]
        axs[0].plot(s.distance_threshold, s.recall_small, "o-", label=f"voxel {v} m")
        axs[1].plot(s.distance_threshold, s.low_pts_kept_pct, "o-", label=f"voxel {v} m")
        axs[2].plot(s.distance_threshold, s.clusters_per_frame, "o-", label=f"voxel {v} m")
        axs[3].plot(s.distance_threshold, s.latency_total_p50_ms, "o-", label=f"voxel {v} m")
    for ax, t in zip(axs, ["Recall người/xe đạp (GT có >=5 điểm)",
                           "% điểm thấp (<0.3 m) của người/xe đạp còn lại sau tách mặt đất",
                           "Số cluster / frame", "Latency p50 (ms, toàn pipeline)"]):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("distance_threshold RANSAC (m)")
        ax.grid(alpha=0.3)
    axs[0].set_ylim(0.8, 1.01)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"KITTI mini, 20 frame, eps={base.eps}, min_points={base.min_points}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "sweep_voxel_dist.png", dpi=110)
    plt.close(fig)


def exp_eps(args, base: PipelineConfig) -> None:
    frames = load_all(args.kitti, list_frames(args.kitti))
    eps_list = [0.2, 0.3, 0.5, 0.8, 1.2]
    df = run_grid("sweep_eps", [replace(base, eps=e) for e in eps_list], frames, args.repeats, args.out_dir)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(df.eps, df.recall_small, "o-", label="recall người/xe đạp")
    ax1.plot(df.eps, df.cluster_precision_vs_gt, "s-", label="cluster precision (so với GT)")
    ax1.set_xlabel("eps DBSCAN (m)")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(df.eps - 0.03, df.merged_clusters, width=0.06, color="tab:red", alpha=0.5, label="cluster gộp >=2 GT")
    ax2.bar(df.eps + 0.03, df.split_gt, width=0.06, color="tab:gray", alpha=0.5, label="GT bị tách >=2 cluster")
    ax2.set_ylabel("số lần (tổng 20 frame)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.set_title(f"KITTI mini: sweep eps (voxel={base.voxel_size}, dist_thr={base.distance_threshold})", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "sweep_eps.png", dpi=110)
    plt.close(fig)


def exp_ground(args, base: PipelineConfig) -> None:
    frames = load_all(args.kitti, list_frames(args.kitti))
    cfgs = [replace(base, ground_method="plane", ransac_impl="numpy"),
            replace(base, ground_method="plane", ransac_impl="open3d"),
            replace(base, ground_method="zones", ransac_impl="numpy")]
    df = run_grid("compare_ground_method", cfgs, frames, args.repeats, args.out_dir)
    # tính tái lập: chạy 5 lần, đếm số frame mà số điểm ground thay đổi giữa các lần chạy
    unstable = []
    for cfg in cfgs:
        n = 0
        for fr in frames:
            counts = {int(run_pipeline(fr["points"], fr["calib"], fr["image"].shape, cfg)["ground"].sum())
                      for _ in range(5)}
            n += len(counts) > 1
        unstable.append(n)
    df.insert(6, "frames_not_reproducible_5_runs", unstable)
    df.round(4).to_csv(args.out_dir / "compare_ground_method.csv", index=False)
    print(df[["ground_method", "ransac_impl", "frames_not_reproducible_5_runs", "recall_all",
              "nearest_obstacle_median_m", "latency_total_p50_ms"]])


def exp_dataset(args, base: PipelineConfig) -> None:
    """Cùng cấu hình trên KITTI (64 beam) và nuScenes (32 beam): recall theo khoảng cách."""
    rows = []
    for name, root, step in [("kitti_mini", args.kitti, 1), ("nuscenes_mini_subset", args.nusc, 2)]:
        frames = load_all(root, list_frames(root)[::step])
        for eps in (0.5, 1.0):
            cfg = replace(base, eps=eps)
            row, per_gt = evaluate(cfg, frames, args.repeats)
            for lo, hi in RANGE_BINS:
                g = [x for x in per_gt if lo <= x["range_m"] < hi]
                rows.append({"dataset": name, "eps": eps, "frames": len(frames), "range_bin_m": f"{lo}-{hi}",
                             "gt": len(g), "recall": sum(x["hit"] for x in g) / len(g) if g else np.nan,
                             "median_raw_points_per_gt": float(np.median([x["n_raw_points"] for x in g])) if g else np.nan,
                             "clusters_per_frame": row["clusters_per_frame"],
                             "latency_total_p50_ms": row["latency_total_p50_ms"]})
            print(f"[dataset] {name} eps={eps}: recall={row['recall_all']:.3f} p50={row['latency_total_p50_ms']:.1f}ms")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(args.out_dir / "compare_dataset_range.csv", index=False)
    print(df.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(RANGE_BINS))
    for i, ((ds, eps), s) in enumerate(df.groupby(["dataset", "eps"], sort=False)):
        ax.bar(x + (i - 1.5) * 0.2, s.recall, width=0.2, label=f"{ds}, eps={eps}")
    ax.set_xticks(x, [f"{lo}-{hi} m" for lo, hi in RANGE_BINS])
    ax.set_ylabel("recall (GT có >=5 điểm LiDAR)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title(f"Recall theo khoảng cách: KITTI 64 beam vs nuScenes 32 beam (voxel={base.voxel_size}, "
                 f"dist_thr={base.distance_threshold})", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out_dir / "figures" / "compare_dataset_range.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep tham số pipeline vật cản (topic D), ghi CSV + biểu đồ vào results/")
    ap.add_argument("--experiment", choices=["voxel_dist", "eps", "ground", "dataset", "all"], default="all")
    ap.add_argument("--kitti", default="data/kitti_mini")
    ap.add_argument("--nusc", default="data/nuscenes_mini_subset")
    ap.add_argument("--repeats", type=int, default=20, help="số lần lặp mỗi cấu hình để đo latency (sau warmup)")
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)
    print(f"CPU: {platform.processor() or platform.machine()} | {platform.platform()}")

    base = PipelineConfig()
    exps = {"voxel_dist": exp_voxel_dist, "eps": exp_eps, "ground": exp_ground, "dataset": exp_dataset}
    for name, fn in exps.items():
        if args.experiment in (name, "all"):
            fn(args, base)


if __name__ == "__main__":
    main()
