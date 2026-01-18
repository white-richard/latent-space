#!/usr/bin/env python3
"""
Evaluate all checkpoints and pick best by Top-1 (k-NN).
"""
import argparse
import glob
import os
import csv
import traceback
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.checkpointing.model_checkpoint import load_dinov3_checkpoint
from dinov3.eval.knn import benchmark_launcher
from dinov3.run.init import job_context

def best_top1_from_results(results_dict):
    # Pick the first "... Top 1" metric found
    for k, v in results_dict.items():
        if k.endswith("Top 1"):
            return float(v)
    raise ValueError("No Top 1 metric found in results.")

def evaluate_checkpoint(ckpt_path, cfg_path, train_dataset_str, test_dataset_str, batch_size, output_base):
    basename = os.path.basename(ckpt_path).replace(".ckpt", "")
    tmp_weights = os.path.join("tmp", f"{basename}.pt")
    os.makedirs(os.path.dirname(tmp_weights), exist_ok=True)
    # Convert checkpoint to plain weights (.pt)
    model = load_dinov3_checkpoint(checkpoint_path=ckpt_path, cfg_path=cfg_path, save_path=tmp_weights)

    out_dir = os.path.join(output_base, basename)
    eval_args = {
        "model": {"config_file": cfg_path, "pretrained_weights": tmp_weights},
        "train": {"dataset": train_dataset_str, "batch_size": batch_size, "skip_first_nn": True},
        "eval": {"test_dataset": test_dataset_str, "batch_size": batch_size},
        "save_results": True,
        "output_dir": out_dir,
    }

    # job_context sets up logging and output dir similarly to main()
    with job_context(output_dir=out_dir):
        results = benchmark_launcher(eval_args)
    top1 = best_top1_from_results(results)
    return top1, results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--imagenet-root", required=True)
    p.add_argument("--imagenet-extra", required=True)
    p.add_argument("--split", default="VAL")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", default="knn_output_all")
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints_dir, "*.ckpt")))
    rows = []
    best = {"ckpt": None, "top1": -1.0, "results": None}

    train_ds = f"ImageNet:root={args.imagenet_root}:extra={args.imagenet_extra}:split={args.split}"
    test_ds = train_ds  # change if you want a different test split

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "ckpt_eval_summary.csv")
    for ckpt in ckpts:
        try:
            print(f"Evaluating {ckpt} ...")
            top1, results = evaluate_checkpoint(ckpt, args.cfg, train_ds, test_ds, args.batch_size, args.output_dir)
            print(f" -> Top1: {top1:.3f}")
            rows.append({"ckpt": ckpt, "top1": top1, "results": results})
            if top1 > best["top1"]:
                best = {"ckpt": ckpt, "top1": top1, "results": results}
        except Exception:
            print(f"Error evaluating {ckpt}:")
            traceback.print_exc()
            rows.append({"ckpt": ckpt, "top1": None, "results": None})

    # write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "top1"])
        for r in rows:
            writer.writerow([r["ckpt"], "" if r["top1"] is None else f"{r['top1']:.4f}"])

    print("=== SUMMARY ===")
    if best["ckpt"]:
        print(f"Best checkpoint: {best['ckpt']} with Top1={best['top1']:.4f}")
    else:
        print("No successful evaluations.")
    print(f"Detailed csv: {csv_path}")

if __name__ == "__main__":
    main()

"""
python scripts/knn_eval_all_ckpts.py \
    --checkpoints-dir /home/richiewhite/.code/latent-space/repos/dinov3_lvit_lightning/output/checkpoints \
    --cfg /home/richiewhite/.code/latent-space/repos/dinov3_lvit_lightning/configs/ssl_lvit_small.yaml \
    --imagenet-root /home/richiewhite/.code/datasets/imagenet \
    --imagenet-extra /home/richiewhite/.code/datasets/imagenet \
    --split VAL \
    --batch-size 64 \
    --output-dir knn_output_all
"""