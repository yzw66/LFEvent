"""Train LFEvent on NED/CrisisMMD S1/S2."""
# 训练入口：主进程调度任务；每个子进程调用 engine.worker 完成一组数据集/设置的训练。
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from lfevent.engine import RECIPE, REPO, SOURCES, worker
from lfevent.utils import read, sha256, write_json


def main(args):
    args.output, args.data_root = args.output.resolve(), args.data_root.resolve()
    if args.epochs < 1 or args.jobs < 1 or args.threads < 1:
        raise ValueError("epochs, jobs and threads must be positive")
    if args.worker:
        # 内部子进程入口；日常使用无需手动传 --worker。
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    # 记录本次任务的数据范围、参数及源码哈希；--resume 要求它们与已有输出一致。
    protocol = {"datasets": args.datasets, "settings": args.settings, "epochs": args.epochs,
                "seed": 230, "batch_size": 64, "recipe": RECIPE, "smoke_only": args.smoke,
                "data_root": str(args.data_root),
                "source_sha256": {name: sha256(REPO / name) for name in SOURCES}}
    path = args.output / "protocol.json"
    if path.exists() and (not args.resume or read(path) != protocol):
        raise ValueError("Output exists or configuration differs; use matching --resume or a new --output")
    write_json(path, protocol)
    snapshot = args.output / "source"
    # 保存首次运行时的代码快照，续训时保留原快照，方便追溯结果来自哪个版本。
    if not snapshot.exists():
        for name in SOURCES:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)

    pending = [(dataset, setting) for dataset in args.datasets for setting in args.settings]
    # 默认组合为 NED/S1、NED/S2、CrisisMMD/S1、CrisisMMD/S2；--jobs 控制同时运行的数量。
    active, failures = [], []
    while pending or active:
        while pending and len(active) < args.jobs:
            dataset, setting = pending.pop(0)
            case = args.output / f"{dataset}_{setting}"
            case.mkdir(exist_ok=True)
            command = [sys.executable, str(REPO / "train.py"), "--worker", dataset, setting,
                       "--output", str(args.output), "--data-root", str(args.data_root),
                       "--epochs", str(args.epochs), "--threads", str(args.threads), "--device", args.device]
            if args.resume:
                command.append("--resume")
            if args.smoke:
                command.append("--smoke")
            # 子进程沿用当前 Python 环境，各组日志和 checkpoint 分目录保存。
            stream = (case / "train.log").open("a", encoding="utf-8")
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            active.append((dataset, setting, process, stream))
            print(f"Started {dataset}/{setting}: pid={process.pid}", flush=True)
        for item in active[:]:
            dataset, setting, process, stream = item
            if process.poll() is None:
                continue
            active.remove(item)
            stream.close()
            print(f"Finished {dataset}/{setting}: exit={process.returncode}", flush=True)
            if process.returncode:
                failures.append({"dataset": dataset, "setting": setting, "exit_code": process.returncode})
        # 只汇总已生成 result.json 的完成任务，并将运行中/失败任务写入 progress.json。
        results = [read(path) for path in sorted(args.output.glob("*/result.json"))]
        write_json(args.output / "results.json", results)
        if results:
            import pandas as pd
            pd.DataFrame(results).to_csv(args.output / "results.csv", index=False)
        write_json(args.output / "progress.json", {
            "running": [(d, s, p.pid) for d, s, p, _ in active], "completed": len(results), "failures": failures})
        if pending or active:
            time.sleep(1)
    if failures:
        raise RuntimeError(f"Training failures: {failures}; see each case's train.log")


def parse_args():
    # 不传参数即跑两套数据的 S1/S2 全量数据、30 轮；--smoke 用于快速检查整个流程。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO / "data/Event_data")
    parser.add_argument("--output", type=Path, default=REPO / "runs")
    parser.add_argument("--datasets", nargs="+", choices=["NED", "CrisisMMD"], default=["NED", "CrisisMMD"])
    parser.add_argument("--settings", nargs="+", choices=["S1", "S2"], default=["S1", "S2"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=1, help="Number of concurrent training processes")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="Use first 257 training and 129 validation/test rows")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
