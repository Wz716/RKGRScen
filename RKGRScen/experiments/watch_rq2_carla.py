import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

VARIANTS = ("full", "without_community", "without_expansion", "without_constraint", "without_semantic_summaries")
DEFAULT_SHARED_MANIFEST = "/home/zxy/apollo/data/test/point2/RKGRScen/data/evaluation/rq1_rq2_20260714_shared/manifest.json"

def log(message: str, log_path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

def matching_processes(required: List[str], excluded: Optional[List[str]] = None) -> List[int]:
    matches: List[int] = []
    excluded = excluded or []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if all(token in command for token in required) and not any(token in command for token in excluded):
            matches.append(int(entry.name))
    return matches

def carla_health(timeout_s: float = 5.0) -> str:
    script = (
        "import carla; "
        "client = carla.Client('localhost', 2000); "
        f"client.set_timeout({timeout_s!r}); "
        "world = client.get_world(); "
        "print(f'{world.get_map().name} frame={world.get_snapshot().frame}')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout_s + 2.0,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"CARLA health subprocess exited {result.returncode}")
    return result.stdout.strip()

def stop_carla(log_path: Path) -> None:
    pids = matching_processes(["/home/zxy/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"])
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"sent SIGTERM to unresponsive CARLA pid={pid}", log_path)
        except ProcessLookupError:
            pass
    deadline = time.time() + 15
    while time.time() < deadline and matching_processes(["/home/zxy/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"]):
        time.sleep(1)
    for pid in matching_processes(["/home/zxy/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"]):
        try:
            os.kill(pid, signal.SIGKILL)
            log(f"sent SIGKILL to unresponsive CARLA pid={pid}", log_path)
        except ProcessLookupError:
            pass

def start_carla(launcher: Path, map_path: str, venv_activate: Path, carla_log: Path, log_path: Path) -> None:
    carla_log.parent.mkdir(parents=True, exist_ok=True)
    output = carla_log.open("a", encoding="utf-8")
    command = f"source {shlex.quote(str(venv_activate))} && exec {shlex.quote(str(launcher))} {shlex.quote(map_path)}"
    process = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        cwd=str(launcher.parent),
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    output.close()
    log(f"started CARLA pid={process.pid} map={map_path} venv={venv_activate}", log_path)

def result_counts(output_root: Path) -> dict:
    return {
        variant: len(list((output_root / variant / "case_results").glob("*.json")))
        for variant in VARIANTS
    }

def rq1_result_count(output_root: Path) -> int:
    return len(list((output_root / "case_results").glob("*.json")))

def rq1_baseline_result_count(output_root: Path, method: str) -> int:
    return len(list((output_root / method / "case_results").glob("*.json")))

def expected_total(output_root: Path, shared_manifest: Path) -> int:
    for path in (output_root / "manifest.json", shared_manifest):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        return int(payload.get("selected_total") or payload.get("scenario_count") or payload.get("total") or len(payload.get("rows", [])))
    raise FileNotFoundError(f"manifest not found: {output_root / 'manifest.json'} or {shared_manifest}")

def experiment_running(output_root: Path) -> bool:
    return bool(
        matching_processes(
            ["run_rq2_ablation.py", "--output-dir", str(output_root)],
            ["--case-index"],
        )
        or matching_processes(
            ["run_rq2_ablation.py", "--output-dir", "RKGRScen/data/evaluation/rq2_ablation_full_execution"],
            ["--case-index"],
        )
    )

def rq1_running(output_root: Path) -> bool:
    return bool(
        matching_processes(
            ["run_rq1_carla_full_execution.py", "--output-dir", str(output_root)],
            ["--case-index"],
        )
    )

def rq1_baseline_running(output_root: Path) -> bool:
    return bool(
        matching_processes(
            ["run_rq1_baseline_execution.py", "--output-dir", str(output_root)],
            ["--case-index"],
        )
    )

def start_experiment(base: Path, output_root: Path, experiment_log: Path, log_path: Path, manifest: Path, include_full: bool, cases_per_type: Optional[int], variants: Optional[List[str]], mode: str) -> None:
    command = [
        "/home/zxy/apollo/data/test/test/carla-clean/bin/python",
        "RKGRScen/experiments/run_rq2_ablation.py",
        "--timeout-s", "20",
        "--output-dir", str(output_root),
        "--manifest", str(manifest),
        "--seed", "20260714",
    ]
    variants_to_run = variants or (list(VARIANTS) if include_full else list(VARIANTS[1:]))
    if not include_full:
        variants_to_run = [variant for variant in variants_to_run if variant != "full"]
    for variant in variants_to_run:
        command.extend(["--variant", variant])
    if cases_per_type is not None:
        command.extend(["--cases-per-type", str(cases_per_type)])
    experiment_log.parent.mkdir(parents=True, exist_ok=True)
    output = experiment_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(base),
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    output.close()
    log(f"started RQ2 recovery pid={process.pid} counts={result_counts(output_root)} variants={variants_to_run} cases_per_type={cases_per_type}", log_path)

def start_rq1_experiment(base: Path, output_root: Path, experiment_log: Path, log_path: Path, manifest: Path) -> None:
    command = [
        "/home/zxy/apollo/data/test/test/carla-clean/bin/python",
        "RKGRScen/experiments/run_rq1_carla_full_execution.py",
        "--timeout-s", "20",
        "--output-dir", str(output_root),
        "--manifest", str(manifest),
    ]
    experiment_log.parent.mkdir(parents=True, exist_ok=True)
    output = experiment_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(base),
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    output.close()
    log(f"started RQ1 recovery pid={process.pid} count={rq1_result_count(output_root)}", log_path)

def start_rq1_baseline(base: Path, output_root: Path, experiment_log: Path, log_path: Path, manifest: Path, method: str) -> None:
    command = [
        "/home/zxy/apollo/data/test/test/carla-clean/bin/python",
        "RKGRScen/experiments/run_rq1_baseline_execution.py",
        "--method", method,
        "--timeout-s", "20",
        "--output-dir", str(output_root / method),
        "--manifest", str(manifest),
    ]
    experiment_log.parent.mkdir(parents=True, exist_ok=True)
    output = experiment_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(base),
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )
    output.close()
    log(f"started RQ1 baseline {method} pid={process.pid} count={rq1_baseline_result_count(output_root, method)}", log_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor CARLA and resume RQ experiments.")
    parser.add_argument("--mode", choices=("rq1", "rq1_baselines", "rq2"), default="rq2")
    parser.add_argument("--interval-s", type=float, default=10.0)
    parser.add_argument("--unhealthy-limit", type=int, default=6)
    parser.add_argument("--launcher", default="/home/zxy/CARLA_0.9.13/CarlaUE4.sh")
    parser.add_argument("--venv-activate", default="/home/zxy/apollo/data/test/test/carla-clean/bin/activate")
    parser.add_argument("--map", default="/Game/Carla/Maps/Town03")
    parser.add_argument("--base", default="/home/zxy/apollo/data/test/point2")
    parser.add_argument("--output-dir", default="/home/zxy/apollo/data/test/point2/RKGRScen/data/evaluation/rq2_ablation_full_execution")
    parser.add_argument("--manifest", default=DEFAULT_SHARED_MANIFEST)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--cases-per-type", type=int, default=None)
    parser.add_argument("--variant", action="append", default=[])
    args = parser.parse_args()

    base = Path(args.base).resolve()
    output_root = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    monitor_dir = output_root / "monitor"
    monitor_log = monitor_dir / "watchdog.log"
    carla_log = monitor_dir / "carla.log"
    experiment_log = monitor_dir / ("rq1_recovery.log" if args.mode == "rq1" else "rq1_baselines_recovery.log" if args.mode == "rq1_baselines" else "rq2_recovery.log")
    launcher = Path(args.launcher).resolve()
    venv_activate = Path(args.venv_activate).resolve()
    if not launcher.is_file():
        raise SystemExit(f"CARLA launcher not found: {launcher}")
    if not venv_activate.is_file():
        raise SystemExit(f"virtualenv activate script not found: {venv_activate}")
    unhealthy_count = 0
    last_health_log = 0.0
    last_experiment_start = 0.0

    log(f"watchdog started mode={args.mode} output={output_root}", monitor_log)
    while True:
        try:
            health = carla_health()
            unhealthy_count = 0
            if time.time() - last_health_log >= 60:
                if args.mode == "rq1":
                    log(f"CARLA healthy {health}; rq1_count={rq1_result_count(output_root)}", monitor_log)
                elif args.mode == "rq1_baselines":
                    log(f"CARLA healthy {health}; template={rq1_baseline_result_count(output_root, 'template_mapping')} arise={rq1_baseline_result_count(output_root, 'arise_derived')}", monitor_log)
                else:
                    log(f"CARLA healthy {health}; counts={result_counts(output_root)}", monitor_log)
                last_health_log = time.time()
        except Exception as exc:
            unhealthy_count += 1
            carla_pids = matching_processes(["/home/zxy/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"])
            log(f"CARLA unhealthy attempt={unhealthy_count}/{args.unhealthy_limit} pids={carla_pids}: {exc!r}", monitor_log)
            if not carla_pids or unhealthy_count >= args.unhealthy_limit:
                if carla_pids:
                    stop_carla(monitor_log)
                start_carla(launcher, args.map, venv_activate, carla_log, monitor_log)
                unhealthy_count = 0
                time.sleep(20)
            time.sleep(args.interval_s)
            continue

        total = expected_total(output_root, manifest_path)
        if args.mode == "rq1":
            pending = rq1_result_count(output_root) < total
            if pending and not rq1_running(output_root) and time.time() - last_experiment_start >= 30:
                start_rq1_experiment(base, output_root, experiment_log, monitor_log, manifest_path)
                last_experiment_start = time.time()
            elif not pending and not rq1_running(output_root):
                log(f"RQ1 complete count={rq1_result_count(output_root)}; watchdog exiting", monitor_log)
                return
        elif args.mode == "rq1_baselines":
            for method in ("template_mapping", "arise_derived"):
                count = rq1_baseline_result_count(output_root, method)
                if count < total:
                    if not rq1_baseline_running(output_root) and time.time() - last_experiment_start >= 30:
                        start_rq1_baseline(base, output_root, experiment_log, monitor_log, manifest_path, method)
                        last_experiment_start = time.time()
                    break
            else:
                if not rq1_baseline_running(output_root):
                    log(f"RQ1 baselines complete template={rq1_baseline_result_count(output_root, 'template_mapping')} arise={rq1_baseline_result_count(output_root, 'arise_derived')}; watchdog exiting", monitor_log)
                    return
        else:
            counts = result_counts(output_root)
            variants_to_check = list(dict.fromkeys(args.variant)) if args.variant else (list(VARIANTS) if args.include_full else list(VARIANTS[1:]))
            if not args.include_full:
                variants_to_check = [variant for variant in variants_to_check if variant != "full"]
            pending = any(counts.get(variant, 0) < total for variant in variants_to_check)
            if pending and not experiment_running(output_root) and time.time() - last_experiment_start >= 30:
                start_experiment(base, output_root, experiment_log, monitor_log, manifest_path, args.include_full, args.cases_per_type, variants_to_check, args.mode)
                last_experiment_start = time.time()
            elif not pending and not experiment_running(output_root):
                log(f"RQ2 complete counts={counts}; watchdog exiting", monitor_log)
                return
        time.sleep(args.interval_s)

if __name__ == "__main__":
    main()
