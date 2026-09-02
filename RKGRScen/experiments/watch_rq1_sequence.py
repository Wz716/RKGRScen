import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

PYTHON = "/home/zxy/apollo/data/test/test/carla-clean/bin/python"
BASE = Path("/home/zxy/apollo/data/test/point2")
WATCHDOG = BASE / "RKGRScen" / "experiments" / "watch_rq2_carla.py"

def log(message: str, path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")

def proc_has(tokens) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        except Exception:
            continue
        if all(token in cmd for token in tokens):
            return True
    return False

def case_count(path: Path) -> int:
    return len(list((path / "case_results").glob("*.json")))

def start_watchdog(mode: str, output_dir: Path, manifest: Path, log_path: Path, include_full: bool = False) -> subprocess.Popen:
    command = [
        PYTHON,
        str(WATCHDOG),
        "--mode", mode,
        "--manifest", str(manifest),
        "--output-dir", str(output_dir),
    ]
    if include_full:
        command.append("--include-full")
    out = (output_dir / "monitor" / f"{mode}_sequence_watchdog.log").open("a", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(BASE), stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    out.close()
    log(f"started {mode} watchdog pid={process.pid} output={output_dir}", log_path)
    return process

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--full-output", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--rq2-output", default="/home/zxy/apollo/data/test/point2/RKGRScen/data/evaluation/rq2_ablation_full_execution")
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    full_output = Path(args.full_output).resolve()
    baseline_output = Path(args.baseline_output).resolve()
    rq2_output = Path(args.rq2_output).resolve()
    log_path = baseline_output / "monitor" / "rq1_sequence.log"
    expected = __import__("json").loads(manifest.read_text(encoding="utf-8")).get("scenario_count", 0)
    log(f"sequence started expected={expected}", log_path)

    baseline_started = False
    rq2_started = False
    while True:
        full_count = case_count(full_output)
        full_running = proc_has(["watch_rq2_carla.py", "--mode", "rq1", "--output-dir", str(full_output)]) or proc_has(["run_rq1_carla_full_execution.py", "--output-dir", str(full_output)])
        if full_count >= expected and not baseline_started:
            if not proc_has(["watch_rq2_carla.py", "--mode", "rq1_baselines", "--output-dir", str(baseline_output)]):
                start_watchdog("rq1_baselines", baseline_output, manifest, log_path)
            baseline_started = True
        template_count = case_count(baseline_output / "template_mapping")
        arise_count = case_count(baseline_output / "arise_derived")
        log(f"progress full={full_count}/{expected} full_running={full_running} template={template_count}/{expected} arise={arise_count}/{expected}", log_path)
        if baseline_started and template_count >= expected and arise_count >= expected:
            rq1_baseline_running = proc_has(["run_rq1_baseline_execution.py", "--output-dir", str(baseline_output)])
            rq2_watchdog_running = proc_has(["watch_rq2_carla.py", "--mode", "rq2", "--output-dir", str(rq2_output)])
            rq2_running = proc_has(["run_rq2_ablation.py", "--output-dir", str(rq2_output)])
            if not rq1_baseline_running and not rq2_started and not rq2_watchdog_running and not rq2_running:
                start_watchdog("rq2", rq2_output, manifest, log_path, include_full=True)
                rq2_started = True
            if rq2_started or rq2_watchdog_running or rq2_running:
                log(f"RQ1 sequence complete; RQ2 active output={rq2_output}", log_path)
                return
        time.sleep(args.poll_s)

if __name__ == "__main__":
    main()
