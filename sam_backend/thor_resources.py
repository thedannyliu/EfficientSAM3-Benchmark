from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any


MEMINFO_KEYS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator or key not in MEMINFO_KEYS:
            continue
        fields = remainder.split()
        if not fields:
            continue
        values[f"{key.lower()}_bytes"] = int(fields[0]) * 1024
    return values


def parse_tegrastats(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {"raw": line.strip()}
    ram = re.search(r"\bRAM\s+(\d+)/(\d+)MB", line)
    if ram:
        result["ram_used_mb"] = int(ram.group(1))
        result["ram_total_mb"] = int(ram.group(2))
    swap = re.search(r"\bSWAP\s+(\d+)/(\d+)MB", line)
    if swap:
        result["swap_used_mb"] = int(swap.group(1))
        result["swap_total_mb"] = int(swap.group(2))
    gr3d = re.search(r"\bGR3D_FREQ\s+([0-9]+(?:\.[0-9]+)?)%", line)
    if gr3d:
        result["gr3d_percent"] = float(gr3d.group(1))
    emc = re.search(r"\bEMC_FREQ\s+([0-9]+(?:\.[0-9]+)?)%", line)
    if emc:
        result["emc_percent"] = float(emc.group(1))

    temperatures = {
        name: float(value)
        for name, value in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*)@([0-9]+(?:\.[0-9]+)?)C", line)
    }
    if temperatures:
        result["temperatures_c"] = temperatures

    power = {
        name: {"current": int(current), "average": int(average)}
        for name, current, average in re.findall(
            r"\b([A-Z][A-Z0-9_]*)\s+(\d+)mW/(\d+)mW", line
        )
    }
    if power:
        result["power_mw"] = power
    return result


def collect_snapshot(containers: list[str], label: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "monotonic_s": time.monotonic(),
        "label": label,
        "loadavg": list(os.getloadavg()),
        "host_memory": parse_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8")),
    }
    disk = shutil.disk_usage("/")
    snapshot["root_disk"] = {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}

    errors: dict[str, str] = {}
    try:
        snapshot["tegrastats"] = parse_tegrastats(_one_tegrastats_line())
    except (OSError, RuntimeError) as exc:
        errors["tegrastats"] = str(exc)
    try:
        snapshot["nvidia_smi"] = _nvidia_smi()
    except (OSError, RuntimeError) as exc:
        errors["nvidia_smi"] = str(exc)
    if containers:
        try:
            snapshot["containers"] = _docker_stats(containers)
        except (OSError, RuntimeError) as exc:
            errors["docker_stats"] = str(exc)
    if errors:
        snapshot["errors"] = errors
    return snapshot


def _one_tegrastats_line() -> str:
    process = subprocess.Popen(
        ["tegrastats", "--interval", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if process.stdout is None:
            raise RuntimeError("tegrastats stdout is unavailable")
        line = process.stdout.readline().strip()
        if not line:
            raise RuntimeError("tegrastats returned no sample")
        return line
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _nvidia_smi() -> dict[str, Any]:
    gpu_fields = [
        "name",
        "driver_version",
        "utilization.gpu",
        "temperature.gpu",
        "power.draw",
        "memory.total",
        "memory.used",
        "memory.free",
    ]
    gpu_line = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(gpu_fields)}",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()[0]
    gpu_values = [_parse_scalar(value.strip()) for value in gpu_line.split(",")]

    app_text = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        allow_empty=True,
    )
    applications = []
    for line in app_text.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        applications.append(
            {
                "pid": _parse_scalar(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": _parse_scalar(fields[2]),
            }
        )
    return {"gpu": dict(zip(gpu_fields, gpu_values)), "compute_applications": applications}


def _docker_stats(containers: list[str]) -> list[dict[str, Any]]:
    output = _run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            *containers,
        ],
        allow_empty=True,
    )
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def _run(command: list[str], allow_empty: bool = False) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"{' '.join(command[:2])}: {message}")
    output = result.stdout.strip()
    if not output and not allow_empty:
        raise RuntimeError(f"{' '.join(command[:2])}: empty output")
    return output


def _parse_scalar(value: str) -> int | float | str | None:
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def record_resources(
    output: Path,
    *,
    duration_s: float,
    interval_s: float,
    containers: list[str],
    label: str,
) -> int:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if interval_s <= 0:
        raise ValueError("interval_s must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + duration_s
    next_sample = time.monotonic()
    samples = 0
    with output.open("w", encoding="utf-8") as stream:
        while time.monotonic() < deadline:
            stream.write(json.dumps(collect_snapshot(containers, label), sort_keys=True) + "\n")
            stream.flush()
            samples += 1
            next_sample += interval_s
            remaining = next_sample - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record Jetson Thor host, GPU, and container resources.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, required=True, help="Recording duration in seconds.")
    parser.add_argument("--interval", type=float, default=5.0, help="Target interval in seconds.")
    parser.add_argument("--container", action="append", default=[], help="Container name to sample.")
    parser.add_argument("--label", default="thor-resource-sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = record_resources(
        args.output,
        duration_s=args.duration,
        interval_s=args.interval,
        containers=args.container,
        label=args.label,
    )
    print(json.dumps({"output": str(args.output), "samples": samples}, sort_keys=True))


if __name__ == "__main__":
    main()
