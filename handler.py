"""
RunPod Serverless worker entrypoint for Boltz (Boltz-2).

This runs Boltz inference as a "job" (no web server). RunPod will invoke
`handler(event)` per request and expects a JSON-serializable response.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _now_ms() -> int:
    return int(time.time() * 1000)


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
    return default


def _write_input_file(job_input: Dict[str, Any], work_dir: Path) -> Tuple[Path, str]:
    """
    Create a local input file for `boltz predict`.

    Supports:
    - input_path: container path (must exist)
    - input_yaml: YAML text (written to work_dir/job.yaml)
    - input_fasta: FASTA text (written to work_dir/job.fasta)
    """
    input_path = job_input.get("input_path")
    if isinstance(input_path, str) and input_path.strip():
        p = Path(input_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"input_path not found: {p}")
        return p, p.suffix.lower().lstrip(".") or "file"

    input_yaml = job_input.get("input_yaml")
    if isinstance(input_yaml, str) and input_yaml.strip():
        p = work_dir / "job.yaml"
        p.write_text(input_yaml)
        return p, "yaml"

    input_fasta = job_input.get("input_fasta")
    if isinstance(input_fasta, str) and input_fasta.strip():
        p = work_dir / "job.fasta"
        p.write_text(input_fasta)
        return p, "fasta"

    raise ValueError("Provide one of: input_path, input_yaml, input_fasta.")


def _collect_affinity_json(predictions_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not predictions_dir.exists():
        return results

    for pred_dir in sorted([p for p in predictions_dir.iterdir() if p.is_dir()]):
        for p in pred_dir.glob("affinity_*.json"):
            try:
                results.append(
                    {
                        "prediction_id": pred_dir.name,
                        "file": str(p),
                        "data": json.loads(p.read_text()),
                    }
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    {
                        "prediction_id": pred_dir.name,
                        "file": str(p),
                        "error": str(e),
                    }
                )
    return results


def _run_boltz_predict(job_input: Dict[str, Any]) -> Dict[str, Any]:
    cache_dir = Path(str(job_input.get("cache_dir") or os.getenv("BOLTZ_CACHE") or "/cache"))
    out_root = Path(str(job_input.get("out_dir") or "/tmp/boltz_out"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    accelerator = str(job_input.get("accelerator") or "gpu").lower().strip()
    model = str(job_input.get("model") or "boltz2").lower().strip()

    use_msa_server = _coerce_bool(job_input.get("use_msa_server"), default=False)
    msa_server_url = str(job_input.get("msa_server_url") or "https://api.colabfold.com").strip()
    msa_pairing_strategy = str(job_input.get("msa_pairing_strategy") or "greedy").strip()

    # Write job input to a temp file (unless user provided input_path).
    work_dir = Path(tempfile.mkdtemp(prefix="runpod_boltz_"))
    try:
        input_file, _kind = _write_input_file(job_input, work_dir)

        # Boltz will create: out_root/boltz_results_<stem>/*
        # Use a deterministic stem when we create the file.
        stem = input_file.stem
        expected_run_dir = out_root / f"boltz_results_{stem}"

        cmd: List[str] = [
            "boltz",
            "predict",
            str(input_file),
            "--out_dir",
            str(out_root),
            "--accelerator",
            accelerator,
            "--model",
            model,
        ]

        if use_msa_server:
            cmd += [
                "--use_msa_server",
                "--msa_server_url",
                msa_server_url,
                "--msa_pairing_strategy",
                msa_pairing_strategy,
            ]

        # Optional knobs (pass only if provided)
        for key, flag in [
            ("recycling_steps", "--recycling_steps"),
            ("sampling_steps", "--sampling_steps"),
            ("diffusion_samples", "--diffusion_samples"),
            ("sampling_steps_affinity", "--sampling_steps_affinity"),
            ("diffusion_samples_affinity", "--diffusion_samples_affinity"),
            ("max_msa_seqs", "--max_msa_seqs"),
            ("num_subsampled_msa", "--num_subsampled_msa"),
            ("devices", "--devices"),
        ]:
            if job_input.get(key) is not None:
                cmd += [flag, str(job_input[key])]

        if _coerce_bool(job_input.get("override"), default=False):
            cmd += ["--override"]
        if _coerce_bool(job_input.get("subsample_msa"), default=False):
            cmd += ["--subsample_msa"]
        if _coerce_bool(job_input.get("no_kernels"), default=False):
            cmd += ["--no_kernels"]
        if _coerce_bool(job_input.get("write_embeddings"), default=False):
            cmd += ["--write_embeddings"]

        # Ensure cache is pointed correctly for the boltz process
        env = dict(os.environ)
        env["BOLTZ_CACHE"] = str(cache_dir)

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)  # noqa: S603
        elapsed = time.time() - t0

        stdout_tail = proc.stdout.splitlines()[-200:]
        stderr_tail = proc.stderr.splitlines()[-200:]

        if proc.returncode != 0:
            return {
                "status": "failed",
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "cmd": cmd,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }

        predictions_dir = expected_run_dir / "predictions"
        affinity = _collect_affinity_json(predictions_dir)

        return {
            "status": "completed",
            "elapsed_seconds": elapsed,
            "cmd": cmd,
            "run_dir": str(expected_run_dir),
            "predictions_dir": str(predictions_dir),
            "affinity": affinity,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    finally:
        # Don't delete outputs; only remove temporary input dir.
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass


def handler(event: Dict[str, Any]) -> Any:
    job_input = event.get("input") if isinstance(event, dict) else None
    job_input = job_input if isinstance(job_input, dict) else {}

    action = str(job_input.get("action") or "echo").lower().strip()

    if action == "health":
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
            cuda_devices = int(torch.cuda.device_count()) if cuda else 0
        except Exception:
            cuda = False
            cuda_devices = 0

        return {
            "status": "ok",
            "service": "boltz-runpod-worker",
            "timestamp_ms": _now_ms(),
            "cuda_available": cuda,
            "cuda_devices": cuda_devices,
            "boltz_cache": os.getenv("BOLTZ_CACHE", "/cache"),
            "actions": ["echo", "health", "predict"],
        }

    if action == "predict":
        return _run_boltz_predict(job_input)

    return {"echo": job_input}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})

