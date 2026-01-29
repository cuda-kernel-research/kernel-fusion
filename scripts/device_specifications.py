#!/usr/bin/env python3
"""
Collects hardware/software info needed to fill your merged table (RTX/T4/A100).
Run it on each machine (one per GPU host). It writes:
  - hw_dump/system_gpu_info.txt
  - hw_dump/deviceQuery.txt (if deviceQuery exists)
  - hw_dump/summary.json
  - hw_dump/latex_rows.txt  (optional table rows)

Usage:
  python3 collect_gpu_platform_info.py
  python3 collect_gpu_platform_info.py --out hw_dump_a100
  python3 collect_gpu_platform_info.py --latex

Notes:
- For GPU arch/#SM/L2/shared-mem/compute capability: best source is CUDA deviceQuery.
  The script will try to locate it automatically. If not found, it will leave fields blank.
- nvidia-smi “CUDA Version” is the driver-reported max runtime version; it is not necessarily your toolkit.
  Use nvcc --version for toolkit version when available.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def run(cmd, timeout=30) -> Tuple[int, str, str]:
    """Run a command and return (rc, stdout, stderr). Compatible with Python 3.6."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stdout = p.stdout.decode("utf-8", errors="replace") if isinstance(p.stdout, (bytes, bytearray)) else (p.stdout or "")
        stderr = p.stderr.decode("utf-8", errors="replace") if isinstance(p.stderr, (bytes, bytearray)) else (p.stderr or "")
        return p.returncode, stdout.strip(), stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def which(binary: str) -> Optional[str]:
    p = shutil.which(binary)
    return p


def read_file(path: str) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def parse_os_release() -> Dict[str, str]:
    data = {}
    content = read_file("/etc/os-release")
    if not content:
        return data
    for line in content.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"')
    return data


def parse_lscpu(text: str) -> Dict[str, str]:
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def parse_free_h(text: str) -> Dict[str, str]:
    # Try to parse total memory from "Mem:" line.
    out = {}
    for line in text.splitlines():
        if line.lower().startswith("mem:"):
            parts = line.split()
            if len(parts) >= 2:
                out["mem_total_human"] = parts[1]
    return out


def _supported_nvidia_smi_fields() -> Optional[set]:
    rc, out, err = run(["nvidia-smi", "--help-query-gpu"])
    if rc != 0 or not out:
        return None
    fields = set()
    for line in out.splitlines():
        m = re.match(r"\s*([a-zA-Z0-9_.]+)\s*[-:]", line)
        if m:
            fields.add(m.group(1).strip())
    return fields or None


def nvidia_smi_query() -> Dict[str, str]:
    """
    Return a dict with key fields from nvidia-smi query interface.
    """
    desired = [
        "name",
        "memory.total",
        "driver_version",
        "vbios_version",
        "pci.bus_id",
        "pci.gen.current",
        "pci.gen.max",
        "pci.link.width.current",
        "pci.link.width.max",
        "power.limit",
        "power.max_limit",
    ]
    supported = _supported_nvidia_smi_fields()
    if supported:
        fields = [f for f in desired if f in supported]
    else:
        fields = [
            "name",
            "memory.total",
            "driver_version",
            "pci.bus_id",
            "power.limit",
        ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    rc, out, err = run(cmd)
    if rc != 0 or not out:
        return {"error": f"nvidia-smi query failed: {err or out}"}
    # If multiple GPUs, there will be multiple lines; we keep them all.
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return {"raw_lines": lines, "fields": fields}


def nvidia_smi_topline() -> Dict[str, str]:
    """
    Parse the nvidia-smi header line for driver and 'CUDA Version' (driver runtime).
    """
    rc, out, err = run(["nvidia-smi"])
    if rc != 0:
        return {"error": f"nvidia-smi failed: {err}"}
    m = re.search(r"Driver Version:\s*([0-9.]+)", out)
    driver = m.group(1) if m else ""
    m2 = re.search(r"CUDA Version:\s*([0-9.]+)", out)
    cuda_driver = m2.group(1) if m2 else ""
    return {"driver_version_header": driver, "cuda_version_driver_reported": cuda_driver, "nvidia_smi_text": out}


def nvcc_version() -> Dict[str, str]:
    nvcc = which("nvcc")
    if not nvcc:
        return {"nvcc_path": "", "nvcc_version_text": "", "note": "nvcc not found in PATH"}
    rc, out, err = run([nvcc, "--version"])
    return {"nvcc_path": nvcc, "nvcc_version_text": out or err}


def _cuda_root_from_nvcc(nvcc_path: str) -> Optional[str]:
    if not nvcc_path:
        return None
    try:
        p = Path(nvcc_path).resolve()
        # Usually .../bin/nvcc -> CUDA root is parent of bin
        if p.parent.name == "bin":
            return str(p.parent.parent)
    except Exception:
        return None
    return None


def find_devicequery() -> Optional[str]:
    # Common locations
    candidates = [
        "/opt/cuda/extras/demo_suite/deviceQuery",
        "/usr/local/cuda/extras/demo_suite/deviceQuery",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Try PATH
    p = which("deviceQuery")
    if p:
        return p

    # Try CUDA samples build output (best effort)
    sample_dirs = [
        "/usr/local/cuda/samples/1_Utilities/deviceQuery",
        "/opt/cuda/samples/1_Utilities/deviceQuery",
    ]

    # Try CUDA root from nvcc
    nvcc = which("nvcc")
    cuda_root = _cuda_root_from_nvcc(nvcc) if nvcc else None
    if cuda_root:
        sample_dirs.append(os.path.join(cuda_root, "samples", "1_Utilities", "deviceQuery"))
        sample_dirs.append(os.path.join(cuda_root, "extras", "demo_suite"))
    for d in sample_dirs:
        exe = os.path.join(d, "deviceQuery")
        if os.path.isfile(exe) and os.access(exe, os.X_OK):
            return exe

    return None


def run_devicequery() -> Dict[str, str]:
    dq = find_devicequery()
    if not dq:
        return {"deviceQuery_path": "", "deviceQuery_text": "", "note": "deviceQuery not found"}
    rc, out, err = run([dq], timeout=60)
    txt = out if out else err
    return {"deviceQuery_path": dq, "deviceQuery_text": txt}


def parse_devicequery(text: str) -> Dict[str, Optional[str]]:
    """
    Extract compute capability, SM count, L2 cache size, shared memory per multiprocessor from deviceQuery output.
    deviceQuery output varies slightly; we use robust regex.
    """
    if not text:
        return {}

    def grab(pattern: str) -> Optional[str]:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        return m.group(1).strip() if m else None

    compute = grab(r"CUDA Capability.*?:\s*([0-9.]+)")
    sm_count = grab(r"Multiprocessor Count\s*:\s*([0-9]+)")
    l2 = grab(r"L2 Cache Size\s*:\s*([0-9]+)\s*bytes")
    if l2:
        # Convert bytes -> MB (exact) for readability
        try:
            l2_mb = str(round(int(l2) / (1024 * 1024), 2))
        except Exception:
            l2_mb = None
    else:
        l2_mb = None

    shmem_mp = grab(r"Total amount of shared memory per multiprocessor\s*:\s*([0-9]+)\s*bytes")
    if shmem_mp:
        try:
            shmem_kb = str(round(int(shmem_mp) / 1024, 2))
        except Exception:
            shmem_kb = None
    else:
        shmem_kb = None

    # Some builds show sharedMemPerBlock too; optional
    shmem_block = grab(r"Total amount of shared memory per block\s*:\s*([0-9]+)\s*bytes")
    if shmem_block:
        try:
            shmem_block_kb = str(round(int(shmem_block) / 1024, 2))
        except Exception:
            shmem_block_kb = None
    else:
        shmem_block_kb = None

    return {
        "compute_capability": compute,
        "sm_count": sm_count,
        "l2_cache_mb": l2_mb,
        "shared_mem_per_sm_kb": shmem_kb,
        "shared_mem_per_block_kb": shmem_block_kb,
    }


def guess_arch_from_compute_cap(cc: Optional[str]) -> Optional[str]:
    """
    Very coarse mapping. If you want exact naming, fill it manually from NVIDIA specs.
    """
    if not cc:
        return None
    try:
        major = int(float(cc))
        minor = int(round((float(cc) - major) * 10))
    except Exception:
        return None

    # Common mappings
    if (major, minor) == (7, 5):
        return "Turing"
    if (major, minor) == (8, 0):
        return "Ampere"
    if (major, minor) == (8, 6):
        return "Ampere"
    if (major, minor) == (9, 0):
        return "Hopper"
    return None


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8", errors="replace")


def make_latex_rows(summary: Dict) -> str:
    """
    Returns LaTeX-ready table rows for the four-column table:
    Field & RTX 3080 Ti & T4 & A100
    This script can only fill the current machine column reliably; you merge later.
    """
    # We output a "single-machine" column suggestion.
    gpu_name = summary.get("gpu", {}).get("name", "")
    vram = summary.get("gpu", {}).get("memory_total_gb", "")
    driver = summary.get("gpu", {}).get("driver_version", "")
    cuda_driver = summary.get("gpu", {}).get("cuda_driver_reported", "")
    nvcc = summary.get("cuda", {}).get("nvcc_toolkit_version", "")
    cc = summary.get("gpu_arch", {}).get("compute_capability", "")
    sm = summary.get("gpu_arch", {}).get("sm_count", "")
    l2 = summary.get("gpu_arch", {}).get("l2_cache_mb", "")
    shmem = summary.get("gpu_arch", {}).get("shared_mem_per_sm_kb", "")
    arch = summary.get("gpu_arch", {}).get("architecture_guess", "")

    cpu = summary.get("system", {}).get("cpu_model", "")
    ram = summary.get("system", {}).get("ram_total_human", "")
    osname = summary.get("system", {}).get("os_pretty", "")

    def dash(x: str) -> str:
        return x if x else r"\textemdash"

    rows = []
    rows.append(rf"GPU model & {dash('')} & {dash('')} & {dash(gpu_name)} \\")
    rows.append(rf"GPU VRAM & {dash('')} & {dash('')} & {dash(vram)} \\")
    rows.append(rf"Architecture & {dash('')} & {dash('')} & {dash(arch)} \\")
    rows.append(rf"Compute capability (SM) & {dash('')} & {dash('')} & {dash(cc)} \\")
    rows.append(rf"\#SMs & {dash('')} & {dash('')} & {dash(sm)} \\")
    rows.append(rf"L2 cache size & {dash('')} & {dash('')} & {dash((l2 + ' MB') if l2 else '')} \\")
    rows.append(rf"Shared memory per SM & {dash('')} & {dash('')} & {dash((shmem + ' KB') if shmem else '')} \\")
    rows.append(rf"OS & {dash('')} & {dash('')} & {dash(osname)} \\")
    rows.append(rf"Driver version & {dash('')} & {dash('')} & {dash(driver)} \\")
    rows.append(rf"CUDA (driver-reported) & {dash('')} & {dash('')} & {dash(cuda_driver)} \\")
    rows.append(rf"CUDA toolkit (nvcc) & {dash('')} & {dash('')} & {dash(nvcc)} \\")
    rows.append(rf"CPU & {dash('')} & {dash('')} & {dash(cpu)} \\")
    rows.append(rf"System RAM & {dash('')} & {dash('')} & {dash(ram)} \\")
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hw_dump", help="Output directory (default: hw_dump)")
    ap.add_argument("--latex", action="store_true", help="Also write LaTeX row suggestions (single-machine column).")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect system
    os_rel = parse_os_release()
    rc, lscpu_txt, _ = run(["lscpu"])
    rc2, free_txt, _ = run(["free", "-h"])
    rc3, lsb_txt, _ = run(["lsb_release", "-a"])  # may fail if not installed

    lscpu_map = parse_lscpu(lscpu_txt) if lscpu_txt else {}
    free_map = parse_free_h(free_txt) if free_txt else {}

    os_pretty = os_rel.get("PRETTY_NAME") or ""
    cpu_model = lscpu_map.get("Model name", "")

    # GPU + CUDA
    smi_q = nvidia_smi_query()
    smi_hdr = nvidia_smi_topline()
    nvcc = nvcc_version()
    dq = run_devicequery()
    dq_parsed = parse_devicequery(dq.get("deviceQuery_text", ""))

    # Parse first GPU line from query (if exists)
    gpu_parsed = {}
    if "raw_lines" in smi_q and smi_q["raw_lines"]:
        # Each line matches fields order
        # name, memory.total, driver_version, vbios_version, pci.bus_id, pci.gen.current, pci.gen.max, pci.link.width.current, pci.link.width.max, power.limit, power.max_limit
        parts = [p.strip() for p in smi_q["raw_lines"][0].split(",")]
        if len(parts) >= 2:
            name = parts[0]
            mem_mb_or_mib = parts[1]
            # nvidia-smi returns in MiB by default unless nounits; with nounits it is numeric (MiB).
            try:
                mem_gb = round(float(mem_mb_or_mib) / 1024, 2)  # MiB -> GiB approx
                mem_gb_str = f"{mem_gb} GB"
            except Exception:
                mem_gb_str = ""
            gpu_parsed["name"] = name
            gpu_parsed["memory_total_gb"] = mem_gb_str
        # driver version also available in parts[2]
        if len(parts) >= 3:
            gpu_parsed["driver_version"] = parts[2]

    gpu_parsed["cuda_driver_reported"] = smi_hdr.get("cuda_version_driver_reported", "")

    # Fill nvcc toolkit version from nvcc --version
    nvcc_text = nvcc.get("nvcc_version_text", "")
    toolkit_ver = ""
    m = re.search(r"release\s+([0-9.]+)", nvcc_text)
    if m:
        toolkit_ver = m.group(1)
    cuda_map = {
        "nvcc_path": nvcc.get("nvcc_path", ""),
        "nvcc_version_text": nvcc_text,
        "nvcc_toolkit_version": toolkit_ver,
        "cuda_driver_reported": gpu_parsed.get("cuda_driver_reported", ""),
    }

    # Architecture guess (optional)
    arch_guess = guess_arch_from_compute_cap(dq_parsed.get("compute_capability")) if dq_parsed else None

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "system": {
            "os_pretty": os_pretty,
            "os_release": os_rel,
            "lsb_release": lsb_txt if rc3 == 0 else "",
            "cpu_model": cpu_model,
            "lscpu_raw": lscpu_txt,
            "ram_total_human": free_map.get("mem_total_human", ""),
            "free_raw": free_txt,
        },
        "gpu": gpu_parsed,
        "gpu_arch": {
            **dq_parsed,
            "architecture_guess": arch_guess or "",
        },
        "cuda": cuda_map,
        "raw": {
            "nvidia_smi_query": smi_q,
            "nvidia_smi_full": smi_hdr.get("nvidia_smi_text", ""),
            "deviceQuery_path": dq.get("deviceQuery_path", ""),
        },
        "notes": [],
    }

    if dq.get("note"):
        summary["notes"].append(dq["note"])
    if "error" in smi_q:
        summary["notes"].append(smi_q["error"])
    if smi_hdr.get("error"):
        summary["notes"].append(smi_hdr["error"])
    if nvcc.get("note"):
        summary["notes"].append(nvcc["note"])

    # Write human-readable dumps
    write_text(outdir / "system_gpu_info.txt",
               "\n".join([
                   f"=== DATE ===\n{summary['timestamp']}\n",
                   "=== OS (PRETTY_NAME) ===",
                   os_pretty,
                   "\n=== lsb_release -a ===",
                   lsb_txt if rc3 == 0 else "(lsb_release not available)",
                   "\n=== uname -a ===",
                   run(["uname", "-a"])[1],
                   "\n=== lscpu ===",
                   lscpu_txt,
                   "\n=== free -h ===",
                   free_txt,
                   "\n=== nvcc --version ===",
                   nvcc_text or "(nvcc not found)",
                   "\n=== nvidia-smi (header + table) ===",
                   smi_hdr.get("nvidia_smi_text", ""),
               ]))

    # deviceQuery dump
    if dq.get("deviceQuery_text"):
        write_text(outdir / "deviceQuery.txt", dq["deviceQuery_text"])
    else:
        write_text(outdir / "deviceQuery.txt", "(deviceQuery not found or failed)\n")

    # JSON summary
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional LaTeX rows
    if args.latex:
        write_text(outdir / "latex_rows.txt", make_latex_rows(summary))

    print(f"Saved outputs to: {outdir.resolve()}")
    print(f"- {outdir/'summary.json'}")
    print(f"- {outdir/'system_gpu_info.txt'}")
    print(f"- {outdir/'deviceQuery.txt'}")
    if args.latex:
        print(f"- {outdir/'latex_rows.txt'}")


if __name__ == "__main__":
    main()
