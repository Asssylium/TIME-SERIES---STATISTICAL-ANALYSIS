"""Bundle the build host's Linux OpenMP runtime for Vercel Functions."""

from pathlib import Path
import ctypes
import shutil
import subprocess
import sys


def main() -> None:
    if sys.platform != "linux":
        raise RuntimeError("Run this deployment build step on Linux (Vercel).")
    candidates = []
    if shutil.which("gcc"):
        result = subprocess.run(
            ["gcc", "-print-file-name=libgomp.so.1"],
            check=True, capture_output=True, text=True,
        )
        candidates.append(Path(result.stdout.strip()))
    for directory in ("/usr/lib64", "/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu"):
        candidates.append(Path(directory) / "libgomp.so.1")
    source = next((path for path in candidates if path.is_absolute() and path.is_file()), None)
    if source is None:
        raise RuntimeError("Build host has no libgomp.so.1; cannot bundle LightGBM's OpenMP runtime.")
    destination = Path(__file__).resolve().parents[2] / "native" / "libgomp.so.1"
    destination.parent.mkdir(exist_ok=True)
    shutil.copy2(source.resolve(), destination)
    ctypes.CDLL(str(destination), mode=ctypes.RTLD_GLOBAL)
    print(f"Bundled OpenMP runtime from {source}")


if __name__ == "__main__":
    main()
