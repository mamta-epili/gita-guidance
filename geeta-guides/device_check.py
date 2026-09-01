"""
device_check.py — report which accelerator PyTorch can actually use here.

Run:  python device_check.py

Prints a one-line verdict plus details. The `device` string it recommends is
the one to paste into gita_gpt.py.
"""

import platform
import sys


def main() -> None:
    print(f"python   : {sys.version.split()[0]}")
    print(f"platform : {platform.system()} {platform.machine()}")

    try:
        import torch
    except ImportError:
        print("\ntorch is not installed. Run ./setup.sh first (or: pip install torch).")
        raise SystemExit(1)

    print(f"torch    : {torch.__version__}")

    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if cuda:
        device = "cuda"
        print(f"\nCUDA available: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {props.name}, {props.total_memory / 1e9:.1f} GB")
    elif mps:
        device = "mps"
        print("\nApple Metal (MPS) available — Apple Silicon GPU acceleration.")
    else:
        device = "cpu"
        print("\nNo GPU backend found. CPU only.")
        if getattr(torch.backends, "mps", None) is not None:
            print(f"  mps built : {torch.backends.mps.is_built()}")

    print(f"\n>>> recommended device = {device!r}")

    # A real tensor op, because "is_available()" is not the same as "works".
    try:
        x = torch.randn(64, 64, device=device)
        y = (x @ x).sum().item()
        print(f"smoke test on {device}: OK (matmul sum = {y:.3f})")
    except Exception as e:  # noqa: BLE001
        print(f"smoke test on {device}: FAILED -> {e}")
        print("Fall back to device = 'cpu'.")
        raise SystemExit(1)

    if device == "cpu":
        print(
            "\nNOTE: on CPU, use the CPU-fallback hyperparameters commented at the\n"
            "top of gita_gpt.py (block_size=64, n_embd=128, batch_size=12).\n"
            "The full config will take hours on CPU."
        )


if __name__ == "__main__":
    main()
