from __future__ import annotations
import importlib.metadata as im
from packaging.version import Version

def version(name):
    try:
        return im.version(name)
    except im.PackageNotFoundError:
        return None

def main():
    nnv=version("nnunetv2")
    if nnv!="2.8.1":
        raise SystemExit(f"nnunetv2==2.8.1 required; found {nnv}")
    try:
        import torch
    except Exception as e:
        raise SystemExit(f"PyTorch is required for neural inference: {e}")
    tv=Version(torch.__version__.split("+")[0])
    if tv < Version("2.1.2") or (tv.major==2 and tv.minor==9):
        raise SystemExit(
            f"Unsupported PyTorch {torch.__version__}; require >=2.1.2 and exclude 2.9.x"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "Raw GraphMS 3-D neural inference requires a CUDA-capable PyTorch runtime. "
            "Verification and Stage16 evaluation replay remain CPU-capable."
        )
    print("GRAPHMS NEURAL RUNTIME PASS")
    print("torch:",torch.__version__,"CUDA:",torch.version.cuda)
    print("GPU:",torch.cuda.get_device_name(0))
    print("nnunetv2:",nnv)

if __name__=="__main__":
    main()
