# Runtime modes

| Mode | Inputs | What executes |
| --- | --- | --- |
| `verify` | Committed manifests and audits | Frozen identity and audit checks |
| `evaluation-replay` | Committed per-case metrics | Verification and five-fold aggregation |
| `patient` | Three aligned MRI volumes, explicit/registered fold, locally imported frozen assets, atlas cache | CNN preprocessing/inference → graph construction → GAT → Hybrid → Stage11 → canonical Stage12 → Stage13 → report |

Patient mode requires CUDA. It has no ground-truth argument and does not train,
select epochs, tune thresholds, or build an unvalidated ensemble.

Frozen numerical behavior is preserved: CNN step size 0.5 with its original
Gaussian/mirroring settings; FP16 encoder storage; FP32 GAT computation with
FP16 context storage; 64-voxel Hybrid patches at stride 32 with uniform overlap
averaging; native-space probability restoration; fold-specific Stage11.
No Gaussian-overlap replacement is applied to the Hybrid windows.

Source parity and CPU functionality are tested. Full neural acceptance with
original checkpoints is pending; `COMPLETE.json` only describes a successful
individual execution and does not claim external or clinical validation.
