# Changelog

All notable changes, problems, validation commands, and commit hashes should be
recorded here before moving the project to the 4080S machine.

## v0.1.0 - 2026-04-30

- Implementation commit: `83859ac`
- Changes:
  - Created independent `WiMANS_Baseline/` project skeleton.
  - Added config, dataset loaders, WiFi backbone wrapper, S3D teacher, CAFD loss, smoke tests, and training entrypoint.
  - Added README and git workflow documentation for transfer to 4080S.
- Problems found:
  - Root workspace is not a git repository.
  - `conda activate WiMANS` is blocked inside this Codex command sandbox, although direct `WiMANS\python.exe` works.
  - Current laptop GPU is for smoke tests only; full V1 online S3D training should run on the 4080S machine.
- Validation commands:
  - `python -m compileall WiMANS_Baseline`
  - `python scripts\inspect_data.py --config config\config.yaml`
  - `python scripts\smoke_v0.py --config config\config.yaml --limit 8 --batch-size 2`
  - `python scripts\smoke_v1.py --config config\config.yaml --limit 1 --num-frames 16 --s3d-weights none`
- Validation result:
  - Static compile passed.
  - `DATA_CHECK_OK`
  - `SMOKE_V0_OK`
    - `logits_shape=(2, 9)`
    - `loss=2.267420`
    - checkpoint saved and reloaded from `output\smoke_v0\checkpoint.pt`
  - `SMOKE_V1_OK`
    - `logits_shape=(1, 9)`
    - `video_shape=(1, 3, 16, 224, 224)`
    - `cls_loss=2.226233`
    - `cafd_loss=1.089128`
    - `total_loss=2.335146`
    - `trainable_teacher_params=0`
    - checkpoint saved and reloaded from `output\smoke_v1\checkpoint.pt`

## v0.1.1 - 2026-05-03

- Code and validation commit: `b287ab6`
- Changes:
  - Set default `train.batch_size` to `1` for the current RTX 3050 smoke-training workflow.
  - Added training CLI overrides: `--sample-limit`, `--num-frames`, and `--s3d-weights`.
  - Made train/validation splitting fall back to non-stratified split for tiny sample-limit runs.
- Problems found:
  - Full V1 online S3D training is not appropriate for the 3050 laptop; use the 4080S for full V1.
  - For V1 laptop validation, use random S3D weights via `--s3d-weights none` to avoid pretrained-weight download.
- Validation commands:
  - `python train.py --config config\config.yaml --stage v0`
  - `python train.py --config config\config.yaml --stage v1 --sample-limit 2 --num-frames 16 --s3d-weights none`
- Validation result:
  - V0 full 5GHz single-user epoch completed on RTX 3050:
    - `epoch=1 train_loss=2.052317 train_acc=0.202105 val_loss=4.460630 val_acc=0.084034`
  - V1 tiny online-video training completed on RTX 3050:
    - `epoch=1 train_loss=2.324833 train_acc=0.000000 val_loss=2.429820 val_acc=0.000000`

## v0.1.2 - 2026-05-06

- Commit: `PENDING`
- Changes:
  - Added timestamped run directories under `output/<experiment>/<stage>/<YYYYMMDD_HHMMSS>/`.
  - Added `train.log` with effective config, dataset split paths, model structure, parameter counts, MAC/FLOP estimates, and detailed training progress.
  - Added saved run artifacts: `config.yaml`, `model.txt`, `model_summary.yaml`, `splits/train.csv`, `splits/val.csv`, `metrics/train_batches.csv`, `metrics/epochs.csv`, and `checkpoints/best.pt`.
  - Added CLI-safe model profiling utilities and cleaned the X-Fi WiFi student so only actively used feature layers and the 9-class head are registered.
- Problems found:
  - The previous WiFi student kept the original unused 55-class X-Fi head registered through `self.backbone`; this made model structure and parameter/FLOP logs noisy. It is now removed from the registered training model.
  - Full V1 online S3D should still be trained on the 4080S. The 3050 is suitable for V1 tiny checks only.
- Validation commands:
  - `python -m compileall WiMANS_Baseline\train.py WiMANS_Baseline\test.py WiMANS_Baseline\utils`
  - `python train.py --config config\config.yaml --stage v0 --sample-limit 8`
  - `python train.py --config config\config.yaml --stage v1 --sample-limit 2 --num-frames 16 --s3d-weights none`
  - `python train.py --config config\config.yaml --stage v0`
- Validation result:
  - V0 small logging run created all expected run artifacts.
  - V1 tiny online-video logging run created all expected run artifacts.
  - Full V0 5 GHz single-user batch-size-1 run completed on RTX 3050:
    - Run directory: `output/wimans_5g_single_baseline/v0/20260506_094705`
    - `train=1425`, `val=357`
    - `epoch=1 train_loss=2.052317 train_acc=0.202105 val_loss=4.460630 val_acc=0.084034`
    - Model parameters: `5,287,177`
    - Model MACs: `663,320,841`
    - Approx FLOPs: `1,326,641,682`

## Suggested Future Milestones

- `v0.2.0`: WiMANS data checks and label builder validated.
- `v0.3.0`: V0 WiFi-only smoke test validated.
- `v0.4.0`: V1 online S3D + CAFD smoke test validated.
- `v1.0.0`: Full V0/V1 training config stable on 4080S.
