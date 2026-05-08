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

- Code and validation commit: `365b2d0`
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

## v0.1.3 - 2026-05-06

- Code and validation commit: `f616f8a`
- Changes:
  - Set default V0 laptop check to `epochs: 8` with `batch_size: 1`.
  - Added validation prediction export during training:
    - `splits/val_predictions_epoch_*.csv`
    - `splits/val_predictions_best.csv`
  - Added `test.py` prediction export:
    - default path is `splits/test_predictions.csv` beside the checkpoint run directory.
  - Added per-sample prediction rows with true label, predicted label, correctness, loss, and per-class probabilities.
  - Updated the X-Fi WiFi student head to match the X-Fi single-modality style more closely:
    - mean pooled 512-d WiFi feature
    - `LayerNorm(512)`
    - `Linear(512, 9)`
  - Added `XFiWiFiOriginalFC` and `scripts/compare_head_strategies.py` for comparing the feature-head route against direct `fc` replacement.
- Problems found:
  - Per-batch `acc=0` is expected when `batch_size=1`; each batch accuracy can only be `0` or `1`.
  - The 8-epoch V0 run updates parameters successfully, but validation accuracy stays low while train accuracy rises, indicating overfitting or a remaining train/validation/domain mismatch to investigate on the 4080S.
  - The direct original-`fc` route can run, but it does not expose WiFi tokens for V1 CAFD; the feature-head route remains the default for V0/V1 consistency.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\models\xfi_wifi_resnet.py .\WiMANS_Baseline\models\__init__.py .\WiMANS_Baseline\train.py .\WiMANS_Baseline\test.py .\WiMANS_Baseline\scripts\compare_head_strategies.py`
  - `python scripts\compare_head_strategies.py --config config\config.yaml --sample-limit 90 --epochs 2 --batch-size 1`
  - `python train.py --config config\config.yaml --stage v0`
  - `python test.py --config output\wimans_5g_single_baseline\v0\20260506_103030\config.yaml --stage v0 --checkpoint output\wimans_5g_single_baseline\v0\20260506_103030\checkpoints\best.pt`
- Validation result:
  - Static compile passed.
  - Head comparison run:
    - Run directory: `output/head_strategy_compare/v0/20260506_102943`
    - `xfi_feature_head`: best epoch `1`, best val acc `0.055556`
    - `original_fc_replace`: best epoch `1`, best val acc `0.166667`
    - This is a 90-sample, 2-epoch plumbing check only, not a final model-selection result.
  - Full V0 5 GHz single-user batch-size-1 run completed on RTX 3050:
    - Run directory: `output/wimans_5g_single_baseline/v0/20260506_103030`
    - `train=1425`, `val=357`
    - Epoch 1: `train_loss=2.115181 train_acc=0.204211 val_loss=2.575579 val_acc=0.126050`
    - Epoch 2: `train_loss=1.682299 train_acc=0.352281 val_loss=3.680732 val_acc=0.131653`
    - Epoch 8: `train_loss=0.834449 train_acc=0.672281 val_loss=6.038251 val_acc=0.126050`
    - Best validation accuracy: `0.131653` at epoch 2.
    - Model parameters: `5,288,201`
    - Model MACs: `663,321,353`
    - Approx FLOPs: `1,326,642,706`
    - Batch accuracy rows: `11400` total, `5923` rows with acc `0`, `5477` rows with acc `1`.
  - Test entrypoint loaded the best checkpoint and saved predictions:
    - `test_loss=3.680732 test_acc=0.131653`
    - `output/wimans_5g_single_baseline/v0/20260506_103030/splits/test_predictions.csv`

## v0.1.4 - 2026-05-08

- Code and validation commit: `3095948`
- Changes:
  - Fixed `scripts/inspect_data.py` so data inspection uses the same `wifi_band`, `environment`, `num_users`, and `sample_limit` filters as training.
  - Replaced hard-coded selected-row and class-count assertions with config-aware checks.
- Problems found:
  - The previous data inspection script ignored `data.environment`, so `environment: ["classroom"]` configs still reported the full 5 GHz single-user set of `1782` rows instead of the classroom subset.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\scripts\inspect_data.py`
  - `python scripts\inspect_data.py --config config\config.yaml`
  - `python scripts\inspect_data.py --config output\wimans_5g_single_baseline\v1\20260507_093957\config.yaml`
- Validation result:
  - Static compile passed.
  - Current classroom config:
    - `selected_rows=594`
    - class counts are `66` per activity.
    - `missing_wifi=0`, `missing_video=0`
  - Historical environment-null config:
    - `selected_rows=1782`
    - class counts are `198` per activity.
    - `missing_wifi=0`, `missing_video=0`

## v0.1.5 - 2026-05-08

- Code and validation commit: `cec77d8`
- Changes:
  - Added `train_video_teacher.py` for training a video-only teacher on the current 5 GHz single-user HAR task.
  - Added `config/video_teacher.yaml` with seed `39`, `batch_size: 8`, online video reading, and S3D default backbone.
  - Added selectable WiMANS-style video model names through `--model` / `--backbone`: `S3D`, `ResNet`, `MViT-v1`, `MViT-v2`, `Swin-T`, and `Swin-S`.
  - Added `models/video_teacher.py` with torchvision video backbone wrapping, 9-class head replacement, optional backbone freezing, and feature extraction support for later distillation.
  - Updated the dataset video path so video-only training can skip WiFi loading while still saving run config, splits, metrics, predictions, model structure, parameter count, and checkpoint artifacts.
- Problems found:
  - S3D cannot run the tiny smoke test with only 8 frames after temporal downsampling; use at least `--num-frames 16` for smoke tests. Full training keeps `video.num_frames: 90`.
  - Real pretrained video weights use `weights: kinetics400`; if the 4080S machine has no cached weights, it must be able to download them or receive the cache/weights in advance.
- Validation commands:
  - `python -m compileall .\models\video_teacher.py .\models\__init__.py .\datasets\video_loader.py .\datasets\wimans_dataset.py .\train_video_teacher.py`
  - `python train_video_teacher.py --config config\video_teacher.yaml --model S3D --weights none --sample-limit 4 --num-frames 16 --epochs 1 --batch-size 2 --no-flops`
- Validation result:
  - Static compile passed in the `WiMANS` conda environment.
  - Video teacher smoke test completed on the local GPU:
    - Run directory: `output/wimans_5g_single_video_teacher/video_teacher/20260508_190419`
    - `train=3`, `val=1`
    - `epoch=1 train_loss=2.389160 train_acc=0.000000 val_loss=2.195125 val_acc=0.000000`
    - Model parameters: `7,919,273`
    - Saved `splits/val_predictions_epoch_001.csv`, `splits/val_predictions_best.csv`, and `checkpoints/best.pt`.

## Suggested Future Milestones

- `v0.2.0`: WiMANS data checks and label builder validated.
- `v0.3.0`: V0 WiFi-only smoke test validated.
- `v0.4.0`: V1 online S3D + CAFD smoke test validated.
- `v1.0.0`: Full V0/V1 training config stable on 4080S.
