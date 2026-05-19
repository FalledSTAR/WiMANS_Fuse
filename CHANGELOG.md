# Changelog

All notable changes, problems, validation commands, and commit hashes should be
recorded here before moving the project to the 4080S machine.

## v0.1.27 - 2026-05-19

- Implementation commit: `pending`
- Changes:
  - Added weighted slot-wise CrossEntropy support for `multi_slot_ce`.
  - Added `loss.slot_ce_class_weights` and the shorthand `loss.slot_ce_empty_weight` / `loss.slot_ce_activity_weight`.
  - Added CLI overrides `--slot-ce-empty-weight` and `--slot-ce-activity-weight`.
  - Added 5 GHz X-Fi ResNet-18 slot-CE configs:
    - `config/wimans_multi_xfi_slot_ce_5g.yaml`
    - `config/wimans_multi_xfi_slot_ce_5g_weighted.yaml`
- Problems found:
  - The repeated X-Fi slot-CE run `20260519_211629` exactly matched `20260519_102234`, including split-file MD5 hashes and best metrics.
  - Current slot-CE baseline predicts active slot count well, but active activity classes remain confused.
- Validation commands:
  - `python -m compileall train.py losses config utils datasets models scripts`
  - Weighted slot-CE smoke run with `--sample-limit 24 --epochs 1 --batch-size 4`.
- Validation result:
  - Static compile passed.
  - Weighted `multi_slot_ce` loss forward/backward check passed with finite gradients.
  - Tiny weighted X-Fi slot-CE smoke run completed:
    - `run_dir=output/multi/classroom/wimans_classroom_5g_multi_xfi_slot_ce_weighted_empty03/v0/20260519_224322`
    - `slot_ce_class_weights=[0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
    - `epoch=1 train_loss=1.518262 train_acc=0.692982 val_loss=1.201196 val_acc=0.833333`

## v0.1.26 - 2026-05-19

- Implementation commit: `c42d71f`
- Changes:
  - Added structured output directory routing for future runs.
  - `train.py` now writes single-person runs under `output/single/<experiment>/<stage>/<timestamp>`.
  - Multi-user runs now write under `output/multi/<environment>/<experiment>/<stage>/<timestamp>`.
  - `train_video_teacher.py` now uses the same routing, so video-teacher runs stay separated under `output/single/<teacher_experiment>/video_teacher/<timestamp>`.
  - Cleaned historical output layout for the latest 5 GHz multi-user runs and corrected saved `experiment.name` metadata for the moved 5 GHz runs.
- Problems found:
  - Some copied 5 GHz multi-user runs were still under `24g` experiment folder names.
  - Two 5 GHz multi-user runs were missing the `v0` stage folder after manual output reorganization.
- Validation commands:
  - `python -m compileall train.py train_video_teacher.py utils`
  - `python -c "from pathlib import Path; from utils.run_logging import structured_output_dir; ..."`
  - Local output audit over `output/**/config.yaml`.
- Validation result:
  - Static compile passed.
  - `structured_output_dir` returned `output/multi/classroom` for classroom multi-user configs and `output/single` for single-user configs.
  - Output audit found 42 saved runs and all passed after the directory cleanup.

## v0.1.25 - 2026-05-19

- Implementation commit: `a903dd2`
- Changes:
  - Added `multi_slot_ce` label mode for multi-user activity recognition.
  - Encoded each user slot as a 10-class label: `empty_slot + 9 activities`.
  - Added slot-wise CrossEntropy loss over logits reshaped to `[B, 6, 10]`.
  - Added WiMANS-compatible evaluation by mapping slot-CE predictions back to 9-bit activity vectors.
  - Added `config/wimans_multi_cnn1d_slot_ce.yaml` for CNN-1D slot-wise CE experiments.
  - Kept `official_slot_acc`, `active_slot_acc`, and `sample_exact_acc` in epoch metrics and `result.json`.
  - Reused compact prediction CSV export for readable slot-level true/pred comparisons.
- Problems found:
  - CNN-1D with BCE improved empty-slot accuracy but still underpredicted active slots.
  - The BCE independent-sigmoid formulation is a likely bottleneck because each slot is naturally a single-label `empty/activity` decision.
- Validation commands:
  - `python -m compileall train.py models utils datasets losses scripts`
  - `python train.py --config config\wimans_multi_cnn1d_slot_ce.yaml --stage v0 --sample-limit 24 --epochs 1 --batch-size 4 --scheduler-patience 1`
- Validation result:
  - Static compile passed.
  - Tiny slot-CE smoke run completed and generated config, model summary, split CSV files, metrics, compact predictions, result JSON, and best checkpoint.
  - Smoke `result.json` kept `accuracy.avg == best_official_slot_acc` and recorded the 10-class slot argmax protocol.

## v0.1.24 - 2026-05-18

- Implementation commit: `ff7965c`
- Changes:
  - Added WiMANS-style `CNN1DWiFi` model for standalone WiFi CSI experiments.
  - Added `config/wimans_multi_cnn1d.yaml` for multi-user classroom 2.4 GHz WiFi-only BCE training.
  - Added `model.wifi_backbone: cnn1d` support for `train.py --stage v0`.
  - Added `--wifi-backbone` CLI override for quick X-Fi ResNet-18 vs CNN-1D checks.
  - Documented the CNN-1D experiment command in `README.md`.
- Problems found:
  - Video-feature CAFD did not improve multi-user WiFi recognition, so a simpler WiFi-only architecture comparison is needed before adding more distillation modules.
  - The CNN-1D test should clarify whether the X-Fi ResNet-18 student architecture is itself limiting the multi-user BCE setting.
- Validation commands:
  - `python -m compileall train.py models utils datasets losses scripts`
  - `python train.py --config config\wimans_multi_cnn1d.yaml --stage v0 --sample-limit 24 --epochs 1 --batch-size 4 --scheduler-patience 1`
- Validation result:
  - Static compile passed.
  - Tiny CNN-1D smoke run completed and generated config, model summary, split CSV files, metrics, compact predictions, result JSON, and best checkpoint.
  - Smoke `result.json` kept `accuracy.avg == best_official_slot_acc`.

## v0.1.23 - 2026-05-18

- Implementation commit: `137e33f`
- Changes:
  - Added multi-user V1 CAFD config `config/wimans_multi_cafd.yaml`.
  - Added multi-user V1 RSD config `config/wimans_multi_rsd.yaml`.
  - Kept the model output and loss compatible with WiMANS multi-user activity evaluation: `[B,54]` logits with `BCEWithLogitsLoss`.
  - Kept `accuracy.avg` in `result.json` mapped to the official comparable `official_slot_acc`.
  - Added CLI overrides for `--cafd-temperature` and `--rsd-reduction`.
  - Fixed `--lambda-cafd 0.0` / positive values so the CLI override also updates `cafd.enable`.
  - Disabled misleading multi-user teacher-accuracy logging by default because the video teacher is used as a feature teacher, not a 54-output soft-label teacher.
  - Added `video.return_teacher_logits: false` for the multi-user CAFD/RSD configs to avoid computing unused S3D classifier logits.
  - Logged the loaded X-Fi WiFi ResNet-18 weight path and backbone type in new runs.
- Problems found:
  - WiFi-only multi-user BCE improved the official slot metric partly through empty-slot predictions, while active-slot accuracy remained low.
  - The current single-person video teacher is useful as a frozen feature extractor, but its logits are not directly comparable to the 54-label multi-user BCE target.
  - CAFD/RSD should therefore be tested as feature-space guidance first, before adding any extra logits KD.
- Validation commands:
  - `python -m compileall train.py models utils datasets losses scripts`
  - `python train.py --config config\wimans_multi_cafd.yaml --stage v1 --sample-limit 10 --epochs 1 --batch-size 2 --num-frames 8 --scheduler-patience 1 --lambda-cafd 0.01`
  - `python train.py --config config\wimans_multi_rsd.yaml --stage v1 --sample-limit 10 --epochs 1 --batch-size 2 --num-frames 8 --scheduler-patience 1 --lambda-rsd 1e-5`
- Validation result:
  - Static compile passed.
  - CAFD smoke run completed and wrote CAFD component values to `metrics/train_batches.csv`.
  - RSD smoke run completed and wrote RSD component values to `metrics/train_batches.csv`.
  - `result.json` kept `accuracy.avg == official_slot_acc`.

## v0.1.22 - 2026-05-18

- Implementation commit: `pending`
- Changes:
  - Kept the official WiMANS multi-user activity metric unchanged: `sigmoid(logits) > threshold`, reshape to `[N*6, 9]`, then exact 9-bit slot-vector accuracy.
  - Added auxiliary `slot_argmax_*` metrics for post-hoc diagnosis while preserving the 9-bit output space.
  - Added threshold sweep diagnostics for both official independent-sigmoid decoding and per-slot argmax decoding in `result.json`.
  - Added `--bce-pos-weight` CLI override for quick positive-class weight experiments.
  - Fixed `test.py` to pass the active config into `collect_predictions` and to save compact prediction CSV files.
- Problems found:
  - The previous multi-user BCE run stayed comparable to WiMANS, but `official_slot_acc` was strongly influenced by empty user slots.
  - Auxiliary metrics are needed to tell whether changes improve active-user predictions or only empty-slot predictions.
- Validation commands:
  - `python -m compileall train.py test.py datasets losses utils scripts`
  - `python train.py --config config\wimans_multi_bce.yaml --stage v0 --sample-limit 24 --epochs 1 --batch-size 4 --scheduler-patience 1 --bce-pos-weight 12`
- Validation result:
  - Static compile passed.
  - Tiny smoke run completed.
  - `result.json` preserved `accuracy.avg == official_slot_acc` and added diagnostic threshold sweep and slot-argmax metrics.

## v0.1.21 - 2026-05-18

- Implementation commit: `393c69b`
- Changes:
  - Added compact prediction export for readable prediction-vs-label inspection.
  - Added `splits/val_predictions_best_compact.csv` with one row per sample, per-slot true label, prediction, OK/ERR result, and a concise `wrong_slots` summary.
  - Added `scripts/simplify_predictions.py` to convert an existing detailed `val_predictions_best.csv` into a compact comparison CSV.
  - Updated `config/wimans_multi_bce.yaml` so multi-user BCE runs no longer save detailed prediction CSV files for every epoch by default.
- Problems found:
  - Detailed multi-user BCE prediction CSV files contain per-class probabilities for all six slots, which is useful for debugging but difficult to inspect manually.
  - Saving one detailed prediction CSV per epoch produced too many large files for 200-epoch experiments.
- Validation commands:
  - `python -m compileall train.py scripts utils`
  - `python scripts\simplify_predictions.py --input output\wimans_classroom_24g_multi_bce\v0\20260518_103936\splits\val_predictions_best.csv`
  - `python train.py --config config\wimans_multi_bce.yaml --stage v0 --sample-limit 24 --epochs 1 --batch-size 4 --scheduler-patience 1`
- Validation result:
  - Static compile passed.
  - Existing run `20260518_103936` now has `splits/val_predictions_best_compact.csv`.
  - Smoke run generated only best detailed and best compact prediction CSV files, with no per-epoch prediction CSV files.

## v0.1.20 - 2026-05-18

- Implementation commit: `9a21326`
- Changes:
  - Added `config/wimans_multi_bce.yaml` for a WiMANS-style multi-user WiFi-only BCE baseline.
  - Added `multi_bce` classification loss based on `BCEWithLogitsLoss` with configurable `bce_pos_weight`.
  - Added multi-user metrics:
    - `official_slot_acc`: exact slot-level accuracy over all 6 user slots, including empty slots.
    - `active_slot_acc`: exact slot-level accuracy only over non-empty true user slots.
    - `sample_exact_acc`: exact full-sample accuracy over all 54 binary outputs.
  - Added multi-user prediction export with per-slot true/predicted activities and per-class probabilities.
  - Added multi-user BCE metrics to `result.json` and `metrics/epochs.csv`.
  - Documented the new run command in `README.md`.
- Problems found:
  - Previous single-user WiFi-only and video-distillation experiments remained around 0.3-0.4 validation accuracy.
  - Normalization changes did not appear to be the main bottleneck.
  - `official_slot_acc` can be inflated by correctly predicting empty user slots, so `active_slot_acc` and `sample_exact_acc` must be checked together.
- Validation commands:
  - `python -m compileall train.py datasets models losses scripts utils`
  - `python train.py --config config\wimans_multi_bce.yaml --stage v0 --sample-limit 24 --epochs 1 --batch-size 4 --scheduler-patience 1`
  - `python -c "import torch; from losses import classification_loss; logits=torch.randn(2,54,requires_grad=True); labels=torch.zeros(2,6,9); labels[:,0,1]=1; loss=classification_loss(logits, labels, 'multi_bce', pos_weight=6.0); loss.backward(); print(float(loss.detach()), bool(torch.isfinite(logits.grad).all()))"`
- Validation result:
  - Static compile passed.
  - Tiny smoke run completed and generated `config.yaml`, `model.txt`, `model_summary.yaml`, split CSV files, `metrics/epochs.csv`, `result.json`, prediction CSV files, and `checkpoints/best.pt`.
  - BCE loss backward check passed with finite gradients.

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

## v0.1.6 - 2026-05-08

- Code commit: `1736df7`
- Changes:
  - Kept video teacher data loading as online mp4 reading; no WiMANS-style video `.npy` preprocessing is introduced.
  - Added `train.gradient_accumulation_steps` and CLI override `--grad-accum-steps`.
  - Changed `config/video_teacher.yaml` default from physical `batch_size: 8` to micro-batch `batch_size: 2` with `gradient_accumulation_steps: 4`, preserving effective batch size `8`.
  - Added log output for `video_loading=online_mp4`, micro-batch size, accumulation steps, and effective batch size.
  - Updated README commands for 16 GB friendly video teacher training.
- Problems found:
  - Training the video teacher is memory heavier than using it as a frozen teacher in V1, because unfrozen S3D must store 3D convolution activations for backpropagation.
  - `batch_size: 8` with S3D, 90 frames, 224 resolution, and full teacher backpropagation can exceed 16 GB VRAM.
- Validation commands:
  - `python -m compileall .\train_video_teacher.py`
  - No training smoke test was run after this change by request; runtime validation will be performed on the 4080S machine.
- Validation result:
  - Static compile passed before committing the change.
  - 4080S training result is pending.

## v0.1.7 - 2026-05-09

- Code commit: `3ff8953`
- Changes:
  - Updated default V1 single-person scope to mixed scenes: `classroom`, `meeting_room`, and `empty_room`.
  - Moved the trained S3D teacher checkpoint reference to the external backbone folder:
    `../backbone_models/video/video_s3d.pt`.
  - Corrected the X-Fi WiFi backbone path to `../backbone_models/WiFi/wifi_ResNet18.pt`.
  - Added loading of finetuned video teacher checkpoints in V1 CAFD training.
  - Avoided Kinetics400 download/initialization when a local teacher checkpoint is provided.
  - Added video-teacher top-3 checkpoint retention with epoch, validation accuracy, and validation loss in filenames.
  - Updated the implementation plan to the new route: trained video teacher first, then CAFD distillation, then dual-teacher BLEND-style branch, EA-KD, and optional WiFi masked reconstruction.
- Problems found:
  - The previous `best.pt` policy saved only the first epoch that reached a new best `val_acc`; when later epochs tied accuracy with lower `val_loss`, those better checkpoints were not retained.
  - The copied three-scene video teacher run kept only one checkpoint, so top-3 historical checkpoints cannot be reconstructed from logs alone.
- Validation commands:
  - `python -m compileall .\train_video_teacher.py .\train.py .\models\s3d_teacher.py .\models\video_wifi_cafd_model.py`
  - `python -c "from pathlib import Path; from models.s3d_teacher import S3DTeacher; m=S3DTeacher(weights='kinetics400', freeze=True, checkpoint_path=str(Path('..')/'backbone_models'/'video'/'video_s3d.pt')); print(m.checkpoint_extra); print(m.checkpoint_load_info)"`
- Validation result:
  - Static compile passed.
  - Local teacher checkpoint loaded successfully:
    - checkpoint extra: `epoch=16`, `val_acc=0.997222`, `backbone=s3d`, `feature_dim=1024`, `num_classes=9`
    - load info: `loaded_keys=462`, `unexpected_keys=[]`
    - missing keys are only `classifier.1.weight` and `classifier.1.bias`, which is expected for feature-only teacher loading.

## v0.1.8 - 2026-05-09

- Code commit: `e53f6af`
- Changes:
  - Added `utils/result_report.py` to write WiMANS-style `result.json` files.
  - Added per-epoch validation classification reports, per-class accuracy, per-environment accuracy, and learning-rate snapshots to `result.json`.
  - Added `result.json` output to both `train_video_teacher.py` and `train.py`, so video teacher, V0, and V1 runs share a common result summary format.
  - Clarified README text that video-teacher top-3 retention saves actual `.pt` checkpoint weight files, not only metric rows.
  - Expanded `../WiMANS_Video_WiFi_CAFD_Implementation.md` into a detailed staged implementation plan covering teacher training, CAFD distillation, dual-teacher BLEND-style expansion, EA-KD, masked reconstruction, ablations, and diagnostics.
- Problems found:
  - The previous artifacts required opening several CSV files to inspect per-class and per-scene behavior; this made long experiment comparison inconvenient.
  - The root implementation plan was too high-level for the current route after the successful trained video teacher result.
- Validation commands:
  - `python -m compileall .\utils\result_report.py .\utils\__init__.py .\train_video_teacher.py .\train.py`
- Validation result:
  - Static compile passed.

## v0.1.9 - 2026-05-09

- Code commit: `5db4864`
- Changes:
  - Added V1 video-logit distillation on top of the existing CE + CAFD route.
  - Added `losses/logits_kd_loss.py` with temperature-scaled KL divergence and optional teacher-confidence filtering.
  - Updated frozen `S3DTeacher` so the trained 9-class video classifier head is loaded from `../backbone_models/video/video_s3d.pt` instead of being skipped.
  - Updated V1 forward outputs to include both `video_feature` and `teacher_logits`.
  - Added `logits_kd` config with default `temperature: 4.0` and `lambda_logits: 0.5`.
  - Added CLI overrides for fast 4080S ablations: `--lambda-cafd`, `--lambda-logits`, and `--kd-temperature`.
  - Added V1 teacher prediction fields to validation prediction CSVs and `teacher_accuracy` / `logits_kd_loss` to per-batch logs.
  - Added `logits_kd` settings to the WiMANS-style `result.json` payload for new runs.
- Problems found:
  - Copied CAFD-only V1 run `20260509_103545` with `batch_size=8` reached best `val_acc=0.361667` at epoch 15, still far below the expected single-person HAR target.
  - Copied CAFD-only V1 run `20260509_195416` with `batch_size=16` reached best `val_acc=0.288587` in the copied log, so simply enlarging the physical CAFD relation batch did not solve the issue.
  - CAFD-only results still over-predict easy/broad classes such as `nothing` and `wave`; weak classes include `lie_down`, `rotation`, and `sit_down`.
  - The trained video teacher is strong enough, so the bottleneck is more likely the WiFi student transfer signal than teacher quality.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\train.py .\WiMANS_Baseline\models .\WiMANS_Baseline\losses .\WiMANS_Baseline\utils`
  - `python -c "import sys; sys.path.insert(0,'WiMANS_Baseline'); from models.s3d_teacher import S3DTeacher; m=S3DTeacher(weights='kinetics400', freeze=True, checkpoint_path='backbone_models/video/video_s3d.pt', num_classes=9); print(m.checkpoint_extra); print(m.checkpoint_load_info); print(m.model.classifier[1].out_channels)"`
  - `python -c "import sys, torch; sys.path.insert(0,'WiMANS_Baseline'); from models.s3d_teacher import S3DTeacher; m=S3DTeacher(weights='kinetics400', freeze=True, checkpoint_path='backbone_models/video/video_s3d.pt', num_classes=9).eval(); x=torch.randn(1,3,16,224,224); y=m(x, return_logits=True); print(tuple(y['feature'].shape), tuple(y['logits'].shape), torch.isfinite(y['logits']).all().item())"`
- Validation result:
  - Static compile passed.
  - Trained video teacher checkpoint loaded with `missing_keys=[]`, `unexpected_keys=[]`, `loaded_keys=464`, and classifier output channels `9`.
  - S3D teacher forward check returned feature shape `(1, 1024)`, logits shape `(1, 9)`, and finite logits.
  - Full V1b training is pending on the 4080S.

## v0.1.10 - 2026-05-10

- Code commit: `a4eaa91`
- Changes:
  - Changed train/validation epoch loss and accuracy aggregation from batch-average to sample-weighted aggregation.
  - Changed best-checkpoint selection to use the corrected sample-level `val_acc` returned by validation prediction collection.
  - Added V1 training-loader singleton-tail handling: when the V1 train split leaves exactly one sample after batching, the final one-sample batch is dropped.
  - Added dataloader batch count, configured batch size, and train `drop_last` status to `train.log`.
  - Documented the sample-weighted metric behavior and V1 singleton-tail handling in `README.md`.
- Problems found:
  - Complete old-code CAFD-only run `20260509_223142` finished 30 epochs with logged best `val_acc=0.321739` at epoch 11.
  - Recomputing exact sample-level validation accuracy from `val_predictions_epoch_*.csv` shows epoch 15 was actually highest at `116/357 = 0.324930`; epoch 11 was `114/357 = 0.319328`.
  - The mismatch happened because old code averaged per-batch validation accuracy, giving the smaller final validation batch the same weight as full batches.
  - The same run had `1425` train samples with `batch_size=16`, creating one singleton training batch per epoch. Those singleton batches had noisy losses, for example epoch 30 batch 90 had loss `12.253035`.
  - CAFD-only still underperforms: the complete batch-size-16 run stayed near `0.32` exact validation accuracy and did not approach the required single-person HAR target.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\train.py`
  - `python -c "import sys, yaml; sys.path.insert(0,'WiMANS_Baseline'); import train; cfg=yaml.safe_load(open('WiMANS_Baseline/config/config.yaml',encoding='utf-8')); tr,va,tdf,vdf=train.build_loaders(cfg, use_video=True); print(len(tdf),len(vdf),len(tr),len(va),tr.drop_last); tr0,va0,_,_=train.build_loaders(cfg, use_video=False); print(len(tr0),len(va0),tr0.drop_last)"`
- Validation result:
  - Static compile passed.
  - Default V1 loader check: `train=1425`, `val=357`, `train_batches=178`, `val_batches=45`, `train_drop_last=True`.
  - Default V0 loader check: `train_batches=179`, `val_batches=45`, `train_drop_last=False`.

## v0.1.11 - 2026-05-10

- Code commit: `d8ad31e`
- Changes:
  - Reduced default video-logit KD weight from `lambda_logits: 0.5` to `0.1`.
  - Added `logits_kd.warmup_epochs: 5` so the effective KD weight ramps up gradually.
  - Added CLI override `--kd-warmup-epochs`.
  - Logged `logits_kd_weighted_loss` and `lambda_logits_effective` in `metrics/train_batches.csv` and `train.log`.
  - Updated README ablation commands for conservative KD runs.
- Problems found:
  - New-code V1 run `20260510_141649` correctly loaded the trained 9-class S3D teacher head with `missing_keys=[]` and `loaded_keys=464`.
  - The same run enabled `logits_kd.lambda_logits=0.5`, but best validation accuracy was still only `0.375350` at epoch 21.
  - The frozen video teacher remained reliable: validation teacher accuracy was `356/357 = 0.997199`, and mean teacher true-class probability was about `0.980763`.
  - The weighted KD term dominated training: around epoch 21, raw `logits_kd_loss` was about `7.64`, so `0.5 * KD` contributed about `3.82`, while CE was about `1.30`.
  - This suggests the student was over-constrained by sharp video-teacher probabilities before its WiFi representation had enough class separation.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\train.py`
  - `python -c "import sys, yaml; sys.path.insert(0,'WiMANS_Baseline'); import train; cfg=yaml.safe_load(open('WiMANS_Baseline/config/config.yaml',encoding='utf-8')); kd=train.get_logits_kd_cfg(cfg); print(kd); print([train.effective_logits_kd_lambda(kd,e) for e in [1,2,5,6]])"`
- Validation result:
  - Static compile passed.
  - KD config check returned `lambda_logits=0.1`, `warmup_epochs=5`.
  - Effective KD lambda schedule: epoch 1 `0.02`, epoch 2 `0.04`, epoch 5 `0.1`, epoch 6 `0.1`.

## v0.1.13 - 2026-05-11

- Revert commits: `b32c5f0`, `3b793c8`
- Changes:
  - Reverted the temporary WiMANS official-style `cnn2d/that` baseline integration from v0.1.12.
  - Removed `models/wimans_wifi_models.py` and the related `--wifi-model` training CLI path.
  - Restored the current implementation direction to X-Fi WiFi ResNet-18 as the only WiFi student backbone.
- Problems found:
  - The user wants to continue the project with X-Fi ResNet-18 rather than switching to WiMANS official WiFi models.
  - The current X-Fi ResNet-18 baseline remains below the required single-person HAR standard, so follow-up work should improve the teacher/distillation/input auxiliary branch without replacing the student backbone.
- Validation commands:
  - `git status --short --branch`
  - `Test-Path .\WiMANS_Baseline\models\wimans_wifi_models.py`
  - `Select-String -Path .\WiMANS_Baseline\train.py -Pattern "wifi-model|WiMANSWiFi|cnn2d|that|no-flops|normalize"`
- Validation result:
  - The working tree returned to the X-Fi ResNet-18 main path.
  - `models/wimans_wifi_models.py` no longer exists.
  - The temporary WiMANS WiFi-model CLI switches are absent from `train.py`.

## v0.1.14 - 2026-05-11

- Implementation commit: `844df40`
- Changes:
  - Reset the invalid feature-KD-only `v0.1.14` route back to `v0.1.13` and removed the old local tag before rebuilding this version.
  - Replaced the simplified CAFD implementation with a CMAD-style relation distillation loss.
  - Added weighted MSE, student-student/teacher-teacher correlation alignment, diagonal gap, and bidirectional KL components inside `CAFDLoss`.
  - Kept logits KD as the soft-label distillation signal with the conservative default `lambda_logits=0.1`, `temperature=4.0`, and `warmup_epochs=5`.
  - Added CAFD component values to batch logging for diagnosis.
- Problems found:
  - The previous CAFD loss was likely too simple to represent the richer relation weighting used by CMAD-style public code.
  - Standalone feature KD would not directly answer whether the original CAFD simplification was the bottleneck, so it was removed from this route.
- Validation commands:
  - `python -m compileall train.py models utils losses datasets scripts`
  - `python -c "import torch; from losses.cafd_loss import CAFDLoss; ..."`
- Validation result:
  - Static compile passed.
  - CAFD finite/backward smoke check passed for `[8,256]` features.
  - Student features received gradients, teacher features stayed detached, and batch size 1 remained finite.

## v0.1.15 - 2026-05-12

- Implementation commit: `089118c`
- Changes:
  - Reviewed the latest interrupted CMAD-style CAFD run `20260511_144013`; best validation accuracy stayed at `0.324930` on epoch 15.
  - Added Redundancy Suppression Distillation from `RSD-main` as a V1 feature-level loss.
  - Added `losses/rsd_loss.py` with batch-normalized teacher-student cross-correlation, diagonal invariance maximization, and off-diagonal redundancy suppression.
  - Added `rsd` config and CLI overrides: `--lambda-rsd`, `--rsd-kappa`, and `--rsd-warmup-epochs`.
  - Added RSD raw/weighted loss and correlation diagnostics to `train.log` and `metrics/train_batches.csv`.
- Problems found:
  - The CMAD-style CAFD term remained small after `lambda_cafd=0.1`, while logits KD still dominated the extra supervision signal.
  - RSD is a better match for the current cross-modal/cross-architecture mismatch because it suppresses architecture-exclusive redundant feature dimensions rather than only matching sample relations or class probabilities.
- Validation commands:
  - `python -m compileall train.py models utils losses datasets scripts`
  - `python -c "import torch; from losses.rsd_loss import RSDLoss; ..."`
- Validation result:
  - Static compile passed.
  - RSD finite/backward smoke check passed for `[8,256]` projected features.
  - With random `[8,256]` features, raw RSD was about `395.59`, so default `lambda_rsd=0.001` contributes about `0.396` before warmup scaling.
  - Teacher features stayed detached, student features received gradients, and batch size 1 returns a finite zero loss.

## v0.1.16 - 2026-05-12

- Implementation commit: `aaf6447`
- Changes:
  - Changed the default V1 feature-distillation target from `video_projected` to the frozen S3D teacher raw `video_feature`.
  - Kept `video_projector` in `VideoWiFiCAFDModel`, but froze it by default with `projector.freeze_video_projector: true`.
  - Made the WiFi student projector output match the S3D teacher feature dimension when `projector.target: video_feature`.
  - Added generic `wifi_distill_feature` and `teacher_distill_feature` outputs so CAFD and RSD use the same stable teacher target.
  - Changed the default RSD source to `distill`, which follows the currently selected feature-distillation pair.
  - Logged the active projector target, WiFi projector output dimension, and trainable video-projector parameter count at run start.
- Problems found:
  - The interrupted RSD run `20260512_095804` was still near the low `0.3` validation-accuracy range.
  - Its CAFD/RSD target was `video_projected`, but `video_projector` was randomly initialized and teacher-side losses detached the target, so the WiFi student was aligning to an unstable random projection instead of the trained S3D feature.
  - This does not match the public RSD pattern, where the student side is projected into the teacher feature space while the teacher feature remains fixed.
- Validation commands:
  - `python -m compileall train.py models utils losses datasets scripts`
  - `python -c "import torch; from losses.cafd_loss import CAFDLoss; from losses.rsd_loss import RSDLoss; ..."`
  - `python -c "import yaml, train; cfg=yaml.safe_load(open('config/config.yaml',encoding='utf-8')); cfg['video']['s3d_weights']='none'; cfg['video']['teacher_checkpoint']=None; m=train.build_model(cfg,'v1'); ..."`
- Validation result:
  - Static compile passed.
  - CAFD/RSD finite/backward smoke check passed for `[8,1024]` features.
  - Student features received gradients, teacher features stayed detached, and batch size 1 RSD returned a finite zero loss.
  - Default V1 model construction check returned `projector_target=video_feature`, WiFi projector output dimension `1024`, and `video_trainable=0`.

## v0.1.17 - 2026-05-13

- Implementation commit: `ed59583`
- Changes:
  - Added `ProjectedVideoTeacherClassifier` for supervised video-projector teacher training.
  - Extended `train_video_teacher.py` with `video_teacher.mode: classifier/projector` and CLI overrides:
    - `--mode projector`
    - `--checkpoint`
    - `--projector-hidden-dim`
    - `--projector-out-dim`
    - `--projector-num-heads`
  - Projector mode loads a trained video checkpoint, freezes the video teacher, and trains only `video_projector + projector_classifier`.
  - Saved projector-teacher checkpoints include both frozen video-teacher weights and trained `video_projector` weights.
  - Updated V1 loading so `VideoWiFiCAFDModel` can load `video_projector.*` from the same teacher checkpoint used by `S3DTeacher`.
  - Added `--projector-target` to `train.py` so projected-teacher distillation can be selected from the command line.
  - Updated README with projector-teacher training and projected-feature distillation commands.
- Problems found:
  - Direct alignment to raw S3D `video_feature` stayed near the low `0.3` validation-accuracy range.
  - The original S3D teacher checkpoint does not contain a trained projector, so `video_projector` must first be trained with video labels before being used as a stable teacher feature space.
  - Simply unfreezing `video_projector` inside WiFi distillation would not update it under the current CAFD/RSD losses because teacher-side features are detached.
- Validation commands:
  - `python -m compileall train.py train_video_teacher.py models utils losses datasets scripts`
  - `python -c "import yaml, train_video_teacher as tv; ..."`
  - `python -c "import yaml, torch, train, train_video_teacher as tv; ..."`
- Validation result:
  - Static compile passed.
  - Projector mode instantiated from `../backbone_models/video/video_s3d.pt` with `base_feature_dim=1024`, `projector_out_dim=256`, and `feature_dim=256`.
  - Projector mode optimizer contained only the head group with `594697` trainable parameters; frozen video-teacher trainable parameters were `0`.
  - V1 projected-target load test loaded `12` video-projector keys and `464` S3D teacher keys from a temporary projector-teacher checkpoint.

## v0.1.18 - 2026-05-13

- Implementation commit: `7f366a5`
- Changes:
  - Fixed S3D projector-teacher feature extraction in `VideoTeacherClassifier.forward`.
  - Classification-head input features with shape `[B,C,T,H,W]` are now mean-pooled over temporal/spatial dimensions to `[B,C]` before entering `video_projector`.
- Problems found:
  - Projector-teacher training crashed with `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x10240 and 1024x256)`.
  - The hook captured the S3D classifier input and flattened all non-batch dimensions, producing `10240`-d features instead of the intended `1024`-d pooled S3D feature.
- Validation commands:
  - `python -m compileall models/video_teacher.py train_video_teacher.py`
  - `python -c "import yaml, torch, train_video_teacher as tv; ..."`
- Validation result:
  - Static compile passed.
  - Projector-teacher forward check returned logits shape `(2, 9)`, projected feature shape `(2, 256)`, base feature shape `(2, 1024)`, and finite logits.

## v0.1.19 - 2026-05-13

- Implementation commit: `dbe15aa`
- Changes:
  - Updated V1 projected-teacher distillation so `projector_target: projected` loads both trained `video_projector.*` and `projector_classifier.*` from the teacher checkpoint.
  - When `projector.use_projector_logits: true`, logits KD now uses the trained projected-teacher classifier logits instead of the raw S3D classifier logits.
  - Added a fail-fast check: `projector_target: projected` now raises an error if the checkpoint does not contain trained projector weights, and projected logits KD raises an error if the checkpoint does not contain `projector_classifier.*`.
  - Kept `projector_classifier` frozen and in eval mode during V1 student training so dropout does not randomize teacher logits.
  - Logged `projector_classifier_checkpoint_load_info`, `use_projector_logits`, and `teacher_logits_source` in `train.log`.
  - Documented the projected-teacher command and shifted the next recommended run to CAFD-first, RSD-off: `--lambda-cafd 0.5 --lambda-logits 0.05 --lambda-rsd 0.0`.
- Problems found:
  - The copied run `20260513_150741` still peaked at only `val_acc=0.336134` on epoch 17, with epoch 19 at `val_acc=0.302521`.
  - Its projected-feature run loaded `video_projector.*` correctly, but logits KD was still sourced from the raw S3D classifier, so feature and soft-label supervision came from different teacher heads.
  - In epoch 17 and 19, `lambda_cafd=0.1` made CAFD contribute only about `0.054` weighted loss, while logits KD contributed about `0.83`; the relation feature signal was too small relative to soft logits KD.
  - Best validation predictions were still biased toward `wave` and `nothing`: `wave=101` predictions and `nothing=66`, while `stand_up` class accuracy was only `0.10` and `lie_down` only `0.125`.
  - The locally copied projector-teacher checkpoint files under `output/wimans_5g_single_video_teacher/20260513_113237/checkpoints/` could not be opened by `torch.load` (`PytorchStreamReader failed reading zip archive`), so they appear incomplete on this machine. Use the complete 4080S checkpoint for the next run.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\models\video_wifi_cafd_model.py .\WiMANS_Baseline\train.py`
  - `python -c "import sys, tempfile, pathlib, torch; ..."`
  - `Import-Csv .\WiMANS_Baseline\output\wimans_5g_single_baseline\v1\20260513_150741\metrics\epochs.csv | Sort-Object @{Expression={[double]$_.val_acc};Descending=$true} | Select-Object -First 5`
- Validation result:
  - Static compile passed.
  - Synthetic projector checkpoint load test returned `teacher_logits_source=projector_classifier`, `projector_load=12`, and `classifier_load=2`.
  - `model.train()` keeps both frozen `video_projector` and `projector_classifier` in eval mode.
  - Latest copied projected-teacher run analysis confirmed no meaningful improvement over the raw-feature CAFD range, so RSD should remain off for the next run until the projected teacher logits path is tested.

## v0.1.20 - 2026-05-14

- Implementation commits: `9a2c6b7`, `03d5070`
- Changes:
  - Reworked `losses/cafd_loss.py` to match the paper-style CAFD weighted-MSE branch used in the CMAD public code.
  - CAFD weighted-MSE now uses raw student/teacher features for MSE, while cosine-normalized features are used only for similarity matrices.
  - Removed the extra `student_student_similarity` correlation branch from the active CAFD loss.
  - Deprecated `alpha`, `beta`, `gamma`, `use_weighted_mse`, and `use_correlation` from the default CAFD config and training log.
  - The active CAFD formula is now exactly `loss = weighted_mse + diagonal_gap`, where `weighted_mse = mean(bi_kl(sim_stu_tea/tau, sim_tea_tea/tau) * mse_per_sample)`.
  - Kept the historical `cafd_correlation` CSV/log field as `0.0` for compatibility with previous analysis scripts.
  - Updated `scripts/smoke_v1.py` so it no longer expects removed CAFD config keys.
  - Changed the default V1 config to CAFD-only for the next clean run: `lambda_cafd=0.5`, `lambda_logits=0.0`, and `lambda_rsd=0.0`.
  - Kept logits KD and RSD code paths available as explicit ablations, but removed them from the default CAFD validation run.
- Problems found:
  - The copied run `20260513_215214` correctly used `teacher_logits_source=projector_classifier`, but still peaked at only `val_acc=0.350140` on epochs 31 and 34.
  - That run used `lambda_cafd=0.5`, `lambda_logits=0.05`, and `lambda_rsd=0.0`; average weighted CAFD contributed about `0.25-0.34`, while weighted logits KD contributed about `0.18-0.20` after warmup.
  - Because CAFD already uses relation-distribution KL on features, adding category-distribution soft-label KD is not mathematically identical but can blur diagnosis. The next run should isolate CAFD first.
  - Best predictions remained class-biased: `nothing` stayed near perfect, but `stand_up` was only `0.05`, `rotation` `0.1282`, and `sit_down` `0.1795`.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\losses\cafd_loss.py .\WiMANS_Baseline\train.py`
  - `python -m compileall .\WiMANS_Baseline\losses\cafd_loss.py .\WiMANS_Baseline\train.py .\WiMANS_Baseline\scripts\smoke_v1.py`
  - `python -c "import importlib.util, pathlib, sys, torch; ..."`
  - `Import-Csv .\WiMANS_Baseline\output\wimans_5g_single_baseline\v1\20260513_215214\metrics\epochs.csv | Sort-Object @{Expression={[double]$_.val_acc};Descending=$true} | Select-Object -First 8`
- Validation result:
  - Static compile passed.
  - Random-feature comparison against `CMAD-main/CMAD_sentiment/Student_Model/loss.py` matched exactly for `compute_weighted_mse_loss`: absolute difference `0.0`.
  - New CAFD details returned `cafd_correlation=0.0`; the active loss matched `cafd_weighted_mse + cafd_diagonal_gap`.
  - Student features received finite gradients.
  - Batch size 1 remains finite for local smoke checks, returning the direct MSE fallback instead of the original CMAD relation-matrix division by zero.

## v0.1.21 - 2026-05-14

- Implementation commit: `cb9d4e7`
- Changes:
  - Removed the logits-KD loss path from the active project code.
  - Deleted `losses/logits_kd_loss.py` and removed its export from `losses/__init__.py`.
  - Removed logits-KD CLI arguments from `train.py`: `--lambda-logits`, `--kd-temperature`, and `--kd-warmup-epochs`.
  - Removed logits-KD config from `config/config.yaml` and removed logits-KD metadata from new `result.json` payloads.
  - Removed logits-KD loss computation, per-batch CSV columns, model-name suffix, and train-log lines from new V1 runs.
  - Updated README so the current route is CE + paper-style CAFD, with RSD kept only as a later optional ablation.
- Problems found:
  - CAFD contains a feature-relation KL term, while logits KD used a class-probability KL term. They are not identical, but previous experiments showed logits KD did not provide a decisive gain and made CAFD-only diagnosis harder.
  - The current paper-style CAFD-only run `20260514_130639` reached `val_acc=0.378151` by epoch 17, which is a small improvement but still far from the target. This supports reducing CAFD weight first, then moving to a WiFi heatmap teacher branch if the result remains low.
  - RSD should not be the next main direction because it is another feature-correlation/distillation regularizer; if CAFD already fails to transfer enough class-discriminative structure, RSD is more likely a small ablation than the main fix.
- Validation commands:
  - `python -m compileall .\WiMANS_Baseline\train.py .\WiMANS_Baseline\losses .\WiMANS_Baseline\utils .\WiMANS_Baseline\scripts\smoke_v1.py`
  - `rg -n "logits_kd|lambda_logits|kd_|logit_kd" .\WiMANS_Baseline\train.py .\WiMANS_Baseline\config\config.yaml .\WiMANS_Baseline\README.md .\WiMANS_Baseline\losses .\WiMANS_Baseline\utils .\WiMANS_Baseline\scripts`
  - `python -c "import sys, yaml; sys.path.insert(0, r'D:\PYTHON\Project\WiMANS_fuse\WiMANS_Baseline'); import train; ..."`
- Validation result:
  - Static compile passed.
  - Active code/config/docs search returned no logits-KD references.
  - Config check returned `has_logits_kd=False`; RSD config parsing still works.

## v0.1.22 - 2026-05-16

- Implementation commit: `f738248`
- Changes:
  - Added `XFiWiFiOriginalFC`, an optional WiFi student that follows the original X-Fi ResNet-18 forward path and replaces only the final `fc` layer with a 9-class classifier.
  - Added `--wifi-student {token_pool,original_fc}` for direct V0/V1 comparison between the token-pool route and the original X-Fi classifier route.
  - Added scheduler CLI overrides: `--epochs`, `--batch-size`, `--scheduler-factor`, `--scheduler-patience`, and `--scheduler-min-lr`.
- Problems found:
  - The original X-Fi FC route did not improve the WiFi-only ceiling. The copied V0 run `20260516_195909` reached only `best_val_acc=0.380952`, while the previous token-pool V0 run `20260516_133121` reached `best_val_acc=0.392157`.
  - This suggests the low single-person WiFi result is not mainly caused by the token-pool head implementation.
- Validation commands:
  - `python -m compileall train.py models losses datasets scripts`
  - `python -c "from models import XFiWiFiOriginalFC; ..."`
- Validation result:
  - Static compile passed.
  - `XFiWiFiOriginalFC` output matched the wrapped X-Fi backbone output exactly in eval mode.

## v0.1.23 - 2026-05-17

- Implementation commit: `c153731`
- Changes:
  - Changed the default WiFi amplitude preprocessing from `log1p_zscore` to `none`.
  - Added `train.py --normalize {none,zscore,log1p_zscore}` so normalization can be controlled from the command line and recorded in each run's saved config.
  - Updated README to document that `../dataset/wifi_csi/amp/*.npy` already contains preprocessed CSI amplitude and should be used directly for the X-Fi input-distribution check.
- Problems found:
  - The project was reading the processed amplitude `.npy` files but then applying `log1p_zscore` in `datasets/wifi_amp_loader.py`.
  - WiMANS official WiFi code and X-Fi data loaders read the amplitude arrays directly and do not apply this extra per-sample log z-score normalization at load time.
  - The extra normalization can shift the input distribution away from the X-Fi WiFi ResNet-18 pretraining distribution.
- Validation commands:
  - `python -m compileall train.py datasets models losses scripts utils`
  - `python -c "import argparse, train; ..."`
- Validation result:
  - Static compile passed.
  - CLI config override check confirmed `--normalize none` and `--normalize log1p_zscore` update `cfg["data"]["normalize"]` as expected.

## Suggested Future Milestones

- `v0.2.0`: WiMANS data checks and label builder validated.
- `v0.3.0`: V0 WiFi-only smoke test validated.
- `v0.4.0`: V1 online S3D + CAFD smoke test validated.
- `v1.0.0`: Full V0/V1 training config stable on 4080S.
