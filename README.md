# WiMANS_Baseline

WiMANS_Baseline is an independent project for WiMANS video-to-WiFi HAR experiments.
It references the local dataset, papers, official WiMANS code, X-Fi code, and X-Fi
WiFi backbone weights from the parent directory without tracking those large external
resources in git.

## Goals

- V0: run a stable 5 GHz single-user WiFi-only HAR baseline with X-Fi WiFi ResNet-18 initialization.
- Video teacher: train a selectable WiMANS-style visual teacher before distillation.
- V1: add frozen trained video teacher, Hybrid Projector, CAFD feature relation distillation, and optional RSD diagnostics.
- Keep the local laptop workflow focused on smoke tests and small runs.
- Move the whole project, including `.git/`, to the 4080S machine for full training.

The global random seed is fixed to `39` for Python, NumPy, PyTorch, and CUDA.

## Directory Layout

```text
WiMANS_Baseline/
|-- config/
|   |-- config.yaml
|   `-- video_teacher.yaml
|-- datasets/
|-- losses/
|-- models/
|-- scripts/
|-- utils/
|-- train.py
|-- train_video_teacher.py
|-- test.py
|-- README.md
|-- CHANGELOG.md
|-- GIT_WORKFLOW.md
`-- .gitignore
```

## External Paths

Default paths in `config/config.yaml` are relative to this project directory:

```text
../dataset
../dataset/annotation.csv
../backbone_models/WiFi/wifi_ResNet18.pt
../backbone_models/video/video_s3d.pt
```

WiFi inputs are loaded from the already preprocessed amplitude files under
`../dataset/wifi_csi/amp/*.npy`. The default `data.normalize` is `none`, so the
loader only pads/truncates each sample and reshapes it to `[270, target_len]`
before sending it to the X-Fi WiFi ResNet. Use `--normalize zscore` or
`--normalize log1p_zscore` only for explicit ablation runs.

The official reference folders remain outside git:

```text
../WiMANS-main
../X-Fi-main
../reference_paper
../WiMANS_Data_Info
```

## Environment

In your own terminal:

```powershell
conda activate WiMANS
cd D:\PYTHON\Project\WiMANS_fuse\WiMANS_Baseline
```

In this Codex sandbox, `conda activate WiMANS` is blocked by the command sandbox, so checks are run with:

```powershell
D:\SoftWare\ANACONDA\envs\WiMANS\python.exe
```

That points to the same WiMANS environment with PyTorch installed.

## Data Checks

```powershell
python scripts\inspect_data.py --config config\config.yaml
```

Expected key checks:

- annotation rows: `11286`
- 5 GHz single-user rows: `1782`
- 9 activity classes: `198` samples each
- all selected WiFi amplitude and video files exist by `label`

## Smoke Tests

V0 WiFi-only smoke test:

```powershell
python scripts\smoke_v0.py --config config\config.yaml --limit 8 --batch-size 2
```

V1 online S3D + CAFD smoke test:

```powershell
python scripts\smoke_v1.py --config config\config.yaml --limit 1 --num-frames 16 --s3d-weights none
```

The V1 smoke command uses `s3d_weights=none` to avoid a pretrained-weight download on the laptop.
For real V1 training on the 4080S machine, use `s3d_weights: kinetics400` in the config.

Video teacher smoke test:

```powershell
python train_video_teacher.py --config config\video_teacher.yaml --weights none --sample-limit 4 --num-frames 16 --epochs 1 --batch-size 2 --no-flops
```

This command checks online video loading, model forward/backward, validation prediction
export, and checkpoint saving. S3D needs at least 16 frames for this tiny smoke test.

## Training

V0:

```powershell
python train.py --config config\config.yaml --stage v0
```

For the current X-Fi input-distribution check, keep the direct amplitude setting:

```powershell
python train.py --config config\config.yaml --stage v0 --wifi-student original_fc --normalize none --epochs 100 --batch-size 16 --scheduler-patience 20 --scheduler-min-lr 1e-6
```

The shared config currently targets 4080S-side comparison runs. For a laptop smoke
run, use `--sample-limit` or copy the config and reduce `train.batch_size`,
`train.epochs`, and `video.num_frames`.

V1:

```powershell
python train.py --config config\config.yaml --stage v1
```

The default V1 config now uses the trained teacher at
`../backbone_models/video/video_s3d.pt` for the CAFD feature-relation signal:

- CMAD-style CAFD relation distillation with `cafd.lambda_cafd`.
- RSD feature redundancy suppression with `rsd.lambda_rsd` is kept as an optional feature-correlation ablation.

The current default feature target is `projector.target: video_feature`.
This keeps the trained S3D teacher frozen and maps the WiFi student feature into
the raw S3D 1024-d feature space. The retained `video_projector` is frozen by
default through `projector.freeze_video_projector: true` and is not used as the
CAFD/RSD teacher target in this experiment.

The current default is CAFD-only: `rsd.lambda_rsd: 0.0`. This avoids mixing
other distillation losses with CAFD while checking whether the paper-style CAFD
feature relation loss can improve the X-Fi ResNet-18 WiFi student by itself.
The CAFD loss follows the original formula used here for diagnosis:
`loss = weighted_mse + diagonal_gap`. It does not use the extra
student-student correlation branch or internal `alpha/beta/gamma` weights.
For a WiFi-only same-split ablation, run `--stage v0` with the same data section.
The previous logits-KD path was removed from the active project because it did
not produce a reliable improvement and made CAFD-only diagnosis less clean.

Useful 4080S ablation commands:

```powershell
python train.py --config config\config.yaml --stage v1
python train.py --config config\config.yaml --stage v1 --lambda-cafd 0.2 --lambda-rsd 0.0
python train.py --config config\config.yaml --stage v1 --lambda-cafd 0.5 --lambda-rsd 0.0001 --rsd-warmup-epochs 5
```

Run the default CAFD-only V1 command first. The earlier projected-target run
`20260512_095804` still improved slowly and only approached the low `0.3` range,
because CAFD/RSD were aligned to `video_projected`, a retained but untrained
video projection head. The current default instead aligns the WiFi student to
the stable S3D teacher feature. If this remains far below the single-person HAR
target, move next to a WiFi heatmap teacher instead of spending more time on
CAFD-only component ablations.

## WiMANS Multi-User BCE Check

The WiMANS official WiFi activity task uses six user slots and nine activity
labels per slot. The initial multi-user check is WiFi-only and uses
`BCEWithLogitsLoss` with logits shaped as `[B, 54]`, reshaped to `[B, 6, 9]`
for metrics.

```powershell
python train.py --config config\wimans_multi_bce.yaml --stage v0
```

  The primary metric is `official_slot_acc`, matching the WiMANS-style exact
  9-way vector match per user slot after `sigmoid(logits) > threshold`. The run
  also saves `sample_exact_acc`, `active_slot_acc`, per-slot prediction columns,
  and per-activity slot accuracy in `result.json`. For readability, this config
  does not save detailed prediction CSV files for every epoch. It saves the best
  detailed file as `splits/val_predictions_best.csv` and a compact comparison
  file as `splits/val_predictions_best_compact.csv`.

  `result.json` keeps WiMANS comparability by setting `accuracy.avg` to the
  official slot-level activity accuracy. Additional fields such as
  `slot_argmax_*` and `threshold_sweep` are diagnostic only; they do not replace
  the official `sigmoid(logits) > threshold` result.

  Positive-class BCE weight can be changed from the command line for diagnosis:

  ```powershell
  python train.py --config config\wimans_multi_bce.yaml --stage v0 --bce-pos-weight 12
  ```

  Multi-user V1 distillation keeps the same `[B, 54]` BCE activity head and the
  same official WiMANS metric. The video branch supplies only feature-space
  guidance; multi-user teacher logits are not used as a soft-label KD target.

  CAFD feature-relation distillation:

  ```powershell
  python train.py --config config\wimans_multi_cafd.yaml --stage v1
  ```

  RSD feature redundancy-suppression distillation:

  ```powershell
  python train.py --config config\wimans_multi_rsd.yaml --stage v1
  ```

  Both configs use `../backbone_models/video/video_s3d.pt`,
  `projector.target: video_feature`, `bce_pos_weight: 6.0`, and
  `data.normalize: none`. If the 4080S run is close to memory limits, first try
  `--batch-size 4`; if video decoding is the bottleneck, try `--num-frames 64`.

  To compact an existing detailed prediction file:

  ```powershell
  python scripts\simplify_predictions.py --input output\wimans_classroom_24g_multi_bce\v0\<run_id>\splits\val_predictions_best.csv
  ```

  For the supervised projected-video teacher route, use a checkpoint trained in
`train_video_teacher.py --mode projector`. When `--projector-target projected`
is selected, V1 now requires the checkpoint to contain trained
`video_projector.*` weights. The trained `projector_classifier.*` logits are
kept for teacher prediction diagnostics, but they are not used as an additional
soft-label KD loss.

Laptop-sized V1 training check:

```powershell
python train.py --config config\config.yaml --stage v1 --sample-limit 2 --num-frames 16 --s3d-weights none
```

The command above is only for checking the training path on the 3050 laptop.
Use the default `s3d_weights: kinetics400` and larger data settings on the 4080S.

Video teacher:

```powershell
python train_video_teacher.py --config config\video_teacher.yaml --model S3D
```

The default video-teacher config keeps online mp4 loading and uses a 16 GB friendly
micro-batch:

```yaml
train:
  batch_size: 4
  gradient_accumulation_steps: 4
```

This gives an effective batch size of `16` without storing all sixteen videos'
backpropagation activations in GPU memory at once. If the 4080S still runs out of
memory, use:

```powershell
python train_video_teacher.py --config config\video_teacher.yaml --model S3D --batch-size 2 --grad-accum-steps 8
```

The video teacher follows the official WiMANS-style model switch. Supported names:

```text
S3D
ResNet
MViT-v1
MViT-v2
Swin-T
Swin-S
```

Torchvision-style backbone names also work, for example:

```powershell
python train_video_teacher.py --config config\video_teacher.yaml --backbone r3d_18
python train_video_teacher.py --config config\video_teacher.yaml --backbone swin3d_t --batch-size 2
```

The trained teacher checkpoint is saved under:

```text
output/wimans_5g_single_video_teacher/video_teacher/<RUN_ID>/checkpoints/best.pt
```

For distillation, copy the selected teacher checkpoint into the external backbone
folder and keep this path stable:

```text
../backbone_models/video/video_s3d.pt
```

`config/config.yaml` uses that path by default through `video.teacher_checkpoint`.
Video-teacher training also keeps the top 3 validation checkpoint weight files.
Filenames include epoch, validation accuracy, and validation loss, with
`top_k_checkpoints.csv` recording their ranking.

Projected video teacher:

```powershell
python train_video_teacher.py --config config\video_teacher.yaml --mode projector --checkpoint ../backbone_models/video/video_s3d.pt --weights none --epochs 20 --batch-size 4 --grad-accum-steps 4 --projector-out-dim 256
```

This loads the trained S3D teacher checkpoint, freezes the video teacher, and
trains only `video_projector + projector_classifier`. The saved checkpoint keeps
both the frozen S3D weights and the trained `video_projector` weights, so it can
be copied into the external backbone folder and used directly by V1.

To distill from the trained projected teacher space:

```powershell
python train.py --config config\config.yaml --stage v1 --teacher-checkpoint ../backbone_models/video/video_projector_s3d.pt --projector-target projected --lambda-cafd 0.2 --lambda-rsd 0.0
```

Use that checkpoint later as the visual teacher branch for the first distillation
experiment. Keep `--weights none` only for smoke tests; real teacher training should
use `weights: kinetics400`.

## Testing And Predictions

Evaluate a saved checkpoint and write prediction-vs-ground-truth rows into the run
`splits/` directory:

```powershell
python test.py --config output\wimans_5g_single_baseline\v0\<RUN_ID>\config.yaml --stage v0 --checkpoint output\wimans_5g_single_baseline\v0\<RUN_ID>\checkpoints\best.pt
```

The default output is:

```text
output/wimans_5g_single_baseline/v0/<RUN_ID>/splits/test_predictions.csv
```

Prediction CSV files include `sample_id`, true class, predicted class, correctness,
loss, and per-class probabilities. V1 prediction files also include frozen video
teacher prediction fields and teacher per-class probabilities.

## Head Strategy Check

The default V0 model follows the X-Fi single-modality style: X-Fi WiFi feature
extractor tokens are mean-pooled, normalized with `LayerNorm`, and passed through
a new `Linear(512, 9)` head. A direct `backbone.fc = Linear(512, 9)` comparison
is available for sanity checks:

```powershell
python scripts\compare_head_strategies.py --config config\config.yaml --sample-limit 90 --epochs 2 --batch-size 1
```

This comparison is a plumbing and early-signal check, not a final accuracy benchmark.

## Run Outputs

Every training run creates a unique timestamped directory:

```text
output/<experiment.name>/<stage>/<YYYYMMDD_HHMMSS>/
```

Each run directory contains:

```text
config.yaml                 # effective config after CLI overrides
train.log                   # config, split paths, model structure, params/FLOPs, training details
model.txt                   # full model structure
model_summary.yaml          # parameter counts and MAC/FLOP estimates
result.json                 # WiMANS-style metrics plus per-epoch/per-class/per-scene details
splits/train.csv            # saved training split
splits/val.csv              # saved validation split
splits/val_predictions_*.csv # validation prediction-vs-ground-truth files
splits/test_predictions.csv # test.py prediction-vs-ground-truth file
metrics/train_batches.csv   # per-batch training loss/accuracy details
metrics/epochs.csv          # per-epoch train/validation summary
checkpoints/best.pt         # best validation checkpoint
checkpoints/epoch_*_acc_*_loss_*.pt # top-3 video teacher checkpoint weights
checkpoints/top_k_checkpoints.csv # top checkpoint ranking for video teacher runs
```

For V1 runs, `metrics/train_batches.csv` records `classification_loss`,
`cafd_loss`, CAFD component values, optional RSD values, student accuracy, and
frozen teacher accuracy.
Historical `output/` folders are not backfilled; use new run directories for the
current result format.

Training and validation epoch metrics are sample-weighted. V1 training also drops
a singleton tail batch when the training split leaves exactly one sample after
batching, because CAFD relation matrices are not meaningful for a one-sample
training batch.

Current 3050 validation run:

```text
output/wimans_5g_single_baseline/v0/20260506_103030
```

That run used full 5 GHz single-user data with `batch_size: 1` and `epochs: 8`,
saving `1425` training rows and `357` validation rows. Best validation accuracy
was `0.131653` at epoch 2; the training path is functional, but this laptop run
does not yet show stable generalization.

## 4080S Transfer Notes

Copy the entire `WiMANS_Baseline/` folder, including the hidden `.git/` directory.
Then copy or mount the external resources so the config paths resolve:

```text
dataset/
backbone_models/WiFi/wifi_ResNet18.pt
backbone_models/video/video_s3d.pt
```

On the 4080S machine, update these config fields first:

- `data.root`
- `data.annotation`
- `model.xfi_weight_path`
- `train.batch_size`
- `train.num_workers`
- `train.epochs`
- `video.s3d_weights`
- `video_teacher.weights`
- `video_teacher.backbone`

Verify version history after transfer:

```powershell
git status
git log --oneline --decorate -n 10
```
