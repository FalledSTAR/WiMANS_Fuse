# WiMANS_Baseline

WiMANS_Baseline is an independent project for WiMANS video-to-WiFi HAR experiments.
It references the local dataset, papers, official WiMANS code, X-Fi code, and X-Fi
WiFi backbone weights from the parent directory without tracking those large external
resources in git.

## Goals

- V0: run a stable 5 GHz single-user WiFi-only HAR baseline with X-Fi WiFi ResNet-18 initialization.
- Video teacher: train a selectable WiMANS-style visual teacher before distillation.
- V1: add frozen trained video teacher, Hybrid Projector, CAFD feature relation distillation, and video-logits KD.
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

The shared config currently targets 4080S-side comparison runs. For a laptop smoke
run, use `--sample-limit` or copy the config and reduce `train.batch_size`,
`train.epochs`, and `video.num_frames`.

V1:

```powershell
python train.py --config config\config.yaml --stage v1
```

The default V1 config now uses the trained teacher at
`../backbone_models/video/video_s3d.pt` for three signals:

- CMAD-style CAFD relation distillation with `cafd.lambda_cafd`.
- Teacher-class probability distillation with `logits_kd.lambda_logits`, which is soft-label KD.
- RSD feature redundancy suppression with `rsd.lambda_rsd`.

The current default feature target is `projector.target: video_feature`.
This keeps the trained S3D teacher frozen and maps the WiFi student feature into
the raw S3D 1024-d feature space. The retained `video_projector` is frozen by
default through `projector.freeze_video_projector: true` and is not used as the
CAFD/RSD teacher target in this experiment.

For a CAFD-only ablation, set `logits_kd.lambda_logits: 0.0` and
`rsd.lambda_rsd: 0.0` in a copied config.
For a WiFi-only same-split ablation, run `--stage v0` with the same data section.
For a logits-only ablation, set `cafd.lambda_cafd: 0.0` and keep
`logits_kd.lambda_logits` enabled while setting `rsd.lambda_rsd: 0.0`.
The default logits KD is intentionally conservative: `lambda_logits: 0.1` with
`warmup_epochs: 5`. Earlier `lambda_logits: 0.5` runs made the weighted KD term
much larger than CE and only improved best validation accuracy to about `0.375`.

Useful 4080S ablation commands:

```powershell
python train.py --config config\config.yaml --stage v1
python train.py --config config\config.yaml --stage v1 --lambda-rsd 0.0
python train.py --config config\config.yaml --stage v1 --lambda-cafd 0.1 --lambda-logits 0.1 --lambda-rsd 0.001 --kd-warmup-epochs 5 --rsd-warmup-epochs 5
python train.py --config config\config.yaml --stage v1 --lambda-cafd 0.1 --lambda-logits 0.0 --lambda-rsd 0.001
python train.py --config config\config.yaml --stage v1 --lambda-cafd 0.0 --lambda-logits 0.1 --lambda-rsd 0.001 --kd-warmup-epochs 5
```

Run the default RSD-enabled V1 command first. The earlier projected-target run
`20260512_095804` still improved slowly and only approached the low `0.3` range,
because CAFD/RSD were aligned to `video_projected`, a retained but untrained
video projection head. The current default instead aligns the WiFi student to
the stable S3D teacher feature. If this remains far below the single-person HAR
target, move next to a WiFi heatmap teacher instead of spending more time on
CAFD-only component ablations.

For the supervised projected-video teacher route, use a checkpoint trained in
`train_video_teacher.py --mode projector`. When `--projector-target projected`
is selected, V1 now requires the checkpoint to contain trained
`video_projector.*` weights. With `projector.use_projector_logits: true`, it
also uses the trained `projector_classifier.*` logits for soft-label KD so the
feature target and logits target come from the same projected teacher branch.

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
python train.py --config config\config.yaml --stage v1 --teacher-checkpoint ../backbone_models/video/video_projector_s3d.pt --projector-target projected --lambda-cafd 0.5 --lambda-logits 0.05 --lambda-rsd 0.0 --kd-warmup-epochs 5
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
`cafd_loss`, raw and weighted `logits_kd_loss`, effective logits KD lambda,
student accuracy, and frozen teacher accuracy.
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
