# WiMANS_Baseline

WiMANS_Baseline is an independent project for WiMANS video-to-WiFi HAR experiments.
It references the local dataset, papers, official WiMANS code, X-Fi code, and X-Fi
WiFi backbone weights from the parent directory without tracking those large external
resources in git.

## Goals

- V0: run a stable 5 GHz single-user WiFi-only HAR baseline with X-Fi WiFi ResNet-18 initialization.
- V1: add online Frozen S3D video teacher, Hybrid Projector, and CAFD feature relation distillation.
- Keep the local laptop workflow focused on smoke tests and small runs.
- Move the whole project, including `.git/`, to the 4080S machine for full training.

The global random seed is fixed to `39` for Python, NumPy, PyTorch, and CUDA.

## Directory Layout

```text
WiMANS_Baseline/
|-- config/
|   `-- config.yaml
|-- datasets/
|-- losses/
|-- models/
|-- scripts/
|-- utils/
|-- train.py
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
../backbone_models/wifi_ResNet18.pt
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

## Training

V0:

```powershell
python train.py --config config\config.yaml --stage v0
```

V1:

```powershell
python train.py --config config\config.yaml --stage v1
```

Laptop-sized V1 training check:

```powershell
python train.py --config config\config.yaml --stage v1 --sample-limit 2 --num-frames 16 --s3d-weights none
```

The command above is only for checking the training path on the 3050 laptop.
Use the default `s3d_weights: kinetics400` and larger data settings on the 4080S.

## 4080S Transfer Notes

Copy the entire `WiMANS_Baseline/` folder, including the hidden `.git/` directory.
Then copy or mount the external resources so the config paths resolve:

```text
dataset/
backbone_models/wifi_ResNet18.pt
```

On the 4080S machine, update these config fields first:

- `data.root`
- `data.annotation`
- `model.xfi_weight_path`
- `train.batch_size`
- `train.num_workers`
- `train.epochs`
- `video.s3d_weights`

Verify version history after transfer:

```powershell
git status
git log --oneline --decorate -n 10
```
