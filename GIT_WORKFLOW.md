# Git Workflow

This project is versioned inside `WiMANS_Baseline/.git`.
Do not initialize git in the parent `WiMANS_fuse` directory.

## Initialize

```powershell
cd D:\PYTHON\Project\WiMANS_fuse\WiMANS_Baseline
git init
git status
git add .
git commit -m "chore: initialize WiMANS baseline project"
git tag v0.1.0
```

## Commit Rules

- Commit source, configs, scripts, and documentation.
- Do not commit dataset files, videos, `.npy`, `.mat`, checkpoints, or pretrained weights.
- Run the relevant smoke test before each milestone commit.
- Update `CHANGELOG.md` with:
  - date
  - commit hash
  - changes
  - problems found
  - validation commands
  - validation result

Recommended message style:

```text
feat: add V0 WiFi-only baseline
feat: add online S3D CAFD training path
test: add WiMANS data integrity checks
docs: document 4080S transfer workflow
fix: handle X-Fi pickle module alias
```

## Tag Rules

Suggested milestone tags:

```text
v0.1.0  project skeleton, docs, config, git workflow
v0.2.0  data check and label builder validated
v0.3.0  V0 WiFi-only smoke test passed
v0.4.0  V1 online S3D + CAFD smoke test passed
v1.0.0  full V0/V1 training config stable on 4080S
```

Create a tag after the corresponding commit:

```powershell
git tag v0.3.0
git log --oneline --decorate -n 10
```

## Transfer To 4080S

Copy the entire `WiMANS_Baseline/` directory, including hidden `.git/`.

After copying, verify:

```powershell
cd path\to\WiMANS_Baseline
git status
git log --oneline --decorate -n 10
```

Then update `config/config.yaml` paths and training scale fields:

```yaml
data:
  root: path/to/dataset
  annotation: path/to/dataset/annotation.csv

model:
  xfi_weight_path: path/to/backbone_models/wifi_ResNet18.pt

train:
  batch_size: 4
  num_workers: 4
  epochs: 50

video:
  s3d_weights: kinetics400
```

Keep the 4080S changes in git:

```powershell
git add config/config.yaml CHANGELOG.md
git commit -m "config: adapt training settings for 4080S"
```
