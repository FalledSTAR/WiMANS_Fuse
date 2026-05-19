import csv
import logging
from datetime import datetime
from pathlib import Path

import yaml


def _slug_join(values, default: str) -> str:
    if not values:
        return default
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) != 1:
        return "mixed"
    return str(values[0]).strip().replace(" ", "_") or default


def structured_output_dir(project_root: Path, output_dir: str, cfg=None) -> Path:
    base_dir = project_root / output_dir
    if cfg is None:
        return base_dir

    base_parts = {part.lower() for part in Path(output_dir).parts}
    if "single" in base_parts or "multi" in base_parts:
        return base_dir

    data_cfg = cfg.get("data", {})
    label_mode = str(data_cfg.get("label_mode", ""))
    user_mode = str(data_cfg.get("user_mode", ""))
    is_multi = user_mode == "multi" or label_mode.startswith("multi")
    if not is_multi:
        return base_dir / "single"

    environment = _slug_join(data_cfg.get("environment"), "unknown_environment")
    return base_dir / "multi" / environment


def create_run_dir(project_root: Path, output_dir: str, experiment_name: str, stage: str, cfg=None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = structured_output_dir(project_root, output_dir, cfg=cfg) / experiment_name / stage
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "metrics").mkdir()
    (run_dir / "splits").mkdir()
    return run_dir


def setup_run_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("wimans_baseline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_yaml(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def append_csv_rows(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
