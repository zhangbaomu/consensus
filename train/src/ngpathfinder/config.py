"""Configuration dataclasses and YAML loader for NanoGraph PathFinder2."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ComponentConfig:
    """Generic pluggable component description."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    variant: Optional[str] = None


@dataclass
class DataConfig:
    dataset: str
    params: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    num_workers: int = 0
    val_batch_size: Optional[int] = None
    test_batch_size: Optional[int] = None
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    train_transforms: Dict[str, Any] = field(default_factory=dict)
    val_transforms: Dict[str, Any] = field(default_factory=dict)
    test_transforms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    gradient_clip_val: Optional[float] = None
    precision: str = "float32"
    log_interval: int = 100
    val_interval: int = 1000
    decoder_only_epochs: int = 0


@dataclass
class InferenceConfig:
    """Optional inference-time overrides."""

    decode_strategy: str = "greedy"
    beam_width: int = 10
    beam_prune_threshold: Optional[float] = None
    torchaudio: Dict[str, Any] = field(default_factory=dict)
    use_duration_prior: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=lambda: DataConfig(dataset=""))
    encoder: ComponentConfig = field(default_factory=lambda: ComponentConfig(name="identity"))
    fusion: ComponentConfig = field(default_factory=lambda: ComponentConfig(name="identity"))
    aggregator: ComponentConfig = field(default_factory=lambda: ComponentConfig(name="identity"))
    decoder: ComponentConfig = field(default_factory=lambda: ComponentConfig(name="identity"))
    losses: Dict[str, ComponentConfig] = field(default_factory=dict)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(name="adam"))
    scheduler: Optional[ComponentConfig] = None
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    init_checkpoint: Optional[str] = None


def _component_from_dict(value: Dict[str, Any]) -> ComponentConfig:
    if "name" not in value:
        raise ValueError("Component configuration requires a 'name' field")
    params = value.get("params", {})
    if not isinstance(params, dict):
        raise TypeError("Component 'params' must be a mapping")
    variant = value.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise TypeError("Component 'variant' must be a string when provided")
    return ComponentConfig(name=value["name"], params=params, variant=variant)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from YAML."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise TypeError("Experiment YAML must define a mapping at the top level")

    data_cfg = raw.get("data", {})
    train_transforms_raw = data_cfg.get("train_transforms")
    val_transforms_raw = data_cfg.get("val_transforms")
    test_transforms_raw = data_cfg.get("test_transforms")

    def _normalize_transform_mapping(value: Any, name: str) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise TypeError(f"data.{name} must be a mapping when provided")

    data = DataConfig(
        dataset=data_cfg.get("dataset", ""),
        params=data_cfg.get("params", {}) or {},
        batch_size=data_cfg.get("batch_size", 1),
        num_workers=data_cfg.get("num_workers", 0),
        val_batch_size=data_cfg.get("val_batch_size"),
        test_batch_size=data_cfg.get("test_batch_size"),
        pin_memory=data_cfg.get("pin_memory", False),
        persistent_workers=data_cfg.get("persistent_workers", False),
        prefetch_factor=data_cfg.get("prefetch_factor"),
        train_transforms=_normalize_transform_mapping(train_transforms_raw, "train_transforms"),
        val_transforms=_normalize_transform_mapping(val_transforms_raw, "val_transforms"),
        test_transforms=_normalize_transform_mapping(test_transforms_raw, "test_transforms"),
    )

    optimizer_cfg = raw.get("optimizer", {"name": "adam"})
    optimizer = OptimizerConfig(
        name=optimizer_cfg.get("name", "adam"),
        params=optimizer_cfg.get("params", {}) or {},
    )

    scheduler_cfg = raw.get("scheduler")
    scheduler = _component_from_dict(scheduler_cfg) if isinstance(scheduler_cfg, dict) else None

    component_keys = ("encoder", "fusion", "aggregator", "decoder")
    components: Dict[str, ComponentConfig] = {}
    for key in component_keys:
        value = raw.get(key, {"name": "identity"})
        if not isinstance(value, dict):
            raise TypeError(f"Component '{key}' must be configured with a mapping")
        components[key] = _component_from_dict(value)

    losses_cfg = raw.get("losses", {})
    losses: Dict[str, ComponentConfig] = {}
    if not isinstance(losses_cfg, dict):
        raise TypeError("'losses' must be a mapping of loss names to definitions")
    for loss_name, loss_value in losses_cfg.items():
        if not isinstance(loss_value, dict):
            raise TypeError(f"Loss '{loss_name}' must be specified as a mapping")
        losses[loss_name] = _component_from_dict(loss_value)

    trainer_cfg = raw.get("trainer", {})
    decoder_only_epochs_raw = trainer_cfg.get("decoder_only_epochs", 0)
    if decoder_only_epochs_raw is None:
        decoder_only_epochs_raw = 0
    try:
        decoder_only_epochs = int(decoder_only_epochs_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("trainer.decoder_only_epochs must be an integer") from exc
    if decoder_only_epochs < 0:
        raise ValueError("trainer.decoder_only_epochs must be >= 0")

    trainer = TrainerConfig(
        max_steps=trainer_cfg.get("max_steps"),
        max_epochs=trainer_cfg.get("max_epochs"),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val"),
        precision=trainer_cfg.get("precision", "float32"),
        log_interval=trainer_cfg.get("log_interval", 100),
        val_interval=trainer_cfg.get("val_interval", 1000),
        decoder_only_epochs=decoder_only_epochs,
    )

    inference_cfg_raw = raw.get("inference", {}) or {}
    if not isinstance(inference_cfg_raw, dict):
        raise TypeError("'inference' must be specified as a mapping when provided")
    torchaudio_cfg = inference_cfg_raw.get("torchaudio", {}) or {}
    if torchaudio_cfg is not None and not isinstance(torchaudio_cfg, dict):
        raise TypeError("inference.torchaudio must be a mapping when provided")

    inference = InferenceConfig(
        decode_strategy=inference_cfg_raw.get("decode_strategy", "greedy"),
        beam_width=int(inference_cfg_raw.get("beam_width", 10) or 10),
        beam_prune_threshold=inference_cfg_raw.get("beam_prune_threshold"),
        torchaudio=torchaudio_cfg if isinstance(torchaudio_cfg, dict) else {},
        use_duration_prior=bool(inference_cfg_raw.get("use_duration_prior", False)),
    )

    return ExperimentConfig(
        seed=raw.get("seed", 42),
        data=data,
        encoder=components["encoder"],
        fusion=components["fusion"],
        aggregator=components["aggregator"],
        decoder=components["decoder"],
        losses=losses,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer=trainer,
        output_dir=raw.get("output_dir", "outputs"),
        checkpoint_dir=raw.get("checkpoint_dir", raw.get("checkpoints_dir", "checkpoints")),
        inference=inference,
        init_checkpoint=raw.get("init_checkpoint"),
    )


__all__ = [
    "ComponentConfig",
    "DataConfig",
    "OptimizerConfig",
    "TrainerConfig",
    "InferenceConfig",
    "ExperimentConfig",
    "load_config",
]