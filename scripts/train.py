"""Training entry point for NanoGraph PathFinder2."""
from __future__ import annotations

import argparse
import logging
import os
import pprint
import random
import sys
import time
from contextlib import nullcontext
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ngpathfinder.config import ComponentConfig, ExperimentConfig, load_config
from ngpathfinder.data import DataModule, ReadDataset, list_block_segments
from ngpathfinder.losses import LOSS_REGISTRY
from ngpathfinder.losses.ctc_constants import NUM_BASES as CTC_NUM_BASES
from ngpathfinder.modules.aggregator import AGGREGATOR_REGISTRY
from ngpathfinder.modules.decoder import DECODER_REGISTRY
from ngpathfinder.modules.encoder import ENCODER_REGISTRY
from ngpathfinder.modules.fusion import FUSION_REGISTRY
from ngpathfinder.utils.logging import configure_logging


def _component_kwargs(component: ComponentConfig) -> Dict[str, Any]:
    params = dict(component.params)
    if component.variant:
        params.setdefault("variant", component.variant)
    return params


def _dataset_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    use_fasta = params.get("use_fasta_reference", True)
    if not use_fasta:
        raise ValueError(
            "ReadDataset now requires FASTA references; set data.params.use_fasta_reference to true"
        )
    dataset_type = str(params.get("type", "legacy")).lower()
    return {
        "dataset_type": dataset_type,
        "max_mv_len": params.get("max_mv_len"),
        "max_reads_per_segment": params.get("max_reads_per_segment"),
        "fasta_glob_patterns": params.get("fasta_glob_patterns"),
        "ambiguous_base_policy": params.get("ambiguous_base_policy", "error"),
        "fasta_sequence_policy": params.get("fasta_sequence_policy", "first"),
        "suppress_mv_len_warnings": params.get("suppress_mv_len_warnings", False),
        "use_fastq_base_sequence": params.get("use_fastq_base_sequence", True),
        "use_read_flag": params.get("use_read_flag", True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NanoGraph PathFinder2 models")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--init-checkpoint",
        help="Optional checkpoint to use for module initialization",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path_str: str | None) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _split_segments(
    segment_names: Sequence[str], split_config: Mapping[str, Any], seed: int
) -> Tuple[List[str], List[str], List[str]]:
    if not segment_names:
        raise ValueError("No segments available to split")

    train_ratio = float(split_config.get("train", 0.8))
    val_ratio = float(split_config.get("val", 0.1))
    test_ratio = float(split_config.get("test", max(0.0, 1.0 - train_ratio - val_ratio)))

    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")
    if total_ratio > 1.0 + 1e-6:
        raise ValueError("Split ratios may not sum to more than 1")

    names = list(segment_names)
    names.sort()
    rng = random.Random(seed)
    rng.shuffle(names)

    normalized = [train_ratio, val_ratio, test_ratio]
    ratio_sum = sum(normalized)
    normalized = [value / ratio_sum for value in normalized]

    counts = [int(len(names) * value) for value in normalized]
    while sum(counts) < len(names):
        for idx in range(len(counts)):
            if sum(counts) >= len(names):
                break
            counts[idx] += 1

    train_count, val_count, test_count = counts
    if train_count <= 0:
        raise ValueError("Train split must contain at least one segment")

    train = names[:train_count]
    val = names[train_count : train_count + val_count]
    test = names[train_count + val_count : train_count + val_count + test_count]
    return train, val, test


def _resolve_precision(precision: str, device: torch.device) -> Tuple[torch.dtype, bool]:
    normalized = precision.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32, False
    if normalized in {"float16", "fp16"}:
        if device.type != "cuda":
            logging.getLogger("train").warning(
                "Requested float16 precision but CUDA is unavailable; falling back to float32"
            )
            return torch.float32, False
        return torch.float16, True
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16, True
    raise ValueError(
        "trainer.precision must be one of {'float32', 'float16', 'bfloat16'}"
    )


def _build_datasets(cfg: ExperimentConfig) -> Tuple[Any, Any, Any]:
    dataset_name = cfg.data.dataset.lower()
    if dataset_name in {"", "read", "readdataset", "read_dataset"}:
        params = cfg.data.params
        dataset_params = _dataset_kwargs(params)
        dataset_type = dataset_params.get("dataset_type", "legacy")

        def _instantiate(path: Path, *, segment_names: Optional[List[str]] = None):
            return ReadDataset(path, segment_names=segment_names, **dataset_params)

        if dataset_type in {"legacy", "npy"}:
            if "train_dir" in params or "val_dir" in params or "test_dir" in params:
                train_path = _resolve_path(params.get("train_dir"))
                if train_path is None:
                    raise ValueError(
                        "ReadDataset requires 'train_dir' to be specified under data.params"
                    )
                val_path = _resolve_path(params.get("val_dir"))
                test_path = _resolve_path(params.get("test_dir"))
                train_dataset = _instantiate(train_path)
                val_dataset = _instantiate(val_path) if val_path else None
                test_dataset = _instantiate(test_path) if test_path else None
                return train_dataset, val_dataset, test_dataset

            segment_path = _resolve_path(params.get("segment_path"))
            if not segment_path:
                raise ValueError(
                    "ReadDataset requires either 'train_dir' or 'segment_path' in data.params"
                )
            dataset = _instantiate(segment_path)
            return dataset, None, None

        dataset_root = _resolve_path(params.get("dataset_dir"))
        if dataset_root is None:
            raise ValueError(
                "Monolithic datasets require 'dataset_dir' under data.params"
            )

        explicit_segment = params.get("segment")
        if explicit_segment:
            dataset = _instantiate(dataset_root, segment_names=[explicit_segment])
            return dataset, None, None

        split_config = params.get("split", {}) or {}
        split_seed = int(params.get("split_seed", cfg.seed))
        available_segments = list_block_segments(dataset_root)
        train_segments, val_segments, test_segments = _split_segments(
            available_segments, split_config, split_seed
        )

        train_dataset = _instantiate(dataset_root, segment_names=train_segments)
        val_dataset = _instantiate(dataset_root, segment_names=val_segments) if val_segments else None
        test_dataset = _instantiate(dataset_root, segment_names=test_segments) if test_segments else None
        return train_dataset, val_dataset, test_dataset

    raise ValueError(f"Unsupported dataset type: {cfg.data.dataset}")


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, dict):
            result[key] = _move_to_device(value, device)
        elif isinstance(value, list):
            result[key] = [
                item.to(device=device, non_blocking=True) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _has_non_finite_grad(modules: Iterable[torch.nn.Module]) -> bool:
    for module in modules:
        for param in module.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
    return False


def _sanitize_params_(modules: Iterable[torch.nn.Module]) -> None:
    with torch.no_grad():
        for module in modules:
            for param in module.parameters():
                if not torch.isfinite(param).all():
                    torch.nan_to_num_(param, nan=0.0, posinf=1e4, neginf=-1e4)


def _iter_named_params(modules: Dict[str, torch.nn.Module]):
    for module_name, module in modules.items():
        for param_name, param in module.named_parameters(recurse=True):
            yield f"{module_name}.{param_name}", param


def _report_bad_grads(modules: Dict[str, torch.nn.Module], step: int, logger: logging.Logger) -> None:
    bad_entries = []
    for full_name, param in _iter_named_params(modules):
        grad = param.grad
        if grad is None:
            continue
        if torch.isfinite(grad).all():
            continue
        grad_safe = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        if grad_safe.numel() == 0:
            g_min = 0.0
            g_max = 0.0
            g_norm = 0.0
        else:
            g_min = grad_safe.min().item()
            g_max = grad_safe.max().item()
            g_norm = torch.linalg.vector_norm(grad_safe).item()
        bad_entries.append(
            (
                full_name,
                tuple(grad.shape),
                bool(torch.isnan(grad).any().item()),
                bool(torch.isinf(grad).any().item()),
                g_min,
                g_max,
                g_norm,
            )
        )

    if not bad_entries:
        return

    logger.warning("=== step=%d non-finite gradients (count=%d) ===", step, len(bad_entries))
    for name, shape, has_nan, has_inf, gmin, gmax, gnorm in bad_entries[:32]:
        logger.warning(
            "bad grad | %s | shape=%s | nan=%s inf=%s | min=%.3e max=%.3e | ||g||=%.3e",
            name,
            shape,
            has_nan,
            has_inf,
            gmin,
            gmax,
            gnorm,
        )


def _format_name_list(names: Iterable[str], limit: int = 16) -> str:
    names = list(names)
    if not names:
        return "<none>"
    if len(names) <= limit:
        return ", ".join(names)
    shown = ", ".join(names[:limit])
    return f"{shown}, ... (total {len(names)})"


def _load_module_state(
    module: torch.nn.Module,
    module_state: Dict[str, Tensor],
    module_name: str,
    logger: logging.Logger,
) -> None:
    current_state = module.state_dict()
    module_state = dict(module_state)

    if "blank_head.weight" in current_state:
        blank_prefix = "blank_head."
        output_prefix = "output."
        has_blank_entries = any(key.startswith(blank_prefix) for key in module_state)
        has_output_entries = any(key.startswith(output_prefix) for key in module_state)
        if has_output_entries and not has_blank_entries:
            for key, value in list(module_state.items()):
                if key.startswith(output_prefix):
                    suffix = key[len(output_prefix) :]
                    module_state[f"{blank_prefix}{suffix}"] = value

    compatible_state: Dict[str, Tensor] = {}
    shape_mismatches = []
    unexpected_keys = []

    for key, value in module_state.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        if current_state[key].shape != value.shape:
            shape_mismatches.append(key)
            continue
        compatible_state[key] = value

    missing_after_load, unexpected_after_load = module.load_state_dict(compatible_state, strict=False)

    if compatible_state:
        logger.info(
            "Loaded %d parameters for module '%s'", len(compatible_state), module_name
        )
    else:
        logger.info("No matching parameters loaded for module '%s'", module_name)

    if unexpected_keys:
        logger.warning(
            "Module '%s': skipped unexpected parameters from checkpoint: %s",
            module_name,
            _format_name_list(unexpected_keys),
        )

    if shape_mismatches:
        logger.warning(
            "Module '%s': skipped parameters with shape mismatch: %s",
            module_name,
            _format_name_list(shape_mismatches),
        )

    filtered_missing = [key for key in missing_after_load if key not in shape_mismatches]
    if filtered_missing:
        logger.warning(
            "Module '%s': checkpoint did not provide parameters: %s",
            module_name,
            _format_name_list(filtered_missing),
        )

    if unexpected_after_load:
        logger.warning(
            "Module '%s': received unexpected keys during load: %s",
            module_name,
            _format_name_list(unexpected_after_load),
        )


def _initialize_from_checkpoint(
    modules: Dict[str, torch.nn.Module],
    checkpoint_path: Path,
    logger: logging.Logger,
) -> None:
    logger.info("Initializing modules from checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    for module_name, module in modules.items():
        module_state = checkpoint.get(module_name)
        if module_state is None:
            logger.warning(
                "Checkpoint does not contain state for module '%s'; keeping random initialization",
                module_name,
            )
            continue
        _load_module_state(module, module_state, module_name, logger)


def _clone_batch_for_replay(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().clone()
    if isinstance(data, dict):
        return {key: _clone_batch_for_replay(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_clone_batch_for_replay(item) for item in data]
    return data


def _infer_target_query_budget(batch: Dict[str, Any]) -> Optional[int]:
    move_tensor = batch.get("move")
    base_tensor = batch.get("base_index")
    if not (isinstance(move_tensor, torch.Tensor) and isinstance(base_tensor, torch.Tensor)):
        return None

    if move_tensor.shape != base_tensor.shape:
        return None

    target_mask = (move_tensor > 0) & (base_tensor > 0) & (base_tensor <= CTC_NUM_BASES)
    if target_mask.numel() == 0:
        return None

    counts = target_mask.sum(dim=-1)
    if counts.dim() >= 2:
        counts = counts.view(counts.size(0), -1).max(dim=1).values

    if counts.numel() == 0:
        return None

    max_required = int(counts.max().item())
    if max_required <= 0:
        return None

    return max_required


def _set_decoder_only_trainability(
    module_map: Dict[str, torch.nn.Module],
    base_trainable: Dict[int, bool],
    *,
    decoder_only: bool,
    logger: logging.Logger,
) -> None:
    """Enable gradients only for the decoder when requested.

    Parameters that were initially frozen stay frozen when decoder-only training
    ends. The logger receives a brief status message whenever the mode changes.
    """

    desired_trainable = {"decoder"} if decoder_only else set(module_map.keys())
    for name, module in module_map.items():
        allow_grad = name in desired_trainable
        for param in module.parameters():
            base_flag = base_trainable.get(id(param), param.requires_grad)
            param.requires_grad_(allow_grad and base_flag)

    mode_description = "decoder-only" if decoder_only else "full"
    logger.info("Trainable parameters mode set to %s", mode_description)


def _create_optimizer(name: str, params: Dict[str, Any], modules: Iterable[torch.nn.Module]) -> torch.optim.Optimizer:
    name = name.lower()
    parameters = [p for module in modules for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters found for optimizer")

    if name == "adam":
        return torch.optim.Adam(parameters, **params)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **params)
    if name == "sgd":
        return torch.optim.SGD(parameters, **params)
    raise ValueError(f"Unsupported optimizer '{name}'")


def _create_scheduler(
    scheduler_cfg: Any,
    optimizer: torch.optim.Optimizer,
    *,
    total_training_steps: Optional[int],
    steps_per_epoch: Optional[int],
) -> Tuple[torch.optim.lr_scheduler._LRScheduler | None, str]:
    if scheduler_cfg is None:
        return None, "none"
    name = scheduler_cfg.name.lower()
    params = dict(scheduler_cfg.params)
    scheduler_name_suffix = ""
    if name == "cosineannealinglr":
        # Allow users to write either 'T_max' or 't_max'.
        if "T_max" not in params and "t_max" in params:
            params["T_max"] = params.pop("t_max")

        if "T_max" not in params:
            if total_training_steps is None:
                raise ValueError(
                    "CosineAnnealingLR requires 'T_max' but it was not provided and could not "
                    "be inferred from trainer settings."
                )
            params["T_max"] = max(1, int(total_training_steps))
            scheduler_name_suffix = f"auto_T_max={params['T_max']}"

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **params)
    elif name == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **params)
    else:
        raise ValueError(
            f"Unsupported scheduler '{name}'. Supported: CosineAnnealingLR, StepLR, ExponentialLR"
        )
    if scheduler_name_suffix:
        return scheduler, f"{name}({scheduler_name_suffix})"
    return scheduler, name


def _instantiate_losses(cfg: ExperimentConfig) -> Dict[str, torch.nn.Module]:
    if not cfg.losses:
        raise ValueError("At least one loss must be configured under 'losses'")

    losses: Dict[str, torch.nn.Module] = {}
    for name, loss_cfg in cfg.losses.items():
        kwargs = _component_kwargs(loss_cfg)
        losses[name] = LOSS_REGISTRY.create(loss_cfg.name, **kwargs)
    return losses


def _train_one_epoch(
    *,
    epoch: int,
    dataloader: torch.utils.data.DataLoader,
    modules: Tuple[torch.nn.Module, ...],
    losses: Dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    autocast_dtype: torch.dtype,
    use_autocast: bool,
    grad_clip: float | None,
    log_interval: int,
    max_steps: int | None,
    scaler: torch.amp.GradScaler | None,
    logger: logging.Logger,
    precision: torch.dtype,
    global_step: int,
    writer: SummaryWriter | None,
) -> int:
    encoder, fusion, aggregator, decoder = modules
    module_dict = {
        "encoder": encoder,
        "fusion": fusion,
        "aggregator": aggregator,
        "decoder": decoder,
    }
    detect_anomaly = os.environ.get("PF2_DETECT_ANOMALY", "0") == "1"
    replay_bad_batch = os.environ.get("PF2_REPLAY_BAD_BATCH", "1") != "0"

    def _forward_modules(curr_batch: Dict[str, Any]) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Tensor]:
        local_target_query_budget = _infer_target_query_budget(curr_batch)
        encoded = encoder(curr_batch)
        if local_target_query_budget is not None:
            base_queries = int(getattr(fusion, "num_queries", local_target_query_budget))
            encoded["target_query_count"] = max(base_queries, local_target_query_budget)
        fused = fusion(encoded)
        aggregated = aggregator(fused)
        if "decoder_padding_mask" not in aggregated:
            pad_mask = curr_batch.get("decoder_padding_mask")
            if pad_mask is not None:
                aggregated["decoder_padding_mask"] = pad_mask
        outputs = decoder(aggregated)

        loss_values: Dict[str, Tensor] = {}
        total_loss: Tensor | None = None
        for loss_name, criterion in losses.items():
            loss_val = criterion(outputs, curr_batch)
            loss_values[loss_name] = loss_val
            total_loss = loss_val if total_loss is None else total_loss + loss_val

        if total_loss is None:
            raise RuntimeError("No losses were computed; check loss configuration")

        return outputs, loss_values, total_loss

    def _handle_non_finite_gradients(current_batch: Dict[str, Any]) -> bool:
        if not _has_non_finite_grad(modules):
            return False

        _report_bad_grads(module_dict, global_step, logger)

        if replay_bad_batch:
            logger.warning(
                "Replaying current batch with autograd anomaly detection to locate source of non-finite gradients"
            )
            replay_batch = _clone_batch_for_replay(current_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autograd.detect_anomaly(check_nan=True):
                with (
                    torch.autocast(device_type=device.type, dtype=autocast_dtype)
                    if use_autocast
                    else nullcontext()
                ):
                    _, _, replay_loss = _forward_modules(replay_batch)
                replay_loss.backward()
            optimizer.zero_grad(set_to_none=True)

        logger.warning("non-finite gradients detected; skipping optimizer step and sanitizing params")
        optimizer.zero_grad(set_to_none=True)
        _sanitize_params_(modules)
        if writer is not None:
            writer.add_scalar("train/skip_nonfinite", 1, global_step + 1)
        return True

    for module in modules:
        module.train()
    for loss_module in losses.values():
        loss_module.train()

    data_time_acc = 0.0
    batch_time_acc = 0.0
    timing_samples = 0

    epoch_loss_sums: Dict[str, float] = {}
    epoch_total_sum = 0.0
    epoch_batches = 0

    log_stride = max(1, log_interval)

    progress = tqdm(
        dataloader,
        desc=f"Train Epoch {epoch}",
        leave=False,
    )

    data_fetch_start = time.perf_counter()

    for batch_idx, batch in enumerate(progress):
        if max_steps is not None and global_step >= max_steps:
            break

        data_time = time.perf_counter() - data_fetch_start
        step_start = time.perf_counter()

        batch = _move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        ):
            outputs, loss_values, total_loss = _forward_modules(batch)

        if not torch.isfinite(total_loss):
            raise RuntimeError("Encountered non-finite loss; aborting training")

        total_loss_scalar = float(total_loss.detach().to("cpu").item())
        step_loss_scalars: Dict[str, float] = {}
        for loss_name, loss_val in loss_values.items():
            loss_scalar = float(loss_val.detach().to("cpu").item())
            step_loss_scalars[loss_name] = loss_scalar
            epoch_loss_sums[loss_name] = epoch_loss_sums.get(loss_name, 0.0) + loss_scalar
        epoch_total_sum += total_loss_scalar
        epoch_batches += 1

        def finalize_iteration(total_loss_tensor: torch.Tensor) -> None:
            nonlocal data_fetch_start, data_time_acc, batch_time_acc, timing_samples
            batch_time = time.perf_counter() - step_start
            data_time_acc += data_time
            batch_time_acc += batch_time
            timing_samples += 1
            if (global_step % log_stride) == 0:
                total_loss_value = float(total_loss_tensor.detach().to("cpu").item())
                progress.set_postfix(
                    total_loss=f"{total_loss_value:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    data_time=f"{data_time:.3f}s",
                    batch_time=f"{batch_time:.3f}s",
                )
            else:
                progress.set_postfix(
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    data_time=f"{data_time:.3f}s",
                    batch_time=f"{batch_time:.3f}s",
                )
            data_fetch_start = time.perf_counter()

        if scaler is not None:
            scaled_loss = scaler.scale(total_loss)
            if detect_anomaly:
                with torch.autograd.detect_anomaly(check_nan=True):
                    scaled_loss.backward()
            else:
                scaled_loss.backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(chain.from_iterable(m.parameters() for m in modules), grad_clip)
            # Guard against non-finite gradients – skip optimizer step and sanitize params
            if _handle_non_finite_gradients(batch):
                global_step += 1
                finalize_iteration(total_loss)
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            if detect_anomaly:
                with torch.autograd.detect_anomaly(check_nan=True):
                    total_loss.backward()
            else:
                total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    chain.from_iterable(m.parameters() for m in modules), grad_clip
                )
            # Guard against non-finite gradients
            if _handle_non_finite_gradients(batch):
                global_step += 1
                finalize_iteration(total_loss)
                continue
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        if writer is not None and (global_step % log_stride) == 0:
            writer.add_scalar("train/total_loss", total_loss_scalar, global_step)
            for loss_name, loss_value in step_loss_scalars.items():
                writer.add_scalar(f"train/{loss_name}", loss_value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if (global_step % log_stride) == 0:
            suspicious: Dict[str, float] = {}
            for name, value in step_loss_scalars.items():
                if abs(value) < 1e-6:
                    suspicious[name] = value

            if suspicious:
                segment_ids = batch.get("segment_id", [])
                if isinstance(segment_ids, list):
                    seg_preview = segment_ids[:3]
                    segment_repr = f"{seg_preview}"
                    if len(segment_ids) > 3:
                        segment_repr = f"{segment_repr[:-1]}, ... (+{len(segment_ids) - 3})]"
                else:
                    segment_repr = str(segment_ids)

                move_tensor = batch.get("move")
                base_tensor = batch.get("base_index")
                target_counts = None
                if isinstance(move_tensor, torch.Tensor) and isinstance(base_tensor, torch.Tensor):
                    target_mask = (move_tensor > 0) & (base_tensor > 0) & (base_tensor <= CTC_NUM_BASES)
                    target_counts = target_mask.sum(dim=-1).detach().cpu().tolist()

                decoder_mask = outputs.get("decoder_padding_mask")
                decoder_lengths = None
                if isinstance(decoder_mask, torch.Tensor):
                    decoder_lengths = (~decoder_mask.to(torch.bool)).sum(dim=-1).detach().cpu().tolist()

                logger.warning(
                    "Near-zero loss | epoch=%d step=%d | segments=%s | losses=%s | target_tokens=%s | decoder_lengths=%s",
                    epoch,
                    global_step,
                    segment_repr,
                    suspicious,
                    target_counts,
                    decoder_lengths,
                )

        finalize_iteration(total_loss)

        if (global_step % log_stride) == 0:
            avg_data_time = data_time_acc / timing_samples if timing_samples else 0.0
            avg_batch_time = batch_time_acc / timing_samples if timing_samples else 0.0
            loss_str = ", ".join(f"{name}: {value:.6f}" for name, value in step_loss_scalars.items())
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch=%d step=%d lr=%.6g total_loss=%.6f (%s) data_time=%.3fs batch_time=%.3fs",
                epoch,
                global_step,
                lr,
                total_loss_scalar,
                loss_str,
                avg_data_time,
                avg_batch_time,
            )
            data_time_acc = 0.0
            batch_time_acc = 0.0
            timing_samples = 0

    if writer is not None and epoch_batches > 0:
        avg_total_loss = epoch_total_sum / epoch_batches
        writer.add_scalar("train_epoch/total_loss", avg_total_loss, epoch)
        for loss_name, loss_sum in epoch_loss_sums.items():
            writer.add_scalar(
                f"train_epoch/{loss_name}", loss_sum / epoch_batches, epoch
            )

    return global_step


@torch.no_grad()
def _evaluate(
    *,
    epoch: int,
    phase: str,
    dataloader: torch.utils.data.DataLoader[Any] | None,
    modules: Tuple[torch.nn.Module, ...],
    losses: Dict[str, torch.nn.Module],
    device: torch.device,
    precision: torch.dtype,
    use_autocast: bool,
    logger: logging.Logger,
    writer: SummaryWriter | None,
    global_step: int,
) -> Dict[str, float]:
    if dataloader is None:
        return {}

    encoder, fusion, aggregator, decoder = modules
    for module in modules:
        module.eval()
    for loss_module in losses.values():
        loss_module.eval()

    autocast_context = (
        torch.autocast(device_type=device.type, dtype=precision)
        if use_autocast
        else nullcontext()
    )

    loss_sums: Dict[str, float] = {}
    total_sum = 0.0
    batches = 0

    progress = tqdm(dataloader, desc=f"{phase.capitalize()} Epoch {epoch}", leave=False)

    with autocast_context:
        for batch in progress:
            batch = _move_to_device(batch, device)
            target_query_budget = _infer_target_query_budget(batch)

            encoded = encoder(batch)
            if target_query_budget is not None:
                base_queries = int(getattr(fusion, "num_queries", target_query_budget))
                encoded["target_query_count"] = max(base_queries, target_query_budget)
            fused = fusion(encoded)
            aggregated = aggregator(fused)
            if "decoder_padding_mask" not in aggregated:
                pad_mask = batch.get("decoder_padding_mask")
                if pad_mask is not None:
                    aggregated["decoder_padding_mask"] = pad_mask
            outputs = decoder(aggregated)

            loss_values: Dict[str, Tensor] = {}
            total_loss: Tensor | None = None
            for loss_name, criterion in losses.items():
                loss_val = criterion(outputs, batch)
                loss_values[loss_name] = loss_val
                total_loss = loss_val if total_loss is None else total_loss + loss_val

            assert total_loss is not None

            batches += 1
            detached_total = total_loss.detach().item()
            total_sum += detached_total
            for loss_name, loss_val in loss_values.items():
                loss_sums[loss_name] = loss_sums.get(loss_name, 0.0) + loss_val.detach().item()

            progress.set_postfix(
                total_loss=f"{detached_total:.4f}",
            )

    if batches == 0:
        return {}

    avg_losses = {name: value / batches for name, value in loss_sums.items()}
    avg_losses["total_loss"] = total_sum / batches

    loss_str = ", ".join(f"{name}: {value:.6f}" for name, value in avg_losses.items())
    logger.info("epoch=%d phase=%s %s", epoch, phase, loss_str)

    if writer is not None:
        for name, value in avg_losses.items():
            writer.add_scalar(f"{phase}/{name}", value, global_step)
            writer.add_scalar(f"{phase}_epoch/{name}", value, epoch)

    return avg_losses


def run_training(cfg: ExperimentConfig) -> None:
    logger = logging.getLogger("train")

    _set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, use_autocast = _resolve_precision(cfg.trainer.precision, device)

    encoder = ENCODER_REGISTRY.create(cfg.encoder.name, **_component_kwargs(cfg.encoder)).to(device)
    fusion = FUSION_REGISTRY.create(cfg.fusion.name, **_component_kwargs(cfg.fusion)).to(device)
    aggregator = AGGREGATOR_REGISTRY.create(
        cfg.aggregator.name, **_component_kwargs(cfg.aggregator)
    ).to(device)
    decoder = DECODER_REGISTRY.create(cfg.decoder.name, **_component_kwargs(cfg.decoder)).to(device)

    module_map = {
        "encoder": encoder,
        "fusion": fusion,
        "aggregator": aggregator,
        "decoder": decoder,
    }
    base_trainable = {
        id(param): param.requires_grad
        for module in module_map.values()
        for param in module.parameters()
    }

    decoder_only_epochs = max(0, int(getattr(cfg.trainer, "decoder_only_epochs", 0) or 0))
    if decoder_only_epochs:
        logger.info(
            "Decoder-only warmup enabled for first %d epoch(s)",
            decoder_only_epochs,
        )
    else:
        logger.info("Decoder-only warmup disabled")

    if cfg.init_checkpoint:
        checkpoint_path = Path(cfg.init_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {checkpoint_path}"
            )
        _initialize_from_checkpoint(module_map, checkpoint_path, logger)

    losses = _instantiate_losses(cfg)
    for module in losses.values():
        module.to(device)

    train_dataset, val_dataset, test_dataset = _build_datasets(cfg)
    datamodule = DataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_batch_size=cfg.data.val_batch_size,
        test_batch_size=cfg.data.test_batch_size,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        train_collate_config=cfg.data.train_transforms,
        val_collate_config=cfg.data.val_transforms,
        test_collate_config=cfg.data.test_transforms,
    )

    first_train_loader = datamodule.train_dataloader()
    try:
        steps_per_epoch = len(first_train_loader)
    except TypeError:
        steps_per_epoch = None

    if steps_per_epoch == 0:
        raise ValueError(
            "Training dataloader produced zero batches; check dataset and batch size"
        )

    max_epochs = cfg.trainer.max_epochs or 1
    max_steps = cfg.trainer.max_steps
    if max_steps is not None:
        total_planned_steps = max_steps
    elif steps_per_epoch is not None:
        total_planned_steps = steps_per_epoch * max_epochs
    else:
        total_planned_steps = None

    optimizer = _create_optimizer(
        cfg.optimizer.name, dict(cfg.optimizer.params), (encoder, fusion, aggregator, decoder)
    )
    scheduler, scheduler_name = _create_scheduler(
        cfg.scheduler,
        optimizer,
        total_training_steps=total_planned_steps,
        steps_per_epoch=steps_per_epoch,
    )

    use_grad_scaler = use_autocast and device.type == "cuda" and precision == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = output_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = output_dir / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))

    logger.info("Device: %s | Precision: %s | Scheduler: %s", device, precision, scheduler_name)
    logger.info("Encoder: %s", encoder)
    logger.info("Fusion: %s", fusion)
    logger.info("Aggregator: %s", aggregator)
    logger.info("Decoder: %s", decoder)
    logger.info("Checkpoints: %s", checkpoint_dir)

    log_interval = cfg.trainer.log_interval
    grad_clip = cfg.trainer.gradient_clip_val

    steps_completed = 0
    last_decoder_only: Optional[bool] = None
    for epoch in range(1, max_epochs + 1):
        decoder_only_active = decoder_only_epochs > 0 and epoch <= decoder_only_epochs
        if decoder_only_active != last_decoder_only:
            _set_decoder_only_trainability(
                module_map,
                base_trainable,
                decoder_only=decoder_only_active,
                logger=logger,
            )
            optimizer.zero_grad(set_to_none=True)
            last_decoder_only = decoder_only_active
        if epoch == 1:
            train_loader = first_train_loader
            first_train_loader = None
        else:
            train_loader = datamodule.train_dataloader()
        steps_completed = _train_one_epoch(
            epoch=epoch,
            dataloader=train_loader,
            modules=(encoder, fusion, aggregator, decoder),
            losses=losses,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            autocast_dtype=precision,
            use_autocast=use_autocast,
            grad_clip=grad_clip,
            log_interval=log_interval,
            max_steps=max_steps,
            scaler=scaler,
            logger=logger,
            precision=precision,
            global_step=steps_completed,
            writer=writer,
        )

        if max_steps is not None and steps_completed >= max_steps:
            logger.info("Reached max_steps=%d; stopping training", max_steps)
            break

        val_loader = datamodule.val_dataloader()
        val_metrics = _evaluate(
            epoch=epoch,
            phase="val",
            dataloader=val_loader,
            modules=(encoder, fusion, aggregator, decoder),
            losses=losses,
            device=device,
            precision=precision,
            use_autocast=use_autocast,
            logger=logger,
            writer=writer,
            global_step=steps_completed,
        )

        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": steps_completed,
                "encoder": encoder.state_dict(),
                "fusion": fusion.state_dict(),
                "aggregator": aggregator.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "val_metrics": val_metrics,
            },
            checkpoint_path,
        )
        logger.info("Saved checkpoint to %s", checkpoint_path)

    logger.info("Training completed. Total steps: %d", steps_completed)

    test_loader = datamodule.test_dataloader()
    if test_loader is not None:
        _evaluate(
            epoch=max_epochs,
            phase="test",
            dataloader=test_loader,
            modules=(encoder, fusion, aggregator, decoder),
            losses=losses,
            device=device,
            precision=precision,
            use_autocast=use_autocast,
            logger=logger,
            writer=writer,
            global_step=steps_completed,
        )

    writer.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.init_checkpoint:
        cfg.init_checkpoint = args.init_checkpoint

    configure_logging(cfg.output_dir)
    pprint.pprint(cfg)

    run_training(cfg)


if __name__ == "__main__":
    main()