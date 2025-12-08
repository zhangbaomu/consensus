"""Utility to visualise QueryFusion attention on a reference segment."""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt  # noqa: E402  # isort:skip
import numpy as np  # noqa: E402  # isort:skip
import torch  # noqa: E402  # isort:skip
from torch import Tensor  # noqa: E402  # isort:skip
from torch.nn import functional as F  # noqa: E402  # isort:skip

from ngpathfinder.config import load_config  # noqa: E402  # isort:skip
from ngpathfinder.data.datamodule import ReadDataset  # noqa: E402  # isort:skip
from ngpathfinder.modules.encoder import ENCODER_REGISTRY  # noqa: E402  # isort:skip
from ngpathfinder.modules.fusion import FUSION_REGISTRY  # noqa: E402  # isort:skip
from ngpathfinder.modules.fusion.query import QueryFusion  # noqa: E402  # isort:skip
from ngpathfinder.utils.checkpoint import load_checkpoint  # noqa: E402  # isort:skip


def _resample_sequence(sequence: Tensor, target_len: int) -> Tensor:
    """Linearly resample a sequence to ``target_len`` along the time axis."""

    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if sequence.size(0) == target_len:
        return sequence

    if sequence.dim() == 1:
        seq = sequence.unsqueeze(0).unsqueeze(0)
        mode = "linear"
    else:
        seq = sequence.transpose(0, 1).unsqueeze(0)
        mode = "linear"

    resampled = F.interpolate(seq, size=target_len, mode=mode, align_corners=True)
    if sequence.dim() == 1:
        return resampled.squeeze(0).squeeze(0)
    return resampled.squeeze(0).transpose(0, 1)


def _align_time_dimension(
    embeddings: List[Tensor],
    hints: List[Tensor],
    target_len: int | None,
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], int]:
    """Align embeddings and hints to a common time dimension."""

    lengths = [tensor.size(0) for tensor in embeddings]
    max_length = max(lengths)
    if target_len is None:
        target_len = max_length

    aligned_embeddings: List[Tensor] = []
    aligned_hints: List[Tensor] = []
    masks: List[Tensor] = []

    for emb, hint, length in zip(embeddings, hints, lengths):
        if target_len == length:
            aligned_emb = emb
            aligned_hint = hint
        elif target_len > length and target_len == max_length:
            pad = target_len - length
            aligned_emb = F.pad(emb, (0, 0, 0, pad))
            aligned_hint = F.pad(hint, (0, pad))
        else:
            aligned_emb = _resample_sequence(emb, target_len)
            aligned_hint = _resample_sequence(hint, target_len)

        aligned_embeddings.append(aligned_emb)
        aligned_hints.append(aligned_hint)

        mask = torch.zeros(target_len, device=emb.device, dtype=emb.dtype)
        if target_len == length:
            mask.fill_(1.0)
        elif target_len > length and target_len == max_length:
            mask[:length] = 1.0
        else:
            mask.fill_(1.0)
        masks.append(mask)

    return aligned_embeddings, aligned_hints, masks, target_len


def _dataset_kwargs(params: Dict[str, object]) -> Dict[str, object]:
    """Extract ``ReadDataset`` keyword arguments from config parameters."""

    use_fasta = params.get("use_fasta_reference", True)
    if not use_fasta:
        raise ValueError(
            "ReadDataset requires FASTA supervision; set data.params.use_fasta_reference to true"
        )

    return {
        "max_mv_len": params.get("max_mv_len"),
        "max_reads_per_segment": params.get("max_reads_per_segment"),
        "fasta_glob_patterns": params.get("fasta_glob_patterns"),
        "ambiguous_base_policy": params.get("ambiguous_base_policy", "error"),
        "fasta_sequence_policy": params.get("fasta_sequence_policy", "first"),
        "suppress_mv_len_warnings": params.get("suppress_mv_len_warnings", False),
        "use_fastq_base_sequence": params.get("use_fastq_base_sequence", True),
        "use_read_flag": params.get("use_read_flag", True),
    }


def _prepare_fusion_batch(
    dataset: ReadDataset,
    encoder: torch.nn.Module,
    num_reads: int,
    device: torch.device,
    target_time: int | None,
) -> Tuple[Dict[str, Tensor], List[int]]:
    """Encode and align ``num_reads`` reads for fusion."""

    if num_reads < 1:
        raise ValueError("num_reads must be at least 1")
    if len(dataset) == 0:
        raise ValueError("ReadDataset did not yield any segments")

    segment = dataset[0]
    signals = segment["signal"]
    moves = segment["move"]
    base_indices = segment["base_index"]

    if isinstance(signals, torch.Tensor):
        signals = [signals]
        moves = [moves]
        base_indices = [base_indices]

    available_reads = len(signals)
    if available_reads == 0:
        raise ValueError("Selected segment does not contain any reads")
    if num_reads > available_reads:
        raise ValueError(
            f"Requested {num_reads} reads but segment only provides {available_reads};"
            " reduce --num-reads or pick a different segment"
        )

    embeddings: List[Tensor] = []
    hints: List[Tensor] = []
    original_lengths: List[int] = []

    for idx in range(num_reads):
        signal = signals[idx].to(device)
        move = moves[idx].to(device)
        base_index = base_indices[idx].to(device)

        batch = {
            "signal": signal.unsqueeze(0),
            "move": move.unsqueeze(0),
            "base_index": base_index.unsqueeze(0),
        }

        encoded = encoder(batch)
        embedding = encoded["embedding"].squeeze(0)
        soft_hint = encoded["soft_hint"].squeeze(0)

        embeddings.append(embedding)
        hints.append(soft_hint)
        original_lengths.append(embedding.size(0))

    aligned_embeddings, aligned_hints, masks, _ = _align_time_dimension(
        embeddings, hints, target_time
    )

    embedding_tensor = torch.stack(aligned_embeddings, dim=0).unsqueeze(0)
    hint_tensor = torch.stack(aligned_hints, dim=0).unsqueeze(0)
    mask_tensor = torch.stack(masks, dim=0).unsqueeze(0)

    batch = {
        "embedding": embedding_tensor,
        "soft_hint": hint_tensor,
        "hard_mask": mask_tensor,
    }
    return batch, original_lengths


def _plot_attention(
    attention: Tensor,
    output_dir: Path,
    read_lengths: Iterable[int],
    query_count: int,
    time_steps: int,
) -> None:
    """Save per-head and averaged attention heatmaps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    attention_np = attention.cpu().numpy()
    read_lengths = list(read_lengths)

    batch, reads, heads, queries, steps = attention_np.shape
    assert batch == 1, "This visualiser expects batch size of 1"

    query_axis = np.linspace(0.0, 1.0, queries)
    time_axis = np.linspace(0.0, 1.0, steps)

    for read_idx in range(reads):
        head_grid = attention_np[0, read_idx]
        rows = math.ceil(heads / 4)
        cols = min(4, heads)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
        for head_idx in range(heads):
            row = head_idx // cols
            col = head_idx % cols
            ax = axes[row][col]
            img = ax.imshow(
                head_grid[head_idx],
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[time_axis[0], time_axis[-1], query_axis[0], query_axis[-1]],
            )
            ax.set_title(f"Read {read_idx + 1} · Head {head_idx + 1}")
            ax.set_xlabel("Normalised read time")
            ax.set_ylabel("Query index fraction")
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

        for extra in range(heads, rows * cols):
            row = extra // cols
            col = extra % cols
            axes[row][col].axis("off")

        fig.suptitle(
            f"Attention weights · Read {read_idx + 1} (original length {read_lengths[read_idx]})"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"attention_read{read_idx + 1}.png", dpi=150)
        plt.close(fig)

        mean_fig, mean_ax = plt.subplots(figsize=(6, 4))
        mean_img = mean_ax.imshow(
            head_grid.mean(axis=0),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[time_axis[0], time_axis[-1], query_axis[0], query_axis[-1]],
        )
        mean_ax.set_title(
            f"Mean attention · Read {read_idx + 1} (queries={query_count}, steps={time_steps})"
        )
        mean_ax.set_xlabel("Normalised read time")
        mean_ax.set_ylabel("Query index fraction")
        mean_fig.colorbar(mean_img, ax=mean_ax, fraction=0.046, pad=0.04)
        mean_fig.tight_layout()
        mean_fig.savefig(output_dir / f"attention_read{read_idx + 1}_mean.png", dpi=150)
        plt.close(mean_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--segment-dir",
        type=Path,
        default=Path(
            "signal_data/sup_5.2.0_top5k+2k_hp_windows200_float32_corrected_1/with_ref_poa/train/chr1_28496_28695"
        ),
        help="Path to the corrected segment directory (with .signals/.mv/.fastq files).",
    )
    parser.add_argument("--num-reads", type=int, default=3, help="Number of reads to fuse.")
    parser.add_argument(
        "--time-steps",
        type=int,
        default=400,
        help="Common time steps after normalising each read. Use <=0 to disable resampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./tmp/query_fusion_attention"),
        help="Directory to store attention visualisations.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda).")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Optional checkpoint containing encoder/fusion weights to load.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.time_steps <= 0:
        target_time = None
    else:
        target_time = args.time_steps

    device = torch.device(args.device)
    torch.manual_seed(0)

    cfg = load_config(args.config)
    dataset_kwargs = _dataset_kwargs(cfg.data.params)
    dataset = ReadDataset(args.segment_dir, **dataset_kwargs)

    encoder = ENCODER_REGISTRY.create(cfg.encoder.name, **cfg.encoder.params).to(device)
    fusion_module = FUSION_REGISTRY.create(cfg.fusion.name, **cfg.fusion.params).to(device)

    if not isinstance(fusion_module, QueryFusion):
        raise ValueError(
            "This visualiser requires the 'query' fusion module;"
            f" config specifies '{cfg.fusion.name}'"
        )
    fusion: QueryFusion = fusion_module

    if args.ckpt is not None:
        checkpoint_path = args.ckpt
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist")
        logging.info("Loading checkpoint from %s", checkpoint_path)
        checkpoint_state = load_checkpoint(str(checkpoint_path), map_location=device)
        for name, module in ("encoder", encoder), ("fusion", fusion):
            state = checkpoint_state.get(name)
            if state is None:
                logging.warning("Checkpoint missing '%s' parameters; leaving module randomly initialised", name)
                continue
            module.load_state_dict(state, strict=False)
            logging.info("Loaded parameters for %s", name)

    encoder.eval()
    fusion.eval()

    batch, lengths = _prepare_fusion_batch(dataset, encoder, args.num_reads, device, target_time)
    query_budget = getattr(fusion, "num_queries", None)
    if query_budget is None:
        raise AttributeError("QueryFusion instance does not expose 'num_queries'")
    batch["target_query_count"] = int(query_budget)

    if target_time is None:
        print(f"Loaded {args.num_reads} reads with original lengths: {lengths}")
    else:
        print(
            f"Loaded {args.num_reads} reads with original lengths: {lengths} → resampled to {target_time} steps"
        )

    with torch.no_grad():
        with fusion.capture_attention(True):
            output = fusion(batch)

    attention = output["fusion_attention_full"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.save(args.output_dir / "fusion_attention.npy", attention.cpu().numpy())
    _plot_attention(attention, args.output_dir, lengths, int(query_budget), attention.size(-1))

    print(f"Saved attention visualisations to {args.output_dir}")


if __name__ == "__main__":
    main()