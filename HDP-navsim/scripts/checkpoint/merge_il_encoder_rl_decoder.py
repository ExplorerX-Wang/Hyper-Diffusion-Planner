#!/usr/bin/env python3
"""Merge an IL encoder with an RL-finetuned decoder.

The HDP NAVSIM RL trainer is constructed with ``with_encoder=False`` and
therefore exports a decoder-only checkpoint.  This utility takes the encoder
from the full IL checkpoint, replaces its decoder with the RL decoder, and
writes a HuggingFace-style directory containing:

    config.json
    model.safetensors

Both HuggingFace checkpoint directories and Lightning ``.ckpt`` files are
accepted as inputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


TensorState = Dict[str, torch.Tensor]
MODEL_PREFIXES = ("module.", "agent.model.", "model.")
MODEL_KEY_PREFIXES = ("encoder.", "decoder.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge encoder.* from a full IL checkpoint with decoder.* from "
            "an RL checkpoint."
        )
    )
    parser.add_argument(
        "--il-checkpoint",
        "--il-ckpt",
        dest="il_checkpoint",
        type=Path,
        required=True,
        help="Full IL checkpoint directory or Lightning .ckpt file.",
    )
    parser.add_argument(
        "--rl-checkpoint",
        "--rl-ckpt",
        dest="rl_checkpoint",
        type=Path,
        required=True,
        help="RL checkpoint directory or Lightning .ckpt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "New HuggingFace-style output directory. It must not already "
            "exist."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional config.json. By default the IL directory config is "
            "used, falling back to the RL directory config."
        ),
    )
    return parser.parse_args()


def _find_state_dict(payload: object, source: Path) -> Mapping[str, object]:
    """Unwrap common Lightning / training checkpoint containers."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {source}")

    for container_key in ("state_dict", "model"):
        candidate = payload.get(container_key)
        if isinstance(candidate, Mapping):
            return candidate

    return payload


def _strip_model_prefixes(key: str) -> str:
    """Strip known wrapper prefixes without altering real model key names."""
    changed = True
    while changed:
        changed = False
        for prefix in MODEL_PREFIXES:
            if not key.startswith(prefix):
                continue
            candidate = key[len(prefix) :]
            if candidate.startswith(MODEL_KEY_PREFIXES) or any(
                candidate.startswith(next_prefix)
                for next_prefix in MODEL_PREFIXES
            ):
                key = candidate
                changed = True
                break
    return key


def _clean_model_state(
    state_dict: Mapping[str, object],
    source: Path,
) -> TensorState:
    cleaned: TensorState = {}
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str) or not isinstance(value, torch.Tensor):
            continue
        key = _strip_model_prefixes(raw_key)
        if key.startswith(MODEL_KEY_PREFIXES):
            cleaned[key] = value.detach().cpu()

    if not cleaned:
        raise RuntimeError(
            f"No encoder.* or decoder.* tensors were found in {source}"
        )
    return cleaned


def _torch_load_trusted(path: Path) -> object:
    """Load a trusted legacy PyTorch checkpoint across torch versions."""
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # torch versions predating the weights_only argument.
        return torch.load(path, map_location="cpu")


def _load_checkpoint(path: Path) -> TensorState:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")

    if path.is_dir():
        safetensors_path = path / "model.safetensors"
        pytorch_path = path / "pytorch_model.bin"

        if safetensors_path.is_file():
            state_dict = load_safetensors(
                str(safetensors_path),
                device="cpu",
            )
        elif pytorch_path.is_file():
            payload = _torch_load_trusted(pytorch_path)
            state_dict = _find_state_dict(payload, pytorch_path)
        else:
            raise FileNotFoundError(
                f"{path} contains neither model.safetensors nor "
                "pytorch_model.bin"
            )
    else:
        # Only load checkpoints from a trusted source. torch.load uses pickle
        # for legacy Lightning checkpoint files.
        payload = _torch_load_trusted(path)
        state_dict = _find_state_dict(payload, path)

    return _clean_model_state(state_dict, path)


def _select_config_path(
    explicit_config: Optional[Path],
    il_checkpoint: Path,
    rl_checkpoint: Path,
) -> Path:
    candidates = []
    if explicit_config is not None:
        candidates.append(explicit_config)
    if il_checkpoint.is_dir():
        candidates.append(il_checkpoint / "config.json")
    if rl_checkpoint.is_dir():
        candidates.append(rl_checkpoint / "config.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No config.json was found. Pass one explicitly with --config."
    )


def _validate_and_merge(
    il_state: TensorState,
    rl_state: TensorState,
) -> tuple[TensorState, int, int]:
    encoder_state = {
        key: value
        for key, value in il_state.items()
        if key.startswith("encoder.")
    }
    il_decoder_state = {
        key: value
        for key, value in il_state.items()
        if key.startswith("decoder.")
    }
    rl_decoder_state = {
        key: value
        for key, value in rl_state.items()
        if key.startswith("decoder.")
    }

    if not encoder_state:
        raise RuntimeError(
            "The IL checkpoint has no encoder.* tensors. Use the full IL "
            "checkpoint that was used to build the RL feature cache, not the "
            "Florence-2 directory or a decoder-only checkpoint."
        )
    if not il_decoder_state:
        raise RuntimeError(
            "The IL checkpoint has no decoder.* tensors, so decoder "
            "compatibility cannot be verified."
        )
    if not rl_decoder_state:
        raise RuntimeError("The RL checkpoint has no decoder.* tensors.")

    missing = sorted(set(il_decoder_state) - set(rl_decoder_state))
    unexpected = sorted(set(rl_decoder_state) - set(il_decoder_state))
    if missing:
        raise RuntimeError(
            f"The RL checkpoint is missing {len(missing)} decoder keys; "
            f"examples: {missing[:5]}"
        )
    if unexpected:
        raise RuntimeError(
            f"The RL checkpoint contains {len(unexpected)} unexpected "
            f"decoder keys; examples: {unexpected[:5]}"
        )

    for key, rl_value in rl_decoder_state.items():
        il_value = il_decoder_state[key]
        if il_value.shape != rl_value.shape:
            raise RuntimeError(
                f"Shape mismatch for {key}: IL={tuple(il_value.shape)}, "
                f"RL={tuple(rl_value.shape)}"
            )

    merged_state = {
        key: value.clone().contiguous()
        for key, value in encoder_state.items()
    }
    merged_state.update(
        {
            key: value.clone().contiguous()
            for key, value in rl_decoder_state.items()
        }
    )
    return merged_state, len(encoder_state), len(rl_decoder_state)


def _write_output(
    output: Path,
    merged_state: TensorState,
    config_path: Path,
) -> None:
    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}. Choose a new directory to "
            "avoid overwriting an existing checkpoint."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.tmp-",
            dir=str(output.parent),
        )
    )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        config["with_encoder"] = True

        with (temporary / "config.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")

        save_safetensors(
            merged_state,
            str(temporary / "model.safetensors"),
            metadata={"format": "pt"},
        )

        saved_state = load_safetensors(
            str(temporary / "model.safetensors"),
            device="cpu",
        )
        if set(saved_state) != set(merged_state):
            raise RuntimeError(
                "Saved checkpoint verification failed: state-dict keys "
                "changed during serialization."
            )

        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = _parse_args()
    il_checkpoint = args.il_checkpoint.expanduser().resolve()
    rl_checkpoint = args.rl_checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()

    il_state = _load_checkpoint(il_checkpoint)
    rl_state = _load_checkpoint(rl_checkpoint)
    merged_state, encoder_count, decoder_count = _validate_and_merge(
        il_state,
        rl_state,
    )
    config_path = _select_config_path(
        args.config.expanduser().resolve()
        if args.config is not None
        else None,
        il_checkpoint,
        rl_checkpoint,
    )
    _write_output(output, merged_state, config_path)

    print(f"Merged checkpoint: {output}")
    print(f"IL encoder tensors: {encoder_count}")
    print(f"RL decoder tensors: {decoder_count}")
    print(f"Total tensors: {len(merged_state)}")
    print("config.with_encoder: true")


if __name__ == "__main__":
    main()
