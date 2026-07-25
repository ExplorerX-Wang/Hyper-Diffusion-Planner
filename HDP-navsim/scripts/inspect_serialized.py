#!/usr/bin/env python3
"""Export fields from pickle-based cache files to a human-readable CSV.

Supported inputs:

* regular pickle files (commonly ``.pkl``)
* gzip-compressed pickle files (commonly ``.gz``)
* XZ/LZMA-compressed pickle files (NAVSIM metric caches may still use
  the ``.pkl`` suffix)

The loaded object is recursively flattened. Each leaf is written as one CSV
row with its field path, Python type, optional dtype/shape, and value.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import enum
import gzip
import json
import lzma
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import numpy as np
except ImportError:  # pragma: no cover - only relevant outside the NAVSIM env
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - only relevant outside the NAVSIM env
    torch = None


GZIP_MAGIC = b"\x1f\x8b"
XZ_MAGIC = b"\xfd7zXZ\x00"
CSV_COLUMNS = ["field_path", "python_type", "dtype", "shape", "value"]


def qualified_type_name(value: Any) -> str:
    """Return a concise, qualified Python type name."""
    value_type = type(value)
    if value_type.__module__ == "builtins":
        return value_type.__qualname__
    return f"{value_type.__module__}.{value_type.__qualname__}"


def load_serialized(path: Path) -> Any:
    """Load a plain, gzip, or XZ/LZMA-compressed pickle."""
    with path.open("rb") as stream:
        magic = stream.read(max(len(GZIP_MAGIC), len(XZ_MAGIC)))

    if magic.startswith(GZIP_MAGIC) or path.suffix.lower() == ".gz":
        opener = gzip.open
        compression = "gzip"
    elif magic.startswith(XZ_MAGIC) or path.suffix.lower() in {".xz", ".lzma"}:
        opener = lzma.open
        compression = "xz/lzma"
    else:
        opener = open
        compression = "plain pickle"

    try:
        with opener(path, "rb") as stream:
            return pickle.load(stream)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Cannot import {exc.name!r} while unpickling {path}. "
            "Run this script inside the same NAVSIM/PyTorch environment that "
            "created the file."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {path} as {compression}: {exc}"
        ) from exc


def child_path(parent: str, key: Any, is_index: bool = False) -> str:
    """Build a readable path for a mapping key, attribute, or list index."""
    if is_index:
        return f"{parent}[{key}]" if parent else f"[{key}]"

    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}" if parent else key

    key_text = json.dumps(str(key), ensure_ascii=False)
    return f"{parent}[{key_text}]" if parent else f"[{key_text}]"


def safe_repr(value: Any) -> str:
    """Return a representation even for objects with a broken ``repr``."""
    try:
        return repr(value)
    except Exception as exc:  # pragma: no cover - unusual third-party objects
        return f"<repr failed: {exc}>"


def json_value(value: Any) -> str:
    """Serialize common scalar/container values without losing Unicode."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return str(value.value)
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=True, default=str)
    except (TypeError, ValueError):
        return safe_repr(value)


class ObjectFlattener:
    """Recursively convert an arbitrary object into CSV rows."""

    def __init__(self, max_items: int = 0, max_depth: int = 0) -> None:
        self.max_items = max_items
        self.max_depth = max_depth
        self.rows: List[Dict[str, str]] = []
        self.seen: Dict[int, str] = {}

    def add_row(
        self,
        path: str,
        value: Any,
        *,
        dtype: str = "",
        shape: str = "",
        rendered_value: Optional[str] = None,
    ) -> None:
        self.rows.append(
            {
                "field_path": path or "$",
                "python_type": qualified_type_name(value),
                "dtype": dtype,
                "shape": shape,
                "value": rendered_value if rendered_value is not None else json_value(value),
            }
        )

    def flatten(self, value: Any, path: str = "", depth: int = 0) -> None:
        """Recursively append rows for ``value``."""
        if self.max_depth > 0 and depth >= self.max_depth:
            self.add_row(path, value, rendered_value=f"<max depth reached> {safe_repr(value)}")
            return

        if torch is not None and isinstance(value, torch.Tensor):
            self._flatten_tensor(value, path)
            return

        if np is not None and isinstance(value, np.ndarray):
            self._flatten_ndarray(value, path)
            return

        if np is not None and isinstance(value, np.generic):
            scalar = value.item()
            self.add_row(path, value, dtype=str(value.dtype), rendered_value=json_value(scalar))
            return

        if value is None or isinstance(value, (str, bytes, bool, int, float, complex, Path, enum.Enum)):
            self.add_row(path, value)
            return

        track_identity = (
            isinstance(value, (Mapping, Sequence))
            or dataclasses.is_dataclass(value)
            or hasattr(value, "__dict__")
            or hasattr(type(value), "__slots__")
        )
        if track_identity:
            object_id = id(value)
            if object_id in self.seen:
                self.add_row(
                    path,
                    value,
                    rendered_value=f"<reference to {self.seen[object_id]}>",
                )
                return
            self.seen[object_id] = path or "$"

        if isinstance(value, Mapping):
            self._flatten_mapping(value, path, depth)
            return

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
            self._flatten_mapping(fields, path, depth)
            return

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            self._flatten_sequence(value, path, depth)
            return

        if hasattr(value, "_asdict"):
            try:
                self._flatten_mapping(value._asdict(), path, depth)
                return
            except Exception:
                pass

        attributes: Dict[str, Any] = {}
        if hasattr(value, "__dict__"):
            try:
                attributes.update(vars(value))
            except TypeError:
                pass

        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot not in attributes and hasattr(value, slot):
                attributes[slot] = getattr(value, slot)

        if attributes:
            self._flatten_mapping(attributes, path, depth)
            return

        self.add_row(path, value, rendered_value=safe_repr(value))

    def _flatten_mapping(self, value: Mapping, path: str, depth: int) -> None:
        if not value:
            self.add_row(path, value, rendered_value="{}")
            return

        items = list(value.items())
        limit = self._limit(len(items))
        for key, child in items[:limit]:
            self.flatten(child, child_path(path, key), depth + 1)
        if limit < len(items):
            self.add_row(
                child_path(path, "..."),
                value,
                rendered_value=f"<truncated {len(items) - limit} mapping entries>",
            )

    def _flatten_sequence(self, value: Sequence, path: str, depth: int) -> None:
        if len(value) == 0:
            self.add_row(path, value, rendered_value="[]")
            return

        limit = self._limit(len(value))
        for index in range(limit):
            self.flatten(value[index], child_path(path, index, is_index=True), depth + 1)
        if limit < len(value):
            self.add_row(
                child_path(path, "...", is_index=True),
                value,
                rendered_value=f"<truncated {len(value) - limit} sequence items>",
            )

    def _flatten_tensor(self, value: Any, path: str) -> None:
        tensor = value.detach().cpu()
        shape = str(tuple(tensor.shape))
        dtype = str(tensor.dtype)
        numel = int(tensor.numel())

        if self.max_items > 0 and numel > self.max_items:
            flat = tensor.reshape(-1)[: self.max_items].tolist()
            rendered = json.dumps(flat, ensure_ascii=False, allow_nan=True)
            rendered += f" ... <truncated {numel - self.max_items} values>"
        else:
            rendered = json_value(tensor.tolist())

        self.add_row(path, value, dtype=dtype, shape=shape, rendered_value=rendered)

    def _flatten_ndarray(self, value: Any, path: str) -> None:
        shape = str(tuple(value.shape))
        dtype = str(value.dtype)
        size = int(value.size)

        if self.max_items > 0 and size > self.max_items:
            flat = value.reshape(-1)[: self.max_items].tolist()
            rendered = json.dumps(flat, ensure_ascii=False, allow_nan=True, default=str)
            rendered += f" ... <truncated {size - self.max_items} values>"
        else:
            rendered = json_value(value.tolist())

        self.add_row(path, value, dtype=dtype, shape=shape, rendered_value=rendered)

    def _limit(self, length: int) -> int:
        if self.max_items <= 0:
            return length
        return min(length, self.max_items)


def default_output_path(input_path: Path) -> Path:
    """Return ``<input-name>.csv`` while handling double suffixes like .pkl.gz."""
    name = input_path.name
    for suffix in (".pkl.gz", ".pickle.gz", ".gz", ".pkl", ".pickle", ".xz", ".lzma"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return input_path.with_name(f"{name}.csv")


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten a .gz/.pkl pickle cache into a field/value CSV."
    )
    parser.add_argument("input", type=Path, help="Input .gz, .pkl, .pickle, .xz, or .lzma file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV path. Defaults to a CSV next to the input file.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help=(
            "Maximum values per tensor/array and entries per sequence/mapping. "
            "Use 0 (default) to export everything."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="Maximum recursion depth. Use 0 (default) for unlimited depth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path)
    )

    if not input_path.is_file():
        print(f"error: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if args.max_items < 0 or args.max_depth < 0:
        print("error: --max-items and --max-depth must be non-negative", file=sys.stderr)
        return 2
    if input_path == output_path:
        print("error: input and output paths must be different", file=sys.stderr)
        return 2

    try:
        data = load_serialized(input_path)
        flattener = ObjectFlattener(
            max_items=args.max_items,
            max_depth=args.max_depth,
        )
        flattener.flatten(data)
        write_csv(flattener.rows, output_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded: {input_path}")
    print(f"Root type: {qualified_type_name(data)}")
    print(f"Rows written: {len(flattener.rows)}")
    print(f"CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
