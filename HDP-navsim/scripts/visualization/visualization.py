#!/usr/bin/env python3
"""Randomly sample NAVSIM v1.1 navtest scenes and visualize HDP predictions.

For every sampled token this script:

1. loads the official NAVSIM ``navtest`` scene;
2. runs the HDP/DP-VLA agent once;
3. plots the predicted trajectory and human ground truth in BEV;
4. saves both a PNG and the two trajectory arrays in an NPZ file.

The checkpoint should normally be a full checkpoint containing the IL encoder
and the RL-finetuned decoder, for example one produced by:

    scripts/checkpoint/merge_il_encoder_rl_decoder.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from matplotlib.lines import Line2D

import navsim
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import (
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
)
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG
from navsim.visualization.plots import configure_ax, configure_bev_ax


logger = logging.getLogger("visualize_random_navtest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample official NAVSIM v1.1 navtest scenes, run "
            "HDP-navsim inference, and save BEV trajectory visualizations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Full HDP checkpoint (.ckpt or HuggingFace directory). Defaults "
            "to the DP_VLA_CKPT environment variable."
        ),
    )
    parser.add_argument(
        "--encoder-path",
        default=None,
        help=(
            "Florence-2 model ID or local directory. Defaults to "
            "DP_VLA_ENCODER_PATH."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "OpenScene root containing navsim_logs/test and "
            "sensor_blobs/test. Defaults to OPENSCENE_DATA_ROOT."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New output directory. Defaults to "
            "$NAVSIM_EXP_ROOT/visualizations/hdp_navtest/<timestamp>."
        ),
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="Number of random navtest scenes to visualize (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random sampling and inference seed (default: 0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG resolution in dots per inch (default: 180).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed scene instead of recording and skipping it.",
    )
    return parser.parse_args()


def _required_path(
    cli_value: Optional[Path],
    environment_name: str,
    description: str,
) -> Path:
    value = cli_value
    if value is None:
        environment_value = os.environ.get(environment_name)
        if environment_value:
            value = Path(environment_value)
    if value is None:
        raise RuntimeError(
            f"{description} is required. Pass it on the command line or set "
            f"{environment_name}."
        )
    resolved = value.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _default_output_dir() -> Path:
    experiment_root = Path(
        os.environ.get(
            "NAVSIM_EXP_ROOT",
            Path.cwd() / "navsim-exp",
        )
    ).expanduser()
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    return (
        experiment_root
        / "visualizations"
        / "hdp_navtest"
        / timestamp
    ).resolve()


def _prepare_output_dir(configured: Optional[Path]) -> Path:
    output_dir = (
        configured.expanduser().resolve()
        if configured is not None
        else _default_output_dir()
    )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(
            f"Output path exists and is not a directory: {output_dir}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new "
            "directory to avoid overwriting previous visualizations."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _compose_evaluation_config(checkpoint: Path):
    navsim_package_root = Path(navsim.__file__).resolve().parent
    config_dir = (
        navsim_package_root
        / "planning"
        / "script"
        / "config"
        / "pdm_scoring"
    )
    if not config_dir.is_dir():
        raise FileNotFoundError(
            "Cannot find NAVSIM v1.1 PDM scoring configs under the imported "
            f"navsim package: {config_dir}"
        )

    search_path = (
        "[pkg://hdp_navsim.config,"
        "pkg://navsim.planning.script.config,"
        "pkg://navsim.planning.script.config.common]"
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
        job_name="visualize_random_navtest",
    ):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                "train_test_split=navtest",
                "agent=dp_vla_agent_base",
                "worker=sequential",
                f"hydra.searchpath={search_path}",
            ],
        )

    cfg.agent.config.test_config.checkpoint_path = str(checkpoint)
    return cfg


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_tokens(
    available_tokens: List[str],
    num_scenes: int,
    seed: int,
) -> List[str]:
    if num_scenes <= 0:
        raise ValueError(f"--num-scenes must be positive, got {num_scenes}")
    unique_tokens = sorted(set(available_tokens))
    if not unique_tokens:
        raise RuntimeError("The navtest SceneLoader returned no tokens.")

    count = min(num_scenes, len(unique_tokens))
    if count < num_scenes:
        logger.warning(
            "Requested %d scenes but navtest only exposed %d; visualizing all.",
            num_scenes,
            len(unique_tokens),
        )
    return random.Random(seed).sample(unique_tokens, count)


def _legend_handle(config: Dict[str, Any], label: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=config["line_color"],
        alpha=config["line_color_alpha"],
        linewidth=config["line_width"],
        linestyle=config["line_style"],
        marker=config["marker"],
        markersize=config["marker_size"],
        label=label,
    )


def _render_prediction(
    scene,
    prediction,
    human_trajectory,
    token: str,
):
    frame_idx = scene.scene_metadata.num_history_frames - 1
    fig, ax = plt.subplots(
        1,
        1,
        figsize=BEV_PLOT_CONFIG["figure_size"],
    )
    add_configured_bev_on_ax(
        ax,
        scene.map_api,
        scene.frames[frame_idx],
    )
    add_trajectory_to_bev_ax(
        ax,
        human_trajectory,
        TRAJECTORY_CONFIG["human"],
    )
    add_trajectory_to_bev_ax(
        ax,
        prediction,
        TRAJECTORY_CONFIG["agent"],
    )
    configure_bev_ax(ax)
    configure_ax(ax)
    ax.set_title(f"NAVSIM navtest | token={token}", fontsize=10)
    ax.legend(
        handles=[
            _legend_handle(
                TRAJECTORY_CONFIG["agent"],
                "HDP prediction",
            ),
            _legend_handle(
                TRAJECTORY_CONFIG["human"],
                "Human GT",
            ),
        ],
        loc="upper right",
        framealpha=0.9,
    )
    fig.tight_layout()
    return fig


def _save_manifest(
    output_dir: Path,
    checkpoint: Path,
    data_root: Path,
    seed: int,
    requested_scenes: int,
    available_scenes: int,
    selected_tokens: List[str],
    results: List[Dict[str, Any]],
) -> None:
    payload = {
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "split": "navtest",
        "dataset_split": "test",
        "seed": seed,
        "requested_scenes": requested_scenes,
        "available_navtest_scenes": available_scenes,
        "selected_tokens": selected_tokens,
        "successful_scenes": sum(result["valid"] for result in results),
        "failed_scenes": sum(not result["valid"] for result in results),
        "results": results,
    }
    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    checkpoint = _required_path(
        args.checkpoint,
        "DP_VLA_CKPT",
        "HDP checkpoint",
    )
    data_root = _required_path(
        args.data_root,
        "OPENSCENE_DATA_ROOT",
        "OpenScene data root",
    )
    os.environ["OPENSCENE_DATA_ROOT"] = str(data_root)
    test_logs = data_root / "navsim_logs" / "test"
    test_sensors = data_root / "sensor_blobs" / "test"
    if not test_logs.exists():
        raise FileNotFoundError(f"Missing navtest logs: {test_logs}")
    if not test_sensors.exists():
        raise FileNotFoundError(f"Missing navtest sensor blobs: {test_sensors}")

    if args.encoder_path is not None:
        os.environ["DP_VLA_ENCODER_PATH"] = args.encoder_path
    if not os.environ.get("DP_VLA_ENCODER_PATH"):
        raise RuntimeError(
            "Florence-2 is required. Pass --encoder-path or set "
            "DP_VLA_ENCODER_PATH."
        )

    output_dir = _prepare_output_dir(args.output_dir)
    _seed_everything(args.seed)

    logger.info("Composing NAVSIM v1.1 navtest + HDP agent configuration")
    cfg = _compose_evaluation_config(checkpoint)
    agent = instantiate(cfg.agent)
    logger.info("Loading HDP checkpoint: %s", checkpoint)
    agent.initialize()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=test_sensors,
        data_path=test_logs,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    available_tokens = list(scene_loader.tokens)
    selected_tokens = _sample_tokens(
        available_tokens,
        args.num_scenes,
        args.seed,
    )

    logger.info(
        "Randomly selected %d of %d navtest scenes (seed=%d)",
        len(selected_tokens),
        len(available_tokens),
        args.seed,
    )
    logger.info("Output directory: %s", output_dir)

    results: List[Dict[str, Any]] = []
    for index, token in enumerate(selected_tokens, start=1):
        stem = f"{index:03d}_{token}"
        image_path = output_dir / f"{stem}.png"
        trajectory_path = output_dir / f"{stem}.npz"
        logger.info(
            "[%d/%d] Inference token=%s",
            index,
            len(selected_tokens),
            token,
        )

        try:
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()
            prediction = agent.compute_trajectory(agent_input)
            human_trajectory = scene.get_future_trajectory()

            fig = _render_prediction(
                scene,
                prediction,
                human_trajectory,
                token,
            )
            try:
                fig.savefig(
                    image_path,
                    dpi=args.dpi,
                    bbox_inches="tight",
                )
            finally:
                plt.close(fig)

            np.savez_compressed(
                trajectory_path,
                prediction=np.asarray(prediction.poses),
                human_gt=np.asarray(human_trajectory.poses),
                token=np.asarray(token),
            )
            results.append(
                {
                    "token": token,
                    "valid": True,
                    "image": image_path.name,
                    "trajectory": trajectory_path.name,
                }
            )
        except Exception as error:
            logger.exception("Visualization failed for token=%s", token)
            results.append(
                {
                    "token": token,
                    "valid": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if args.fail_fast:
                _save_manifest(
                    output_dir,
                    checkpoint,
                    data_root,
                    args.seed,
                    args.num_scenes,
                    len(available_tokens),
                    selected_tokens,
                    results,
                )
                raise

    _save_manifest(
        output_dir,
        checkpoint,
        data_root,
        args.seed,
        args.num_scenes,
        len(available_tokens),
        selected_tokens,
        results,
    )
    successful = sum(result["valid"] for result in results)
    logger.info(
        "Finished: %d successful, %d failed. Manifest: %s",
        successful,
        len(results) - successful,
        output_dir / "manifest.json",
    )
    if successful == 0:
        raise RuntimeError(
            "All selected scenes failed. Inspect manifest.json and the logs."
        )


if __name__ == "__main__":
    main()
