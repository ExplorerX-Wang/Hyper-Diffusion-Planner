source /mnt/workspace/miniconda3/bin/activate navsim
source /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/env.sh

export DP_VLA_CKPT="/mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/navsim-exp/dp_vla_agent/2026.07.20.22.03.13/hf_checkpoints/epoch_179"
export DP_VLA_ENCODER_PATH="/mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/Florence-2-large"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python visualization.py \
  --num-scenes 20 \
  --seed 0 \
  --output-dir "/mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/navsim-exp/visualizations/il_179epoch_seed0"