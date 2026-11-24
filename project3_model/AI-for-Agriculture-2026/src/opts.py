import argparse
import os

def parse_args():
    is_kaggle = (os.name != "nt") and os.path.exists("/kaggle/input")
    default_paths_yaml = (
        "src/configs/paths_kaggle.yaml"
        if is_kaggle
        else "src/configs/paths_local.yaml"
    )

    parser = argparse.ArgumentParser(description="Training Baseline Model")
    parser.add_argument(
        "--configs", 
        nargs='+', 
        default=[
            default_paths_yaml, 
            "src/configs/model.yaml", 
            "src/configs/train.yaml", 
            "src/configs/dataset.yaml"
        ], 
        help="Paths to config files"
    )
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="AI-for-Agriculture-2026", 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", 
        type=str, 
        default=None, 
        help="Wandb run name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=os.environ.get("WANDB_ENTITY", "thanhphat20502-studocu"), 
        help="Wandb entity (username or team name)"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=os.environ.get("WANDB_API_KEY", "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"),
        help="Wandb API Key for login"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="Wandb mode: 'online', 'offline' (save local plots without internet), or 'disabled'"
    )
    parser.add_argument(
        "--upload_model",
        action="store_true",
        help="Upload heavy .pth model checkpoints to WandB artifacts (recommended on Kaggle, optional on local)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override base data directory (automatically updates train/test paths)"
    )
    parser.add_argument(
        "--submission_data_dir",
        type=str,
        default=None,
        help="Optional original competition data directory containing unlabeled val/ for submission"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Force hardware device ('cpu', 'cuda', 'auto')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override base learning rate"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Override model name (e.g. resnet18, resnet18_rgb, resnet34, efficientnet_b0_rgb)"
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default=None,
        choices=["rgb", "multimodal"],
        help="Override input mode ('rgb' or 'multimodal')"
    )
    args = parser.parse_args()
    return args
