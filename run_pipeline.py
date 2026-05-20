import argparse
import sys
from src.preprocess import preprocess_pipeline
from src.train import train_model
from src.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(
        description="ChromoDiff: Generative Zero-Shot Pathogenicity Prediction via Discrete Genomic Diffusion"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["preprocess", "train", "evaluate"],
        help="Pipeline phase to execute."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/checkpoints/genodiff_best.pth",
        help="Path to pretrained model checkpoint (required for evaluation)."
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate synthetic/dummy data during preprocessing (recommended for fast testing)."
    )

    args = parser.parse_args()

    if args.mode == "preprocess":
        preprocess_pipeline(args.config, dummy=args.dummy)
    elif args.mode == "train":
        train_model(args.config)
    elif args.mode == "evaluate":
        run_evaluation(args.config, args.weights)
    else:
        print(f"Error: Unknown mode {args.mode}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
