"""Command-line interface for the Terato pipeline."""

from __future__ import annotations

import argparse

from .pipeline import predict_with_model, train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teratogenicity SAR Visualizer & Predictor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models and generate SAR outputs")
    train_parser.add_argument("--train", required=True, help="Path to training CSV")
    train_parser.add_argument("--outdir", required=True, help="Output directory")
    train_parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    train_parser.add_argument("--bits", type=int, default=2048, help="Morgan fingerprint bits")
    train_parser.add_argument("--no-fingerprints", action="store_true", help="Disable fingerprints")

    predict_parser = subparsers.add_parser("predict", help="Predict with a trained model bundle")
    predict_parser.add_argument("--model", required=True, help="Path to model_bundle.joblib")
    predict_parser.add_argument("--input", required=True, help="Path to prediction CSV")
    predict_parser.add_argument("--outdir", required=True, help="Output directory")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_and_evaluate(
            train_path=args.train,
            output_dir=args.outdir,
            fingerprint_radius=args.radius,
            fingerprint_bits=args.bits,
            use_fingerprints=not args.no_fingerprints,
        )
    elif args.command == "predict":
        predict_with_model(
            model_bundle_path=args.model,
            data_path=args.input,
            output_dir=args.outdir,
        )


if __name__ == "__main__":
    main()
