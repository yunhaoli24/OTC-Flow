"""Train OTC-Flow stage one from an image JSONL manifest."""

from dataclasses import dataclass

from transformers import HfArgumentParser

from otc_flow.train import VQVAETrainConfig, train_vqvae


@dataclass
class Arguments:
    image_manifest: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_workers: int = 4
    disc_weight: float = 0.5


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    train_vqvae(VQVAETrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
