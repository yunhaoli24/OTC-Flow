"""Convert a VQ-VAE checkpoint to the public HF format."""

from dataclasses import dataclass

from transformers import HfArgumentParser

from otc_flow.vqvae import OTCVQVAE


@dataclass
class Arguments:
    source: str
    output_dir: str


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    model = OTCVQVAE.from_pretrained(args.source, device_map="cpu")
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
