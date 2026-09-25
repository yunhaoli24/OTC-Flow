"""Convert an OTC-Flow Lightning checkpoint into a public HF checkpoint."""

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import HfArgumentParser

from otc_flow.model import OTCFlowConfig, OTCFlowModel
from otc_flow.vqvae import OTCVQVAEConfig


@dataclass
class Arguments:
    checkpoint: str
    vqvae_source: str
    output_dir: str


def _checkpoint_state_dict(checkpoint: str) -> dict[str, torch.Tensor]:
    path = Path(checkpoint)
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    prefixes = ("model.", "module.model.")
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if (prefix := next((candidate for candidate in prefixes if key.startswith(candidate)), None)) is not None
    }


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    vqvae_config = OTCVQVAEConfig.from_pretrained(args.vqvae_source).to_dict()
    model = OTCFlowModel(OTCFlowConfig(vqvae_source=args.vqvae_source))
    model.load_state_dict(_checkpoint_state_dict(args.checkpoint), strict=True)
    model.config.vqvae_source = ""
    model.config.vqvae_config = vqvae_config
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
