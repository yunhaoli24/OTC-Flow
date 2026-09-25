"""Generate a TEE image from one CT condition tensor."""

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import HfArgumentParser

from otc_flow.data import _load_tensor
from otc_flow.model import OTCFlowModel


@dataclass
class Arguments:
    checkpoint: str
    ct: str
    conditions: str
    output: str


def main() -> None:
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OTCFlowModel.from_pretrained(args.checkpoint).to(device).eval()
    ct = _load_tensor(Path(args.ct)).float().to(device)
    conditions = _load_tensor(Path(args.conditions)).float().to(device)
    if ct.ndim == 3:
        ct = ct.unsqueeze(0)
    if conditions.ndim == 1:
        conditions = conditions.unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(ct, conditions)
    torch.save(generated.cpu(), args.output)


if __name__ == "__main__":
    main()
