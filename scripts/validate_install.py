"""Check that the public OTC-Flow package imports and exposes its API."""

from otc_flow import OTCVQVAE, OTCFlowConfig, OTCFlowModel, OTCFlowOutput, OTCVQVAEConfig


def main() -> None:
    assert OTCFlowConfig.model_type == "otc_flow"
    assert OTCVQVAEConfig.model_type == "vqvae"
    assert OTCFlowModel is not None
    assert OTCFlowOutput is not None
    assert OTCVQVAE is not None
    print("OTC-Flow public package is importable.")


if __name__ == "__main__":
    main()
