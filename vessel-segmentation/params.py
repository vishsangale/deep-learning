from dataclasses import dataclass


@dataclass
class ModelParams:
    r"""Hyperparameters of Retina UNet model."""

    feature_map_factor: int = 2
    learning_rate = 0.001
