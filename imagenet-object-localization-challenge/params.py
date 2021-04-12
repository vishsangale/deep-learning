import dataclasses

@dataclasses.dataclass(frozen=True)
class DatasetParams:
    """Dataset related params"""
    nr_channels: int = 3
    img_height: int = 299
    img_width: int = 299

    nr_classes: int = 1000
