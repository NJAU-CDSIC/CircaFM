from dataclasses import dataclass


@dataclass
class TASKS:
    RECONSTRUCTION: str = "reconstruction"
    CLASSIFICATION: str = "classification"
    DIFFRHYTHM: str = "diffrhythm"
    EMBED: str = "embedding"
