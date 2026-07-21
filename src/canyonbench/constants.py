"""Stable benchmark ontology and phase constants."""

from typing import Final

FEATURES: Final[tuple[str, ...]] = (
    "water",
    "road",
    "building",
    "forest",
    "snow",
    "field",
)
PRESENCE_VALUES: Final[tuple[str, ...]] = ("yes", "no", "uncertain")
CAPTION_VALUES: Final[tuple[str, ...]] = ("yes", "no", "hedged")
SCORED_PHASES: Final[tuple[str, ...]] = ("Launching", "Floating")
CORE_PHASE: Final[str] = "Floating"
GROUNDING_GRID_SIZE: Final[int] = 4
GRID_THRESHOLD: Final[float] = 0.01
