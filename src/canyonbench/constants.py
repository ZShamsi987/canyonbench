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
TRACE_FEATURES: Final[tuple[str, ...]] = ("water", "road", "field")
TRACE_NOMINAL_FEATURE_WIDTHS_M: Final[dict[str, float]] = {
    # Registered nominal ground widths for the geometric extinction ladder: a
    # two-lane major road, the Colorado River in the flight corridor, and a
    # parcel-scale cultivated field.
    "road": 12.0,
    "water": 120.0,
    "field": 800.0,
}
PRESENCE_VALUES: Final[tuple[str, ...]] = ("yes", "no", "uncertain")
CAPTION_VALUES: Final[tuple[str, ...]] = ("yes", "no", "hedged")
SCORED_PHASES: Final[tuple[str, ...]] = ("Launching", "Floating")
CORE_PHASE: Final[str] = "Floating"
GROUNDING_GRID_SIZE: Final[int] = 4
GRID_THRESHOLD: Final[float] = 0.01
