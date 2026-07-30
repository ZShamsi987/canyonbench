"""Strict public contracts for CanyonBench-Trace specification v4."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from canyonbench.schemas import BudgetConfig, ModelConfig

SPEC_VERSION = "4.0.0"

FeatureClass = Literal["water", "road", "field"]
GeographicGroup = Literal["flight_corridor", "regional_ood", "cross_biome"]
CaseType = Literal["positive", "negative", "extinction"]
SplitName = Literal["development", "validation", "test"]
ViewGeometry = Literal["nadir", "oblique"]
InterventionOperator = Literal["blur", "texture", "frequency", "inpaint"]
TraceSequence = Literal[
    "screening",
    "oracle_deletion",
    "distractor_deletion",
    "self_deletion",
    "self_sufficiency",
    "random_control",
    "texture_control",
    "false_premise",
    "cave",
    "baseline",
    "synthetic_positive_control",
]


class StrictModel(BaseModel):
    """Forbid misspelled or undocumented fields in every trace artifact."""

    model_config = ConfigDict(extra="forbid")


class CameraSpec(StrictModel):
    """Virtual-camera geometry for one generated view."""

    longitude: Annotated[float, Field(ge=-180, le=180)]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    altitude_agl_m: Annotated[float, Field(gt=0)]
    horizontal_fov_deg: Annotated[float, Field(gt=1, lt=179)] = 40.0
    width_px: Annotated[int, Field(ge=128, le=8192)] = 1024
    height_px: Annotated[int, Field(ge=128, le=8192)] = 1024
    pitch_deg: Annotated[float, Field(ge=0, le=45)] = 0.0
    yaw_deg: Annotated[float, Field(ge=-180, le=180)] = 0.0
    source_crs: str = Field(default="EPSG:26912", pattern=r"^EPSG:\d+$")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ground_width_m(self) -> float:
        return 2 * self.altitude_agl_m * math.tan(math.radians(self.horizontal_fov_deg / 2))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ground_height_m(self) -> float:
        return self.ground_width_m * self.height_px / self.width_px

    @computed_field  # type: ignore[prop-decorator]
    @property
    def gsd_m_per_px(self) -> float:
        return self.ground_width_m / self.width_px

    @model_validator(mode="before")
    @classmethod
    def accept_serialized_derived_geometry(cls, value: Any) -> Any:
        """Re-read a written camera.json without loosening the strict contract.

        The derived footprint and GSD are serialized for downstream consumers, so
        loading a frozen bundle would otherwise trip ``extra='forbid'``. Provided
        values are checked against the camera model and then dropped, which keeps
        a hand-edited or corrupted manifest from validating.
        """

        if not isinstance(value, dict):
            return value
        names = ("ground_width_m", "ground_height_m", "gsd_m_per_px")
        if not any(name in value for name in names):
            return value
        # Copy: validation must never mutate the caller's parsed manifest.
        remaining = {key: item for key, item in value.items() if key not in names}
        derived = {name: value[name] for name in names if name in value}
        try:
            altitude = float(remaining["altitude_agl_m"])
            width_px = int(remaining.get("width_px", 1024))
            height_px = int(remaining.get("height_px", 1024))
            fov = float(remaining.get("horizontal_fov_deg", 40.0))
        except (KeyError, TypeError, ValueError):
            return remaining
        ground_width = 2 * altitude * math.tan(math.radians(fov / 2))
        expected = {
            "ground_width_m": ground_width,
            "ground_height_m": ground_width * height_px / width_px,
            "gsd_m_per_px": ground_width / width_px,
        }
        mismatched = [
            name
            for name, provided in derived.items()
            if not math.isclose(float(provided), expected[name], rel_tol=1e-6, abs_tol=1e-6)
        ]
        if mismatched:
            raise ValueError(
                f"camera geometry does not match its derived fields: {sorted(mismatched)}"
            )
        return remaining


class SourceRecord(StrictModel):
    """One immutable public source used to create a site."""

    source_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    product: str = Field(min_length=1)
    version: str = Field(min_length=1)
    acquisition_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    access_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    native_resolution_m: Annotated[float, Field(gt=0)]
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str = Field(min_length=1)
    terms_url: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    redistribution: Literal["allowed", "metadata_only", "prohibited", "pending"]
    tile_ids: list[str] = Field(default_factory=list)
    feature_ids: list[str] = Field(default_factory=list)


class SourceManifest(StrictModel):
    """All frozen sources and gate inputs for a base site."""

    schema_version: str = SPEC_VERSION
    site_id: str = Field(pattern=r"^site_\d{4,}$")
    imagery: SourceRecord
    feature_sources: dict[FeatureClass, list[SourceRecord]]
    detector_sources: dict[FeatureClass, SourceRecord] = Field(default_factory=dict)
    terrain: SourceRecord | None = None
    flight_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def every_feature_has_sources(self) -> SourceManifest:
        if set(self.feature_sources) != {"water", "road", "field"}:
            raise ValueError("feature_sources must contain water, road, and field")
        if any(len(records) < 2 for records in self.feature_sources.values()):
            raise ValueError("every feature must have at least two independent source records")
        for feature, records in self.feature_sources.items():
            source_ids = {record.source_id for record in records}
            source_products = {(record.provider, record.product) for record in records}
            if len(source_ids) != len(records) or len(source_products) != len(records):
                raise ValueError(
                    f"{feature} feature sources must use distinct source IDs "
                    "and provider/product pairs"
                )
        if not set(self.detector_sources).issubset({"water", "road", "field"}):
            raise ValueError("detector_sources contains an unknown feature")
        return self


class SiteSpec(StrictModel):
    """One independent geographic unit and its local source paths."""

    site_id: str = Field(pattern=r"^site_\d{4,}$")
    group: GeographicGroup
    target_class: FeatureClass
    case_type: CaseType
    longitude: Annotated[float, Field(ge=-180, le=180)]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    imagery_path: Path
    primary_mask_paths: dict[FeatureClass, Path]
    secondary_mask_paths: dict[FeatureClass, Path]
    detector_score_paths: dict[FeatureClass, Path] = Field(default_factory=dict)
    dem_path: Path | None = None
    source_manifest_path: Path
    source_tile_ids: list[str] = Field(min_length=1)
    feature_ids: list[str] = Field(default_factory=list)
    imagery_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    label_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    native_resolution_m: Annotated[float, Field(gt=0)]
    split: SplitName | None = None

    @model_validator(mode="after")
    def complete_feature_layers(self) -> SiteSpec:
        expected = {"water", "road", "field"}
        if set(self.primary_mask_paths) != expected:
            raise ValueError("primary_mask_paths must contain water, road, and field")
        if set(self.secondary_mask_paths) != expected:
            raise ValueError("secondary_mask_paths must contain water, road, and field")
        if not set(self.detector_score_paths).issubset(expected):
            raise ValueError("detector_score_paths contains an unknown feature")
        return self


class AcquisitionRegion(StrictModel):
    """One preregistered discovery region inside a geographic group."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    group: GeographicGroup
    west: Annotated[float, Field(ge=-180, le=180)]
    south: Annotated[float, Field(ge=-90, le=90)]
    east: Annotated[float, Field(ge=-180, le=180)]
    north: Annotated[float, Field(ge=-90, le=90)]
    weight: Annotated[int, Field(ge=1)] = 1
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_bounds(self) -> AcquisitionRegion:
        if self.west >= self.east or self.south >= self.north:
            raise ValueError("region bounds must satisfy west < east and south < north")
        return self


class SourceAcquisitionConfig(StrictModel):
    """Frozen source-discovery and storage-safe acquisition policy."""

    schema_version: str = SPEC_VERSION
    seed: int = 2026
    working_resolution_m: Annotated[float, Field(gt=0, le=2.13)] = 2.0
    source_half_extent_m: Annotated[float, Field(ge=12000, le=15000)] = 12500.0
    negative_screen_half_extent_m: Annotated[float, Field(ge=8000, le=12500)] = 10100.0
    minimum_candidate_separation_m: Annotated[float, Field(ge=1000)] = 8000.0
    candidate_multiplier: Annotated[float, Field(ge=1.0, le=5)] = 2.0
    preferred_naip_years: list[Annotated[int, Field(ge=2010, le=2100)]] = Field(
        default=[2023, 2022, 2021, 2020]
    )
    discovery_landcover_year: Annotated[int, Field(ge=1985, le=2100)] = 2023
    detector_landcover_year: Annotated[int, Field(ge=2017, le=2023)] = 2022
    regions: list[AcquisitionRegion] = Field(min_length=3)

    @model_validator(mode="after")
    def all_groups_covered(self) -> SourceAcquisitionConfig:
        groups = {region.group for region in self.regions}
        if groups != {"flight_corridor", "regional_ood", "cross_biome"}:
            raise ValueError("acquisition regions must cover all three geographic groups")
        if len({region.id for region in self.regions}) != len(self.regions):
            raise ValueError("acquisition region ids must be unique")
        return self


class CandidateSeed(StrictModel):
    """A remotely pre-screened center awaiting full source materialization."""

    candidate_id: str = Field(pattern=r"^candidate_\d{4,}$")
    region_id: str = Field(pattern=r"^[a-z0-9_]+$")
    group: GeographicGroup
    target_class: FeatureClass
    case_type: Literal["positive", "negative"]
    longitude: Annotated[float, Field(ge=-180, le=180)]
    latitude: Annotated[float, Field(ge=-90, le=90)]
    discovery_source: str = Field(min_length=1)
    discovery_feature_ids: list[str] = Field(default_factory=list)


class GateConfig(StrictModel):
    """Pre-registered thresholds for gates G1 through G4."""

    maximum_date_gap_days: Annotated[int, Field(ge=0)] = 1095
    field_maximum_date_gap_days: Annotated[int, Field(ge=0)] = 550
    consensus_tolerance_m: Annotated[float, Field(ge=0)] = 12.0
    minimum_consensus_fraction: Annotated[float, Field(ge=0, le=1)] = 0.5
    negative_safety_buffer_m: Annotated[float, Field(ge=0)] = 30.0
    minimum_apparent_width_px: Annotated[float, Field(gt=0)] = 1.0
    road_minimum_apparent_width_px: Annotated[float, Field(gt=0)] = 2.0
    extinction_width_px: Annotated[float, Field(gt=0)] = 1.0
    minimum_component_px: Annotated[int, Field(ge=1)] = 4
    minimum_boundary_distance_px: Annotated[float, Field(ge=0)] = 2.0
    minimum_local_contrast: Annotated[float, Field(ge=0, le=1)] = 0.03
    maximum_occlusion_fraction: Annotated[float, Field(ge=0, le=1)] = 0.25
    exclusion_detector_min_score: Annotated[float, Field(ge=0, le=1)] = 0.25

    @model_validator(mode="after")
    def extinction_not_wider_than_resolvable(self) -> GateConfig:
        if self.extinction_width_px > self.minimum_apparent_width_px:
            raise ValueError("extinction_width_px cannot exceed minimum_apparent_width_px")
        return self


class SiteQuota(StrictModel):
    group: GeographicGroup
    per_class: Annotated[int, Field(gt=0)]


class DatasetConfig(StrictModel):
    """Frozen dataset composition and rendering lattice."""

    source_root: Path
    output_root: Path
    site_manifest: Path
    quality_calibration: Path | None = None
    seed: int = 2026
    width_px: Annotated[int, Field(ge=128, le=4096)] = 1024
    height_px: Annotated[int, Field(ge=128, le=4096)] = 1024
    horizontal_fov_deg: Annotated[float, Field(gt=1, lt=179)] = 40.0
    altitudes_agl_m: list[float] = Field(default=[3000.0, 8000.0, 16000.0, 24000.0])
    geometries: list[ViewGeometry] = Field(default=["nadir", "oblique"])
    oblique_pitch_deg: Annotated[float, Field(gt=0, le=45)] = 20.0
    # Section 4.2 lists DEM-based relief injection as an optional mitigation. It
    # stays off in the frozen v4 run: the gap is quantified and reported instead.
    inject_relief_displacement: bool = False
    minimum_site_separation_m: Annotated[float, Field(gt=0)] = 22000.0
    degraded_view_count: Annotated[int, Field(ge=0)] = 240
    quotas: list[SiteQuota] = Field(
        default=[
            SiteQuota(group="flight_corridor", per_class=20),
            SiteQuota(group="regional_ood", per_class=12),
            SiteQuota(group="cross_biome", per_class=8),
        ]
    )
    split_fractions: dict[SplitName, float] = Field(
        default={"development": 0.2, "validation": 0.2, "test": 0.6}
    )
    gates: GateConfig = Field(default_factory=GateConfig)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def site_count(self) -> int:
        return 3 * sum(quota.per_class for quota in self.quotas)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def clean_view_count(self) -> int:
        return self.site_count * len(self.altitudes_agl_m) * len(self.geometries)

    @model_validator(mode="after")
    def frozen_v4_composition(self) -> DatasetConfig:
        if self.site_count != 120:
            raise ValueError("v4 critical-path dataset must contain exactly 120 sites")
        if self.clean_view_count != 960:
            raise ValueError("v4 critical-path lattice must contain exactly 960 clean views")
        if sorted(self.altitudes_agl_m) != [3000.0, 8000.0, 16000.0, 24000.0]:
            raise ValueError("v4 altitude lattice must be 3, 8, 16, and 24 km AGL")
        if set(self.geometries) != {"nadir", "oblique"}:
            raise ValueError("v4 requires both nadir and oblique geometry")
        if abs(sum(self.split_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        if self.split_fractions != {"development": 0.2, "validation": 0.2, "test": 0.6}:
            raise ValueError("v4 split fractions must be 20/20/60")
        if self.degraded_view_count > self.clean_view_count:
            raise ValueError("degraded_view_count cannot exceed clean views")
        return self


class QualityParameters(StrictModel):
    """One calibrated degradation; parameters remain zero when not selected."""

    degradation: Literal["none", "blur", "haze", "exposure", "saturation", "contrast", "jpeg"] = (
        "none"
    )
    blur_sigma: Annotated[float, Field(ge=0)] = 0.0
    haze_strength: Annotated[float, Field(ge=0, le=1)] = 0.0
    exposure_ev: Annotated[float, Field(ge=-4, le=4)] = 0.0
    saturation_scale: Annotated[float, Field(ge=0, le=3)] = 1.0
    contrast_scale: Annotated[float, Field(ge=0, le=3)] = 1.0
    jpeg_quality: Annotated[int, Field(ge=1, le=100)] = 100
    calibration_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exactly_one_degradation(self) -> QualityParameters:
        changed = {
            "blur": self.blur_sigma > 0,
            "haze": self.haze_strength > 0,
            "exposure": self.exposure_ev != 0,
            "saturation": self.saturation_scale != 1,
            "contrast": self.contrast_scale != 1,
            "jpeg": self.jpeg_quality != 100,
        }
        if self.degradation == "none" and any(changed.values()):
            raise ValueError("quality parameters change pixels while degradation is none")
        active = [name for name, enabled in changed.items() if enabled]
        if self.degradation != "none" and active != [self.degradation]:
            raise ValueError("exactly the named degradation must have a non-default parameter")
        return self


class QualityCalibration(StrictModel):
    """Frozen real-flight quality distribution used to parameterize degradations."""

    schema_version: str = SPEC_VERSION
    frame_count: Annotated[int, Field(gt=0)]
    source_sha256: list[str] = Field(min_length=1)
    quantiles: dict[str, dict[str, float]]

    @model_validator(mode="after")
    def complete_registered_metrics(self) -> QualityCalibration:
        required = {
            "laplacian_variance",
            "dark_channel_mean",
            "luminance_mean",
            "clipped_dark_fraction",
            "clipped_bright_fraction",
            "saturation_mean",
            "contrast_std",
            "jpeg_blockiness",
            "horizon_frequency",
            "obstruction_fraction",
        }
        if set(self.quantiles) != required:
            missing = sorted(required - set(self.quantiles))
            extra = sorted(set(self.quantiles) - required)
            raise ValueError(
                f"quality calibration metrics differ; missing={missing}, extra={extra}"
            )
        expected_quantiles = {"0.05", "0.25", "0.5", "0.75", "0.95"}
        for metric, values in self.quantiles.items():
            if set(values) != expected_quantiles:
                raise ValueError(f"{metric} must contain 0.05/0.25/0.5/0.75/0.95")
        invalid = [
            value
            for value in self.source_sha256
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ]
        if invalid:
            raise ValueError("quality calibration contains an invalid source SHA-256")
        return self


class FeatureDerived(StrictModel):
    present: bool
    case_type: CaseType
    area_px: int = Field(ge=0)
    area_fraction: Annotated[float, Field(ge=0, le=1)]
    component_count: Annotated[int, Field(ge=0)]
    largest_component_px: Annotated[int, Field(ge=0)]
    minimum_width_px: Annotated[float, Field(ge=0)]
    median_width_px: Annotated[float, Field(ge=0)]
    # Closest approach to a frame border (diagnostic) and deepest reach away from
    # every border (the resolvability criterion).
    boundary_distance_px: Annotated[float, Field(ge=0)]
    interior_distance_px: Annotated[float, Field(ge=0)] = 0.0
    occlusion_fraction: Annotated[float, Field(ge=0, le=1)]
    local_contrast: Annotated[float, Field(ge=0, le=1)]
    detector_mean_score: Annotated[float | None, Field(default=None, ge=0, le=1)]
    aliasing_risk: Annotated[float, Field(ge=0, le=1)]
    resolvability_score: Annotated[float, Field(ge=0, le=1)]
    resolvable: bool
    extinction: bool
    grid_occupancy: dict[str, list[str]]
    grid_target_pixel_counts: dict[str, dict[str, int]]


class ReliefDisplacement(StrictModel):
    """Reported sim-to-real relief gap of one synthesized orthophoto view."""

    relief_m: Annotated[float, Field(ge=0)]
    height_agl_m: Annotated[float, Field(gt=0)]
    displacement_ratio: Annotated[float, Field(ge=0)]
    edge_displacement_px: Annotated[float, Field(ge=0)]
    corner_displacement_px: Annotated[float, Field(ge=0)]
    corner_displacement_m: Annotated[float, Field(ge=0)]
    dem_available: bool
    injected: bool = False


class ViewManifest(StrictModel):
    schema_version: str = SPEC_VERSION
    site_id: str = Field(pattern=r"^site_\d{4,}$")
    view_id: str = Field(pattern=r"^view_[a-z0-9_]+$")
    split: SplitName
    target_class: FeatureClass
    geometry: ViewGeometry
    camera: CameraSpec
    quality: QualityParameters
    features: dict[FeatureClass, FeatureDerived]
    relief_displacement: ReliefDisplacement | None = None
    rgb_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_sha256: dict[FeatureClass, str]
    depth_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GateResult(StrictModel):
    """Machine-readable outcome of automatic gates G1 through G4."""

    site_id: str = Field(pattern=r"^site_\d{4,}$")
    feature: FeatureClass
    case_type: CaseType
    g1_time_alignment: bool
    date_gap_days: int = Field(ge=0)
    maximum_date_gap_days: int = Field(ge=0)
    g2_consensus: bool
    primary_secondary_iou: Annotated[float, Field(ge=0, le=1)]
    consensus_within_tolerance: Annotated[float, Field(ge=0, le=1)]
    negative_buffer_clear: bool
    g3_resolvable_or_extinction: bool
    minimum_width_px: Annotated[float, Field(ge=0)]
    median_width_px: Annotated[float, Field(ge=0)]
    component_count: int = Field(ge=0)
    boundary_distance_px: Annotated[float, Field(ge=0)]
    interior_distance_px: Annotated[float, Field(ge=0)] = 0.0
    local_contrast: Annotated[float, Field(ge=0, le=1)]
    occlusion_fraction: Annotated[float, Field(ge=0, le=1)]
    extinction: bool
    g4_detector_pass: bool
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class InterventionConfig(StrictModel):
    operators: list[InterventionOperator] = Field(
        default=["blur", "texture", "frequency", "inpaint"]
    )
    fractions: list[float] = Field(default=[0.25, 0.5, 0.75, 1.0])
    feather_px: Annotated[int, Field(ge=0)] = 5
    random_candidates: Annotated[int, Field(ge=10)] = 256
    maximum_match_smd: Annotated[float, Field(gt=0)] = 0.25
    seed: int = 2026

    @model_validator(mode="after")
    def complete_operator_schedule(self) -> InterventionConfig:
        if set(self.operators) != {"blur", "texture", "frequency", "inpaint"}:
            raise ValueError("v4 requires O1, O2, O3, and secondary O4")
        if self.fractions != [0.25, 0.5, 0.75, 1.0]:
            raise ValueError("v4 deletion schedule must be 25, 50, 75, and 100 percent")
        return self


class RegionCovariates(StrictModel):
    area_px: int = Field(ge=0)
    texture_energy: Annotated[float, Field(ge=0)]
    edge_density: Annotated[float, Field(ge=0, le=1)]
    mean_brightness: Annotated[float, Field(ge=0, le=1)]
    centre_distance: Annotated[float, Field(ge=0)]
    boundary_complexity: Annotated[float, Field(ge=0)]
    mean_depth: float | None = None


class InterventionRecord(StrictModel):
    schema_version: str = SPEC_VERSION
    view_id: str
    sequence: TraceSequence
    operator: InterventionOperator
    fraction: Annotated[float, Field(ge=0, le=1)]
    target_class: FeatureClass
    image_path: Path
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    region_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_covariates: RegionCovariates
    control_covariates: RegionCovariates | None = None
    standardized_mean_differences: dict[str, float] = Field(default_factory=dict)
    artifact_score: Annotated[float | None, Field(default=None, ge=0, le=1)]
    accepted: bool = True


class TraceResponse(StrictModel):
    """Black-box structured response required by every model."""

    answer: Literal["yes", "no", "abstain"]
    confidence: Annotated[int, Field(ge=0, le=100)]
    evidence_cells: list[str] = Field(max_length=10)
    cell_ranking: list[str] = Field(max_length=10)

    @model_validator(mode="after")
    def evidence_is_unique_and_ranked(self) -> TraceResponse:
        valid = {f"{row},{column}" for row in range(8) for column in range(8)}
        for name, values in (
            ("evidence_cells", self.evidence_cells),
            ("cell_ranking", self.cell_ranking),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} contains duplicates")
            invalid = sorted(set(values) - valid)
            if invalid:
                raise ValueError(f"{name} contains invalid cells: {invalid}")
        if set(self.cell_ranking) != set(self.evidence_cells):
            raise ValueError("cell_ranking must rank exactly the selected evidence_cells")
        return self

    def validate_protocol(self, *, grid_size: int, cell_budget: int) -> None:
        valid = {f"{row},{column}" for row in range(grid_size) for column in range(grid_size)}
        invalid = (set(self.evidence_cells) | set(self.cell_ranking)) - valid
        if invalid:
            raise ValueError(f"response contains cells outside the {grid_size}x{grid_size} grid")
        if len(self.evidence_cells) > cell_budget:
            raise ValueError(f"response selects more than K={cell_budget} cells")


class TraceProtocolConfig(StrictModel):
    grid_sizes: list[int] = Field(default=[4, 6, 8])
    cell_budgets: list[int] = Field(default=[3, 6, 10])
    screening_views: int = 960
    causal_core_views: int = 160
    prompt_cave_views: int = 120
    robustness_views: int = 80
    parse_retries: Annotated[int, Field(ge=0, le=10)] = 3
    seed: int = 2026

    @model_validator(mode="after")
    def required_sensitivity_grid(self) -> TraceProtocolConfig:
        if self.grid_sizes != [4, 6, 8] or self.cell_budgets != [3, 6, 10]:
            raise ValueError("V4 requires grids 4/6/8 and cell budgets 3/6/10")
        return self


class TraceRunConfig(StrictModel):
    dataset_dir: Path
    output_dir: Path
    prompt_file: Path
    models: list[ModelConfig] = Field(min_length=1)
    budget: BudgetConfig
    protocol: TraceProtocolConfig = Field(default_factory=TraceProtocolConfig)
    interventions: InterventionConfig = Field(default_factory=InterventionConfig)
    tiers: list[Literal["A", "B", "C"]] = Field(default=["A", "B", "C"])
    analyses: list[Literal["main", "sensitivity", "inpainting"]] = Field(default=["main"])
    fixture_responses: Path | None = None
    enforce_model_roster: bool = True

    @model_validator(mode="after")
    def unique_models(self) -> TraceRunConfig:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model ids must be unique")
        unpriced = [
            model.id
            for model in self.models
            if model.metered
            and (
                model.input_per_million_usd is None
                or model.output_per_million_usd is None
                or (model.input_per_million_usd == 0 and model.output_per_million_usd == 0)
            )
        ]
        if unpriced:
            raise ValueError(f"metered trace models require explicit non-zero pricing: {unpriced}")
        if self.enforce_model_roster:
            hidden_retries = [model.id for model in self.models if model.adapter.max_retries != 0]
            if hidden_retries:
                raise ValueError(
                    "trace adapters must set max_retries=0 so every retry is "
                    f"accounted by the run-wide cap: {hidden_retries}"
                )
            wrong_image_size = [model.id for model in self.models if model.image_max_side != 768]
            if wrong_image_size:
                raise ValueError(
                    f"registered trace models must use image_max_side=768: {wrong_image_size}"
                )
            nondeterministic = [model.id for model in self.models if model.temperature != 0]
            if nondeterministic:
                raise ValueError(
                    "registered trace models must use temperature=0 for "
                    f"deterministic resumability and content caching: {nondeterministic}"
                )
            unsupported = [model.id for model in self.models if not model.supports_json_schema]
            if unsupported:
                raise ValueError(
                    f"registered trace models must support structured output: {unsupported}"
                )
            roles = [model.benchmark_role for model in self.models]
            missing_roles = [model.id for model in self.models if model.benchmark_role is None]
            if missing_roles:
                raise ValueError(f"registered trace models require benchmark_role: {missing_roles}")
            counts: dict[str, int] = {
                role: roles.count(role) for role in set(roles) if role is not None
            }
            valid_composition = (
                2 <= counts.get("proprietary", 0) <= 3
                and 2 <= counts.get("open_weight", 0) <= 3
                and 1 <= counts.get("remote_sensing", 0) <= 2
                and counts.get("detector", 0) == 1
                and counts.get("fixture", 0) == 0
            )
            if not valid_composition or not 6 <= len(self.models) <= 9:
                raise ValueError(
                    "v4 roster requires 6-9 models: 2-3 proprietary, 2-3 "
                    "open-weight, 1-2 remote-sensing, and exactly 1 detector; "
                    f"the registered target is 3/3/1/1 (8 total); found {counts}"
                )
            proprietary = [model for model in self.models if model.benchmark_role == "proprietary"]
            missing_providers = [model.id for model in proprietary if not model.provider]
            providers = [model.provider for model in proprietary]
            if missing_providers or len(set(providers)) != len(providers):
                raise ValueError(
                    "the three proprietary models require distinct, non-empty providers"
                )
            if not any(
                model.supports_pointing
                for model in self.models
                if model.benchmark_role == "open_weight"
            ):
                raise ValueError("at least one open-weight model must support pointing")
        return self


class AuditRecord(StrictModel):
    site: str = Field(pattern=r"^site_\d{4,}$")
    view: str = Field(pattern=r"^view_[a-z0-9_]+$")
    auditor: str = Field(min_length=1, max_length=32)
    overlay_aligned: bool
    feature_resolvable: bool
    obvious_edit_artifact: bool
    source_mismatch: bool
    notes: str = Field(default="", max_length=2000)


class PromptTemplate(StrictModel):
    id: str = Field(min_length=1)
    variant: Literal["neutral", "false_premise", "uncertainty_aware", "evidence_first", "no_image"]
    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class TraceRequest(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    tier: Literal["A", "B", "C"]
    sequence: TraceSequence
    model: str
    site_id: str
    view_id: str
    target_class: FeatureClass
    prompt_id: str
    image_path: Path
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    grid_size: Literal[4, 6, 8] = 6
    cell_budget: Literal[3, 6, 10] = 6
    intervention_fraction: Annotated[float, Field(ge=0, le=1)] = 0
    intervention_operator: InterventionOperator | None = None
    cave_stage: Literal["initial", "necessity", "sufficiency", "nuisance"] | None = None
    baseline_kind: (
        Literal[
            "no_image",
            "blank",
            "shuffled",
            "unrelated",
            "always_yes",
            "always_no",
            "base_rate",
            "geographic_prior",
        ]
        | None
    ) = None
    synthetic_apparent_width_px: Annotated[float | None, Field(default=None, gt=0)] = None


class TracePrediction(StrictModel):
    request: TraceRequest
    response: TraceResponse | None
    raw_response: str | None
    format_failure: bool
    attempts: Annotated[int, Field(ge=0, le=4)]
    cache_hit: bool = False
    error: str | None = None
    latency_s: Annotated[float, Field(ge=0)]
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)] = 0
    provider_request_id: str | None = None
    finish_reason: str | None = None


class CaveDecision(StrictModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer: Literal["yes", "no", "abstain"]
    accepted: bool
    necessity: Annotated[float, Field(ge=-1, le=1)]
    sufficiency: Annotated[float, Field(ge=-1, le=1)]
    nuisance: Annotated[float, Field(ge=0, le=1)]
    calls_used: Annotated[int, Field(ge=1, le=4)] = 4
    reason: str


class CaveAblationRecord(StrictModel):
    model: str
    site_id: str = Field(pattern=r"^site_\d{4,}$")
    view_id: str = Field(pattern=r"^view_[a-z0-9_]+$")
    split: SplitName
    prompt_id: str
    variant: Literal["full", "necessity_only", "sufficiency_only", "nuisance_only"]
    decision: CaveDecision


class CaveThresholds(StrictModel):
    necessity_min: Annotated[float, Field(ge=-1, le=1)]
    sufficiency_min: Annotated[float, Field(ge=-1, le=1)]
    nuisance_max: Annotated[float, Field(ge=0, le=1)]
    confidence_min: Annotated[int, Field(ge=0, le=100)]
    tuned_split: Literal["development"] = "development"


class CaveFrontierPoint(StrictModel):
    thresholds: CaveThresholds
    balanced_accuracy: Annotated[float, Field(ge=0, le=1)]
    coverage: Annotated[float, Field(ge=0, le=1)]
    abstention_rate: Annotated[float, Field(ge=0, le=1)]
    false_positive_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    false_negative_rate: Annotated[float | None, Field(ge=0, le=1)] = None
    mean_calls_per_initial: Annotated[float, Field(ge=1, le=4)]
    calls_per_answered_case: Annotated[float | None, Field(ge=1)] = None
    n_cases: Annotated[int, Field(ge=1)]
    n_answered: Annotated[int, Field(ge=0)]


class ReleaseFile(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: Annotated[int, Field(ge=0)]


class PublicReleaseManifest(StrictModel):
    schema_version: str = SPEC_VERSION
    release_type: Literal["public_development"] = "public_development"
    splits: list[Literal["development", "validation"]]
    site_count: Annotated[int, Field(ge=1)]
    view_count: Annotated[int, Field(ge=1)]
    reserved_test_coordinates_included: Literal[False] = False
    input_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[ReleaseFile] = Field(min_length=1)


class EscrowRecord(StrictModel):
    opaque_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    variant: Literal["clean", "degraded"]
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PrivateTestEscrowManifest(StrictModel):
    schema_version: str = SPEC_VERSION
    release_type: Literal["private_test_escrow"] = "private_test_escrow"
    record_count: Annotated[int, Field(ge=0)]
    coordinates_included: Literal[False] = False
    camera_seeds_included: Literal[False] = False
    degradation_seeds_included: Literal[False] = False
    intervention_masks_included: Literal[False] = False
    records: list[EscrowRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def count_matches_records(self) -> PrivateTestEscrowManifest:
        if self.record_count != len(self.records):
            raise ValueError("record_count does not match escrow records")
        return self


class ProjectConfig(StrictModel):
    schema_version: str = SPEC_VERSION
    dataset: DatasetConfig
    interventions: InterventionConfig = Field(default_factory=InterventionConfig)
    protocol: TraceProtocolConfig = Field(default_factory=TraceProtocolConfig)


class PreregistrationConfig(StrictModel):
    """Frozen hypotheses, endpoints, exclusions, and analysis choices."""

    schema_version: str = SPEC_VERSION
    claims: dict[str, str]
    hypotheses: dict[str, str]
    primary_endpoints: list[Literal["macro_fpr", "ocrs", "efs"]]
    secondary_endpoints: list[str]
    exclusion_rules: list[str]
    ablations: list[str]
    model_roster_rules: list[str]
    outcome_matrix: dict[str, dict[Literal["confirmed", "not_confirmed"], str]]
    kill_criteria: dict[str, str]
    descope_ladder: list[str]
    multiplicity_method: Literal["benjamini_hochberg"] = "benjamini_hochberg"
    alpha: Annotated[float, Field(gt=0, lt=1)] = 0.05
    frozen_at: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def frozen_v4_analysis_contract(self) -> PreregistrationConfig:
        expected_hypotheses = {f"H{index}" for index in range(1, 7)}
        if set(self.claims) != {f"Claim{index}" for index in range(1, 5)}:
            raise ValueError("v4 preregistration must contain exactly Claim1 through Claim4")
        if set(self.hypotheses) != expected_hypotheses:
            raise ValueError("v4 preregistration must contain exactly H1 through H6")
        if set(self.outcome_matrix) != expected_hypotheses:
            raise ValueError("v4 outcome matrix must cover exactly H1 through H6")
        if any(
            set(outcomes) != {"confirmed", "not_confirmed"}
            for outcomes in self.outcome_matrix.values()
        ):
            raise ValueError(
                "each v4 outcome must precommit confirmed/not_confirmed interpretations"
            )
        if self.primary_endpoints != ["macro_fpr", "ocrs", "efs"]:
            raise ValueError("v4 primary endpoints must be macro_fpr, ocrs, and efs")
        return self
