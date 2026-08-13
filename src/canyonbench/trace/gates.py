"""Automatic G1-G4 inclusion gates; no gate can create a positive label."""

from __future__ import annotations

from datetime import date

import cv2
import numpy as np

from canyonbench.trace.derived import boundary_distance, interior_distance, local_contrast
from canyonbench.trace.schemas import FeatureClass, GateConfig, GateResult, SiteSpec


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    a, b = first > 0, second > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _date_gap(first: str, second: str) -> int:
    return abs((date.fromisoformat(first) - date.fromisoformat(second)).days)


def evaluate_gates(
    site: SiteSpec,
    feature: FeatureClass,
    primary: np.ndarray,
    secondary: np.ndarray,
    image: np.ndarray,
    *,
    config: GateConfig,
    native_resolution_m: float,
    detector_score: np.ndarray | None = None,
    occlusion_fraction: float = 0,
) -> GateResult:
    """Apply the four preregistered gates to a site at native resolution."""

    primary_binary = (primary > 0).astype(np.uint8)
    secondary_binary = (secondary > 0).astype(np.uint8)
    gap = _date_gap(site.imagery_date, site.label_date)
    maximum_gap = (
        config.field_maximum_date_gap_days if feature == "field" else config.maximum_date_gap_days
    )
    g1 = gap <= maximum_gap
    consensus = mask_iou(primary_binary, secondary_binary)
    tolerance_px = max(1, round(config.consensus_tolerance_m / native_resolution_m))
    safety_px = max(1, round(config.negative_safety_buffer_m / native_resolution_m))
    tolerance_kernel = np.ones((tolerance_px * 2 + 1, tolerance_px * 2 + 1), np.uint8)
    safety_kernel = np.ones((safety_px * 2 + 1, safety_px * 2 + 1), np.uint8)
    primary_covered = (
        float(np.mean((cv2.dilate(secondary_binary, tolerance_kernel) > 0)[primary_binary > 0]))
        if primary_binary.any()
        else 0.0
    )
    secondary_covered = (
        float(np.mean((cv2.dilate(primary_binary, tolerance_kernel) > 0)[secondary_binary > 0]))
        if secondary_binary.any()
        else 0.0
    )
    tolerance_consensus = min(primary_covered, secondary_covered)
    negative_clear = not np.logical_or(
        cv2.dilate(primary_binary, safety_kernel) > 0,
        cv2.dilate(secondary_binary, safety_kernel) > 0,
    ).any()
    if site.case_type == "negative":
        g2 = negative_clear
    elif feature == "field":
        # CDL is the crop-specific federal reference for this operational
        # ontology.  The specification permits one authoritative source plus
        # strong geometric criteria; G3 supplies those criteria below. Annual
        # NLCD remains a recorded secondary comparison, but its 30 m general
        # land-cover class is not promoted into a false veto of CDL.
        g2 = bool(primary_binary.any())
        tolerance_consensus = 1.0 if g2 else 0.0
    else:
        g2 = bool(
            primary_binary.any()
            and secondary_binary.any()
            and tolerance_consensus >= config.minimum_consensus_fraction
        )

    components, _, stats, _ = cv2.connectedComponentsWithStats(primary_binary)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if components > 1 else np.array([])
    valid_components = int(np.sum(component_areas >= config.minimum_component_px))
    distance = cv2.distanceTransform(primary_binary, cv2.DIST_L2, 5)
    widths = distance[primary_binary > 0] * 2
    minimum_width = float(np.min(widths)) if len(widths) else 0
    median_width = float(np.median(widths)) if len(widths) else 0
    edge_distance = boundary_distance(primary_binary)
    interior_reach = interior_distance(primary_binary)
    contrast = local_contrast(image, primary_binary)
    extinction = bool(
        primary_binary.any()
        and median_width <= config.extinction_width_px
        and site.case_type == "extinction"
    )
    minimum_resolvable_width = (
        config.road_minimum_apparent_width_px
        if feature == "road"
        else config.minimum_apparent_width_px
    )
    resolved = (
        valid_components > 0
        and median_width >= minimum_resolvable_width
        # Interior reach, not closest approach: a feature crossing the footprint
        # touches the border and is still fully resolvable.
        and interior_reach >= config.minimum_boundary_distance_px
        and contrast >= config.minimum_local_contrast
        and occlusion_fraction <= config.maximum_occlusion_fraction
    )
    g3 = bool(resolved or extinction or site.case_type == "negative")

    detector_pass = True
    if detector_score is not None:
        detector_score = detector_score.astype(float)
        if detector_score.max(initial=0) > 1:
            detector_score /= 255 if detector_score.max(initial=0) <= 255 else detector_score.max()
        target_values = detector_score[primary_binary > 0]
        if target_values.size:
            detector_pass = float(np.mean(target_values)) >= config.exclusion_detector_min_score
    reasons: list[str] = []
    if not g1:
        reasons.append("G1_DATE_GAP")
    if not g2:
        reasons.append("G2_SOURCE_DISAGREEMENT")
    if not g3:
        reasons.append("G3_UNRESOLVABLE")
    if not detector_pass:
        reasons.append("G4_DETECTOR_EXCLUSION")
    return GateResult(
        site_id=site.site_id,
        feature=feature,
        case_type=site.case_type,
        g1_time_alignment=g1,
        date_gap_days=gap,
        maximum_date_gap_days=maximum_gap,
        g2_consensus=g2,
        primary_secondary_iou=consensus,
        consensus_within_tolerance=tolerance_consensus,
        negative_buffer_clear=negative_clear,
        g3_resolvable_or_extinction=g3,
        minimum_width_px=minimum_width,
        median_width_px=median_width,
        component_count=valid_components,
        boundary_distance_px=edge_distance,
        interior_distance_px=interior_reach,
        local_contrast=contrast,
        occlusion_fraction=occlusion_fraction,
        extinction=extinction,
        g4_detector_pass=detector_pass,
        accepted=g1 and g2 and g3 and detector_pass,
        reasons=reasons,
    )
