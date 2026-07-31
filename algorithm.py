from __future__ import annotations

import numpy as np

from models import RawAnalysisResult


def analyze_height_map(
    height_map: np.ndarray,
    pixel_size_um: float,
) -> RawAnalysisResult:
    """
    Analyze one 2D height map.

    Future implementation should:
    1. Correct tilt and rotation when required.
    2. Label the Pivot and Xpander.
    3. Calculate the Pivot height difference.
    4. Find the two Pivot cross centres.
    5. Calculate Xpander curvature radii in X and Y.
    6. Calculate fit-quality scores.
    7. Detect faulty elements and explain why.
    8. Return a label map:
       0 = background, 1 = Pivot, 2 = Xpander.

    This setup-only implementation deliberately returns empty measurements and
    an all-background label map, so the complete I/O pipeline can already run.
    """
    del pixel_size_um  # Will be used by the actual algorithm.

    empty_label_map = np.zeros(height_map.shape, dtype=np.uint8)

    return RawAnalysisResult(
        status="not_implemented",
        label_map=empty_label_map,
    )
