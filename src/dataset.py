"""Pair images by explicit identifiers; never depend on filesystem order."""

import re
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


def pair_paths(folder: Path) -> dict[str, dict[str, Path]]:
    pairs: dict[str, dict[str, Path]] = {}
    for side in ("left", "right"):
        for path in sorted((folder / side).glob("*.bmp")):
            match = re.fullmatch(rf"{side}_pair_(\d{{6}})\.bmp", path.name)
            if match is None:
                raise ValueError(f"Unexpected filename: {path}")
            pair = pairs.setdefault(match[1], {})
            pair[side] = path
    if not pairs:
        raise ValueError(f"No BMP image pairs in {folder}")
    return dict(sorted(pairs.items()))


def read_gray(path: Path) -> NDArray[np.uint8]:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return cast(NDArray[np.uint8], image)
