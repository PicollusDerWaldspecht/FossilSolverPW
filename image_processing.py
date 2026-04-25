"""Image processing pipeline for the Pixel Worlds Fossil Solver.

Responsibilities:
- Detect and crop the blue border around the puzzle screenshot.
- Split the inner area into 9 equally sized tiles (3x3).
- Build feature bundles (hashes + color/pixel stats) for the reference tiles.
- Match the tiles of an uploaded screenshot to the reference tiles via the
  Hungarian algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import imagehash
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

GRID = 3
NUM_TILES = GRID * GRID
EMPTY_TILE_INDEX = NUM_TILES - 1


@dataclass
class PuzzleImage:
    """A puzzle image cropped to the playfield and split into a 3x3 grid."""

    cropped: Image.Image
    tiles: list[Image.Image]

    @property
    def tile_size(self) -> tuple[int, int]:
        first = self.tiles[0]
        return first.width, first.height


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def detect_blue_border_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find the bounding box of the inner playfield inside the blue border.

    Returns ``(left, top, right, bottom)`` or ``None`` if no border was found.
    Strategy: locate the blue pixels via an HSV color mask, compute the inner
    rectangle they enclose, then shrink it slightly so leftover border pixels
    do not bleed into the crop.
    """

    rgb = np.array(_to_rgb(image))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([90, 80, 80], dtype=np.uint8)
    upper = np.array([130, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower, upper)

    if cv2.countNonZero(blue_mask) < 50:
        return None

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 30 or h < 30:
        return None

    inner_mask = np.zeros_like(blue_mask)
    cv2.rectangle(inner_mask, (x, y), (x + w, y + h), 255, thickness=-1)
    inner_mask = cv2.bitwise_and(inner_mask, cv2.bitwise_not(blue_mask))

    coords = cv2.findNonZero(inner_mask)
    if coords is None:
        return None
    ix, iy, iw, ih = cv2.boundingRect(coords)

    pad = max(2, int(round(min(iw, ih) * 0.01)))
    left = max(ix + pad, 0)
    top = max(iy + pad, 0)
    right = min(ix + iw - pad, image.width)
    bottom = min(iy + ih - pad, image.height)
    if right - left < 30 or bottom - top < 30:
        return None
    return left, top, right, bottom


def crop_to_grid(image: Image.Image) -> Image.Image:
    """Crop the image to the inner playfield (without the blue border).

    Falls back to the original image if no border could be detected.
    """

    image = _to_rgb(image)
    bbox = detect_blue_border_bbox(image)
    if bbox is None:
        return image
    return image.crop(bbox)


def split_into_tiles(image: Image.Image) -> list[Image.Image]:
    """Split the (already cropped) image into 9 equally sized tiles.

    Order: row-major, top-left to bottom-right.
    """

    image = _to_rgb(image)
    w, h = image.size
    tile_w = w // GRID
    tile_h = h // GRID
    tiles: list[Image.Image] = []
    for row in range(GRID):
        for col in range(GRID):
            left = col * tile_w
            top = row * tile_h
            right = left + tile_w if col < GRID - 1 else w
            bottom = top + tile_h if row < GRID - 1 else h
            tiles.append(image.crop((left, top, right, bottom)))
    return tiles


def load_puzzle_image(image: Image.Image) -> PuzzleImage:
    """Full preprocessing: crop + tile split."""

    cropped = crop_to_grid(image)
    tiles = split_into_tiles(cropped)
    return PuzzleImage(cropped=cropped, tiles=tiles)


FEATURE_TILE_SIZE = 128


@dataclass
class TileFeatures:
    """Bundle of hashes and pixel/color features describing a tile."""

    phash: imagehash.ImageHash
    dhash: imagehash.ImageHash
    ahash: imagehash.ImageHash
    pixels: np.ndarray  # FEATURE_TILE_SIZE x FEATURE_TILE_SIZE x 3, uint8
    hist: np.ndarray  # flattened 3D HSV histogram
    mean_lum: float
    std_lum: float


def _normalized_tile(tile: Image.Image) -> Image.Image:
    return tile.convert("RGB").resize((FEATURE_TILE_SIZE, FEATURE_TILE_SIZE))


def _hsv_histogram(rgb_pixels: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def compute_tile_features(tile: Image.Image) -> TileFeatures:
    norm = _normalized_tile(tile)
    pixels = np.array(norm, dtype=np.uint8)
    lum = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    return TileFeatures(
        phash=imagehash.phash(norm, hash_size=16),
        dhash=imagehash.dhash(norm, hash_size=16),
        ahash=imagehash.average_hash(norm, hash_size=16),
        pixels=pixels,
        hist=_hsv_histogram(pixels),
        mean_lum=float(lum.mean()),
        std_lum=float(lum.std()),
    )


def compute_tile_hashes(tiles: Iterable[Image.Image]) -> list[TileFeatures]:
    return [compute_tile_features(tile) for tile in tiles]


def load_reference(reference_path: str | Path) -> tuple[PuzzleImage, list[TileFeatures]]:
    """Load the solution image and return its tiles plus their feature bundles."""

    path = Path(reference_path)
    image = Image.open(path)
    puzzle = load_puzzle_image(image)
    features = compute_tile_hashes(puzzle.tiles)
    return puzzle, features


def _empty_score(features: TileFeatures) -> float:
    """Lower values strongly indicate the empty/dark tile.

    The empty slot is uniformly dark brown (low brightness, low standard
    deviation), so the sum of both values acts as a reliable score.
    """

    return features.mean_lum + features.std_lum


def _identify_empty_tile(features: list[TileFeatures]) -> int:
    """Return the index of the tile most likely to be the empty slot."""

    return int(min(range(len(features)), key=lambda i: _empty_score(features[i])))


def _pair_distance(a: TileFeatures, b: TileFeatures) -> float:
    """Combined distance score between two tile feature bundles.

    Combines multiple hashes, an HSV histogram and a pixel MSE so that no
    single (potentially noisy) metric can dominate the assignment.
    """

    phash_d = (a.phash - b.phash) / 256.0
    dhash_d = (a.dhash - b.dhash) / 256.0
    ahash_d = (a.ahash - b.ahash) / 256.0
    hist_d = float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA))
    diff = a.pixels.astype(np.float32) - b.pixels.astype(np.float32)
    mse = float(np.mean(diff * diff)) / (255.0 * 255.0)
    return phash_d + dhash_d + ahash_d + hist_d + mse


def match_tiles_to_solution(
    upload_tiles: list[Image.Image],
    solution_features: list[TileFeatures],
) -> list[int]:
    """Assign each uploaded tile a unique solution index.

    Strategy:
    1. Detect the empty tile separately (very dark, low variance) and pin it
       to solution index 8.
    2. For the remaining 8 tiles, build a combined distance score (multiple
       hashes + HSV histogram + pixel MSE) and resolve the assignment via the
       Hungarian algorithm.
    """

    if len(upload_tiles) != NUM_TILES:
        raise ValueError(f"Expected {NUM_TILES} tiles, got {len(upload_tiles)}")
    if len(solution_features) != NUM_TILES:
        raise ValueError(
            f"Expected {NUM_TILES} solution features, got {len(solution_features)}"
        )

    upload_features = compute_tile_hashes(upload_tiles)
    empty_upload_idx = _identify_empty_tile(upload_features)

    other_uploads = [i for i in range(NUM_TILES) if i != empty_upload_idx]
    other_solutions = list(range(NUM_TILES - 1))

    cost = np.zeros((len(other_uploads), len(other_solutions)), dtype=np.float64)
    for ui, u_idx in enumerate(other_uploads):
        for si, s_idx in enumerate(other_solutions):
            cost[ui, si] = _pair_distance(
                upload_features[u_idx], solution_features[s_idx]
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = [0] * NUM_TILES
    assignment[empty_upload_idx] = EMPTY_TILE_INDEX
    for ui, si in zip(row_ind, col_ind):
        assignment[other_uploads[int(ui)]] = other_solutions[int(si)]
    return assignment


def state_from_assignment(assignment: list[int]) -> list[list[int]]:
    """Convert the flat assignment list into a 3x3 matrix."""

    if len(assignment) != NUM_TILES:
        raise ValueError("Assignment must have 9 entries")
    return [assignment[r * GRID : (r + 1) * GRID] for r in range(GRID)]


def render_state_image(
    state: list[list[int]],
    solution_tiles: list[Image.Image],
    tile_size: tuple[int, int] | None = None,
) -> Image.Image:
    """Render a puzzle state using the solution tile images."""

    if tile_size is None:
        tile_w = min(t.width for t in solution_tiles)
        tile_h = min(t.height for t in solution_tiles)
    else:
        tile_w, tile_h = tile_size

    canvas = Image.new("RGB", (tile_w * GRID, tile_h * GRID), (20, 20, 20))
    for r, row in enumerate(state):
        for c, value in enumerate(row):
            tile = solution_tiles[value].resize((tile_w, tile_h))
            canvas.paste(tile, (c * tile_w, r * tile_h))
    return canvas
