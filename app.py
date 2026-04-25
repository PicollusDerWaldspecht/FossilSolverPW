"""Streamlit UI for the Pixel Worlds Fossil Solver."""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from paste_listener import global_paste_image
from image_processing import (
    EMPTY_TILE_INDEX,
    GRID,
    NUM_TILES,
    PuzzleImage,
    load_puzzle_image,
    load_reference,
    match_tiles_to_solution,
    state_from_assignment,
)
from solver import (
    EMPTY_VALUE,
    Move,
    apply_move,
    is_solvable,
    solve,
)

REFERENCE_PATH = Path(__file__).parent / "Puzzle Solution.png"
PREVIEW_TILE_PX = 96
STEPS_PER_ROW = 4


@st.cache_resource(show_spinner=False)
def _load_reference():
    puzzle, features = load_reference(REFERENCE_PATH)
    return puzzle, features


def _annotate_tile(tile: Image.Image, label: str, *, highlight: bool = False) -> Image.Image:
    img = tile.resize((PREVIEW_TILE_PX, PREVIEW_TILE_PX)).convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")
    if highlight:
        draw.rectangle(
            [(0, 0), (img.width - 1, img.height - 1)],
            outline=(255, 196, 0),
            width=4,
        )
    badge_size = 22
    draw.rectangle([(0, 0), (badge_size, badge_size)], fill=(0, 0, 0, 180))
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 3), label, fill=(255, 255, 255), font=font)
    return img


def _render_grid(
    tiles: list[Image.Image],
    labels: list[str],
    *,
    highlight: int | None = None,
) -> Image.Image:
    cell = PREVIEW_TILE_PX
    gap = 4
    total = cell * GRID + gap * (GRID + 1)
    canvas = Image.new("RGB", (total, total), (30, 30, 30))
    for idx in range(NUM_TILES):
        r, c = divmod(idx, GRID)
        x = gap + c * (cell + gap)
        y = gap + r * (cell + gap)
        annotated = _annotate_tile(
            tiles[idx], labels[idx], highlight=(highlight == idx)
        )
        canvas.paste(annotated, (x, y))
    return canvas


def _state_tiles(
    state: list[list[int]],
    solution_tiles: list[Image.Image],
) -> tuple[list[Image.Image], list[str]]:
    tiles: list[Image.Image] = []
    labels: list[str] = []
    for row in state:
        for value in row:
            tiles.append(solution_tiles[value])
            labels.append("·" if value == EMPTY_VALUE else str(value + 1))
    return tiles, labels


def _click_index(move: Move) -> int:
    return move.from_row * GRID + move.from_col


def _build_steps(initial: list[list[int]], moves: list[Move]):
    steps = []
    state = [row.copy() for row in initial]
    for i, move in enumerate(moves, start=1):
        steps.append((i, move, state))
        state = apply_move(state, move)
    steps.append((len(moves) + 1, None, state))
    return steps


def _get_uploaded_image() -> Image.Image | None:
    """Returns a PIL image either from the file uploader or a Ctrl+V paste.

    Whichever input fires most recently wins, so a new upload or paste
    transparently replaces the previous image. A Clear button removes the
    current image and resets the file uploader.
    """

    if "_uploader_nonce" not in st.session_state:
        st.session_state["_uploader_nonce"] = 0

    pasted_image, paste_ts = global_paste_image()
    if (
        pasted_image is not None
        and paste_ts is not None
        and paste_ts != st.session_state.get("_last_paste_ts")
    ):
        st.session_state["_last_paste_ts"] = paste_ts
        st.session_state["_image"] = pasted_image

    uploader_key = f"file_uploader_{st.session_state['_uploader_nonce']}"
    uploaded = st.file_uploader(
        "Screenshot (or press Ctrl+V to paste)",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key=uploader_key,
    )
    if uploaded is not None:
        upload_id = (uploader_key, uploaded.name, uploaded.size)
        if upload_id != st.session_state.get("_last_upload_id"):
            st.session_state["_last_upload_id"] = upload_id
            try:
                st.session_state["_image"] = Image.open(io.BytesIO(uploaded.getvalue()))
            except Exception as exc:
                st.error(f"Could not read image: {exc}")

    image = st.session_state.get("_image")

    if image is not None:
        if st.button("Clear image"):
            st.session_state.pop("_image", None)
            st.session_state.pop("_last_upload_id", None)
            st.session_state["_uploader_nonce"] += 1
            st.rerun()

    return image


def main() -> None:
    st.set_page_config(page_title="Fossil Solver", page_icon="🦴", layout="wide")
    st.title("Pixel Worlds Fossil Solver")

    if not REFERENCE_PATH.exists():
        st.error(f"Reference `{REFERENCE_PATH.name}` is missing in the project folder.")
        return

    reference, solution_features = _load_reference()
    solution_tiles = reference.tiles

    image = _get_uploaded_image()
    if image is None:
        return

    upload_puzzle: PuzzleImage = load_puzzle_image(image)
    assignment = match_tiles_to_solution(upload_puzzle.tiles, solution_features)

    detected_labels = [
        "·" if assignment[i] == EMPTY_TILE_INDEX else str(assignment[i] + 1)
        for i in range(NUM_TILES)
    ]
    detected_grid = _render_grid(
        [solution_tiles[a] for a in assignment],
        detected_labels,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.image(upload_puzzle.cropped, caption="Upload", width="stretch")
    with col_b:
        st.image(detected_grid, caption="Detected", width=320)

    with st.expander("Fix detection"):
        options = list(range(NUM_TILES))

        def _label(value: int) -> str:
            return "·" if value == EMPTY_TILE_INDEX else str(value + 1)

        manual = list(assignment)
        cols = st.columns(GRID)
        for idx in range(NUM_TILES):
            r, c = divmod(idx, GRID)
            with cols[c]:
                manual[idx] = st.selectbox(
                    f"({r + 1},{c + 1})",
                    options=options,
                    index=assignment[idx],
                    format_func=_label,
                    key=f"override_{idx}",
                )
        if len(set(manual)) != NUM_TILES:
            st.warning("Duplicate values – every tile may only appear once.")
        else:
            assignment = manual

    state = state_from_assignment(assignment)

    if state == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]:
        st.success("Already solved.")
        return

    if not is_solvable(state):
        st.error("Detected state is not solvable – please fix the detection above.")
        return

    if not st.button("Solve", type="primary"):
        return

    with st.spinner("Solving..."):
        try:
            moves = solve(state)
        except ValueError as exc:
            st.error(str(exc))
            return

    steps = _build_steps(state, moves)
    st.subheader(f"{len(moves)} moves")

    for row_start in range(0, len(steps), STEPS_PER_ROW):
        cols = st.columns(STEPS_PER_ROW)
        for offset, (step_index, move, state_before) in enumerate(
            steps[row_start : row_start + STEPS_PER_ROW]
        ):
            tiles, labels = _state_tiles(state_before, solution_tiles)
            highlight = _click_index(move) if move is not None else None
            grid_img = _render_grid(tiles, labels, highlight=highlight)
            caption = f"Step {step_index}" if move is not None else "Done"
            with cols[offset]:
                st.image(grid_img, caption=caption, width="stretch")


if __name__ == "__main__":
    main()
