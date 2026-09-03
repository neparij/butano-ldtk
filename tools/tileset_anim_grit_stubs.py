# SPDX-FileCopyrightText: Copyright 2025-2026 Guyeon Yu <copyrat90@gmail.com>, Nikolai Laptev (neparij)
# SPDX-License-Identifier: Zlib

"""Unique 8×8 noise stubs for BG-animation metatiles on the **main** tileset strip.

Grit can merge identical 8×8 graphics even when ``repeated_tiles_reduction`` is off in some
setups, or when the user relies on distinct meta-tile slots for animation logic. We only stub a
metatile whose **representative** ``src`` is listed for animation **and** there is another used
metatile with the **same pixel data** whose ``src`` is **not** in the animation list (the usual
“static copy vs animation copy” pair). Purely static tiles are never overwritten.
Fully transparent metatiles (every pixel index ``0``) are never stubbed, so empty animation
frames do not turn into visible noise.

Stub indices are drawn only from the **same 16-index sub-palette row** (``idx // 16``) as the
dominant row of the **original metatile pixels** before overwrite (fits GBA 4bpp / 16-color banks).
"""

from __future__ import annotations

import random
import struct
from collections import Counter
from typing import List, Optional, Set, Tuple, Iterable

from PIL import Image

from models import Point, TilesetInfos
from tileset_metatile_dedupe import MetatileDedupePlan


def _anim_src_points(tileset_infos: TilesetInfos, uid: int) -> Set[Point]:
    n = tileset_infos.get_tileset_anim_tiles_count(uid)
    return {tileset_infos.get_tileset_anim_tile_src(uid, i) for i in range(n)}


def _advance_paste_px(
    paste_x: int, paste_y: int, strip_w_px: int
) -> Tuple[int, int]:
    paste_x += 8
    if paste_x >= strip_w_px:
        assert paste_x == strip_w_px
        paste_x = 0
        paste_y += 8
    return paste_x, paste_y


def _paste_start_x_px(tile_px: int) -> int:
    m = tile_px >> 3
    return (m * m) * 8


def _strip_cursor_after_transparent(tile_px: int, strip_w_px: int) -> Tuple[int, int]:
    paste_x, paste_y = _paste_start_x_px(tile_px), 0
    while paste_x >= strip_w_px:
        paste_x -= strip_w_px
        paste_y += 8
    return paste_x, paste_y


def _is_all_palette_index_zero_fp(fp: bytes) -> bool:
    """True when every pixel is index 0 (typical fully transparent metatile in mode ``P``)."""
    # Avoid allocating ``b"\\x00" * len``; ``count`` is a fast C loop in CPython.
    return len(fp) > 0 and fp.count(0) == len(fp)


def _metatile_strip_fingerprint(
    im: Image.Image,
    paste_x: int,
    paste_y: int,
    sub: int,
    strip_w_px: int,
) -> bytes:
    tx, ty = paste_x, paste_y
    parts: List[bytes] = []
    for _ in range(sub):
        parts.append(im.crop((tx, ty, tx + 8, ty + 8)).tobytes())
        tx, ty = _advance_paste_px(tx, ty, strip_w_px)
    return b"".join(parts)


def _representative_used_src(
    tileset_infos: TilesetInfos,
    uid: int,
    metatile_index: int,
    plan: Optional[MetatileDedupePlan],
) -> Point:
    if plan is None:
        return tileset_infos.get_tileset_used_tile_src(uid, metatile_index)
    used_n = tileset_infos.get_tileset_used_tiles_count(uid)
    olds = [
        i
        for i in range(used_n)
        if plan.old_to_new[i] == metatile_index
    ]
    rep = min(olds)
    return tileset_infos.get_tileset_used_tile_src(uid, rep)


def _should_stub_metatile_for_grit_dedupe(
    metatile_index: int,
    rep_srcs: List[Point],
    rep_fps: List[bytes],
    anim_srcs: Set[Point],
) -> bool:
    """Stub only the animation-row copy when a pixel-identical **non-animation** used metatile exists."""
    if _is_all_palette_index_zero_fp(rep_fps[metatile_index]):
        # Fully transparent metatiles must stay empty; noise stubs would read as static garbage.
        return False
    src = rep_srcs[metatile_index]
    if src not in anim_srcs:
        return False
    my_fp = rep_fps[metatile_index]
    for j in range(len(rep_srcs)):
        if j == metatile_index:
            continue
        if rep_srcs[j] in anim_srcs:
            continue
        if rep_fps[j] == my_fp:
            return True
    return False


def _collect_all_8x8_fingerprints(im: Image.Image) -> Set[bytes]:
    out: Set[bytes] = set()
    w, h = im.size
    for py in range(0, h, 8):
        for px in range(0, w, 8):
            out.add(im.crop((px, py, px + 8, py + 8)).tobytes())
    return out


def _dominant_subpalette_row(indices: Iterable[int]) -> int:
    """Return ``row`` in ``0..15`` such that indices should stay in ``[row*16, row*16+15]``."""
    idxs = [int(p) & 0xFF for p in indices]
    if not idxs:
        return 0
    rows = [i // 16 for i in idxs]
    return Counter(rows).most_common(1)[0][0]


def _make_unique_stub_bytes(
    *,
    palette_choices: List[int],
    rng: random.Random,
    existing: Set[bytes],
) -> bytes:
    for _ in range(500):
        data = bytes(rng.choice(palette_choices) for _ in range(64))
        if data not in existing:
            return data
    for salt in range(10_000_000):
        data = bytes(
            palette_choices[(salt + i * 17) % len(palette_choices)] for i in range(64)
        )
        if data not in existing:
            return data
    raise RuntimeError("tileset_anim_grit_stubs: could not allocate unique 8x8 stub")


def apply_anim_grit_stubs_to_main_strip(
    im: Image.Image,
    *,
    tileset_infos: TilesetInfos,
    uid: int,
    tile_px: int,
    strip_w_px: int,
    tiles_count: int,
    plan: Optional[MetatileDedupePlan],
) -> None:
    """Overwrite 8×8 cells of **animation-vs-static grit duplicate** metatiles on the main strip.

    Mutates ``im`` in place (mode **P**). Call only when BG animation atlases are generated for
    this tileset; skip when ``generate_bg_animations`` is false.
    """
    if im.mode != "P":
        raise ValueError("apply_anim_grit_stubs_to_main_strip: expected mode P")

    m = tile_px >> 3
    sub = m * m

    existing = _collect_all_8x8_fingerprints(im)

    anim_srcs = _anim_src_points(tileset_infos, uid)
    rep_srcs = [
        _representative_used_src(tileset_infos, uid, met_i, plan)
        for met_i in range(tiles_count)
    ]
    px, py = _strip_cursor_after_transparent(tile_px, strip_w_px)
    rep_fps: List[bytes] = []
    for _ in range(tiles_count):
        rep_fps.append(_metatile_strip_fingerprint(im, px, py, sub, strip_w_px))
        for _ in range(sub):
            px, py = _advance_paste_px(px, py, strip_w_px)

    stub_flags = [
        _should_stub_metatile_for_grit_dedupe(i, rep_srcs, rep_fps, anim_srcs)
        for i in range(tiles_count)
    ]

    paste_x, paste_y = _strip_cursor_after_transparent(tile_px, strip_w_px)

    stub_seq = 0
    for met_i in range(tiles_count):
        if not stub_flags[met_i]:
            for _ in range(sub):
                paste_x, paste_y = _advance_paste_px(paste_x, paste_y, strip_w_px)
            continue

        mx, my = paste_x, paste_y
        # Metatile pixels follow the same strip walk as ``paste_used_tiles_into`` (not a solid
        # ``tile_px``×``tile_px`` rectangle in image space when the strip wraps).
        tx, ty = mx, my
        orig_indices: List[int] = []
        for _sy in range(m):
            for _sx in range(m):
                orig_indices.extend(
                    im.crop((tx, ty, tx + 8, ty + 8)).getdata()
                )
                tx, ty = _advance_paste_px(tx, ty, strip_w_px)

        row = _dominant_subpalette_row(orig_indices)
        palette_choices = [row * 16 + k for k in range(16)]

        for sy in range(m):
            for sx in range(m):
                stub_seq += 1

                # ``Random`` accepts ``int``, ``str``, ``bytes``, … — not tuples (Python 3.13+).
                rng = random.Random(
                    struct.pack(
                        "<QQQQQ",
                        uid & 0xFFFFFFFFFFFFFFFF,
                        met_i & 0xFFFFFFFFFFFFFFFF,
                        sx & 0xFFFFFFFFFFFFFFFF,
                        sy & 0xFFFFFFFFFFFFFFFF,
                        stub_seq & 0xFFFFFFFFFFFFFFFF,
                    )
                )
                data = _make_unique_stub_bytes(
                    palette_choices=palette_choices,
                    rng=rng,
                    existing=existing,
                )
                existing.add(data)

                tile_im = Image.frombytes("P", (8, 8), data)
                pal = im.getpalette()
                if pal is not None:
                    tile_im.putpalette(pal)
                im.paste(tile_im, (paste_x, paste_y))

                paste_x, paste_y = _advance_paste_px(paste_x, paste_y, strip_w_px)
