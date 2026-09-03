# SPDX-FileCopyrightText: Copyright 2025-2026 Guyeon Yu <copyrat90@gmail.com>, Nikolai Laptev (neparij)
# SPDX-License-Identifier: Zlib

"""Meta-tile deduplication: remap used tile indices, compact BMP, adjust map flips.

Grit indexes the BMP raster in export order; merging duplicates requires **fewer** pasted
metatiles **and** updating every reference (layer cells, custom data, enum tags, BG anim slot
templates) plus composing LDtk tile flips with the per-tile mapping to canonical pixels.

Singletons keep the natural LDtk crop and ``f_nat == 0``. For merged groups, the stored atlas tile
is the **representative** (minimum ``old_idx``) natural crop — not a min-bytes flip variant — so the
BMP stays aligned with the source image where possible; only non-representative members get a
non-zero ``f_nat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import LdtkJson
from PIL import Image, ImageOps

from models import Point, TilesetInfos


def ldtk_flip_metatile(im: Image.Image, f: int) -> Image.Image:
    """LDtk ``f``: bit0 = X flip, bit1 = Y flip."""
    out = im
    if f & 1:
        out = ImageOps.mirror(out)
    if f & 2:
        out = ImageOps.flip(out)
    return out


def compose_ldtk_flips(f_outer: int, f_inner: int, w: int, h: int) -> int:
    """Return ``f3`` such that ``ldtk_flip(f_outer, ldtk_flip(f_inner, p)) == ldtk_flip(f3, p)``."""

    def apply_f(f: int, x: int, y: int) -> Tuple[int, int]:
        if f == 0:
            return x, y
        if f == 1:
            return w - 1 - x, y
        if f == 2:
            return x, h - 1 - y
        return w - 1 - x, h - 1 - y

    for f3 in range(4):
        ok = True
        for x in range(w):
            for y in range(h):
                x1, y1 = apply_f(f_inner, x, y)
                x2, y2 = apply_f(f_outer, x1, y1)
                x3, y3 = apply_f(f3, x, y)
                if x2 != x3 or y2 != y3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return f3
    raise AssertionError("compose_ldtk_flips: no solution")


def _anim_src_points(tileset_infos: TilesetInfos, uid: int) -> Set[Point]:
    n = tileset_infos.get_tileset_anim_tiles_count(uid)
    return {tileset_infos.get_tileset_anim_tile_src(uid, i) for i in range(n)}


def _find_flip_natural_to_canonical(natural: Image.Image, canonical: Image.Image) -> int:
    """``canonical == ldtk_flip_metatile(natural, f)`` — return ``f`` (unique if data allows)."""
    for f in range(4):
        if ldtk_flip_metatile(natural, f).tobytes() == canonical.tobytes():
            return f
    raise AssertionError("metatile dedupe: natural cannot reach canonical via flips")


@dataclass(frozen=True)
class MetatileDedupePlan:
    """Per tileset: compact meta index 0..new_used_count-1."""

    new_used_count: int
    old_to_new: Tuple[int, ...]
    """``old_to_new[old_used_idx] -> new_used_idx``."""
    old_to_natural_flip: Tuple[int, ...]
    """``f_nat`` for ``compose_ldtk_flips(ldtk_f, f_nat, ...)``."""
    canonical_metatiles: Tuple[Image.Image, ...]
    """One PIL image per **new** index (mode matches source crop)."""


def build_metatile_dedupe_plan(
    tileset_def: LdtkJson.TilesetDefinition,
    tileset_infos: TilesetInfos,
    tileset_src_path: Path,
    *,
    generate_bg_animations: bool,
) -> Optional[MetatileDedupePlan]:
    """Return ``None`` if nothing to do (0 used tiles)."""
    uid = tileset_def.uid
    used_n = tileset_infos.get_tileset_used_tiles_count(uid)
    if used_n == 0:
        return None

    tile_px = tileset_def.tile_grid_size
    anim_srcs = _anim_src_points(tileset_infos, uid)
    if not generate_bg_animations:
        anim_srcs = set()

    with Image.open(tileset_src_path) as src:
        src.load()
        if src.mode not in ("P", "RGBA", "RGB"):
            raise ValueError(
                f"tileset_metatile_dedupe: unsupported image mode {src.mode} for {tileset_src_path}"
            )
        crops: List[Image.Image] = []
        keys: List[Tuple[bytes, ...]] = []

        for old_i in range(used_n):
            p = tileset_infos.get_tileset_used_tile_src(uid, old_i)
            box = (p.x, p.y, p.x + tile_px, p.y + tile_px)
            natural = src.crop(box)
            if p in anim_srcs:
                key = (b"\xffANIM\x00", old_i.to_bytes(4, "little"))
                keys.append(key)
            else:
                kbytes = min(
                    ldtk_flip_metatile(natural, f).tobytes() for f in range(4)
                )
                keys.append((kbytes,))
            crops.append(natural.copy())

    # Union by key; stable new order by min old_idx in group
    key_to_olds: Dict[Tuple[bytes, ...], List[int]] = {}
    for old_i, k in enumerate(keys):
        key_to_olds.setdefault(k, []).append(old_i)
    sorted_groups = sorted(key_to_olds.values(), key=lambda g: min(g))

    new_used_count = len(sorted_groups)
    old_to_new: List[int] = [0] * used_n
    canonical_metatiles: List[Image.Image] = []
    old_to_nat: List[int] = [0] * used_n

    for new_i, group in enumerate(sorted_groups):
        rep = min(group)
        natural_rep = crops[rep]

        # Singleton: do **not** pick a lexicographically minimal flip — that would add a bogus
        # ``f_nat`` (spurious Mirror H/V in maps), including for animation-isolated metatiles.
        # Flip canonical form is only needed when two or more used indices actually merge.
        if len(group) == 1:
            old_i = group[0]
            canonical_metatiles.append(natural_rep.copy())
            old_to_new[old_i] = new_i
            old_to_nat[old_i] = 0
            continue

        # Merged: atlas stores the representative's **natural** PNG/LDtk orientation (``rep`` =
        # min old_idx). Do not use min-bytes among flips — that often disagrees with the source and
        # forces ``f_nat`` on almost every cell (looks like "everything is mirrored").
        canonical = natural_rep.copy()
        canonical_metatiles.append(canonical)

        for old_i in group:
            old_to_new[old_i] = new_i
            nat = crops[old_i]
            old_to_nat[old_i] = _find_flip_natural_to_canonical(nat, canonical)

    return MetatileDedupePlan(
        new_used_count=new_used_count,
        old_to_new=tuple(old_to_new),
        old_to_natural_flip=tuple(old_to_nat),
        canonical_metatiles=tuple(canonical_metatiles),
    )


def build_metatile_dedupe_plans(
    ldtk_project: LdtkJson.LdtkJSON,
    tileset_infos: TilesetInfos,
    ldtk_project_folder_path: Path,
    *,
    enabled: bool,
    generate_bg_animations: bool,
) -> Dict[int, MetatileDedupePlan]:
    if not enabled:
        return {}
    out: Dict[int, MetatileDedupePlan] = {}
    for tileset_def in ldtk_project.defs.tilesets:
        if tileset_def.rel_path is None:
            continue
        p = ldtk_project_folder_path / tileset_def.rel_path
        if not p.exists():
            continue
        plan = build_metatile_dedupe_plan(
            tileset_def,
            tileset_infos,
            p,
            generate_bg_animations=generate_bg_animations,
        )
        if plan is not None:
            out[tileset_def.uid] = plan
    return out


def remap_used_tile_index(plan: MetatileDedupePlan, old_idx: int) -> int:
    return plan.old_to_new[old_idx]


def combine_tile_flip(
    plan: MetatileDedupePlan, old_idx: int, ldtk_f: int, tile_px: int
) -> int:
    return compose_ldtk_flips(
        ldtk_f, plan.old_to_natural_flip[old_idx], tile_px, tile_px
    )
