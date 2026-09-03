# SPDX-FileCopyrightText: Copyright 2025-2026 Guyeon Yu <copyrat90@gmail.com>, Nikolai Laptev (neparij)
# SPDX-License-Identifier: Zlib

"""Parse LDtk tileset customData tile animations and precompute Butano tile indices."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import LdtkJson
from models import Point, TilesetInfos
from tileset_metatile_dedupe import MetatileDedupePlan

# Must match BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES in ldtk_tileset_bg_anim.h
MAX_FRAME_RESOLVE_ENTRIES: int = 512


@dataclass(frozen=True)
class BgTileAnimSpec:
    rect_w: int
    rect_h: int
    wait_updates: int
    frame_offsets: Tuple[int, ...]


def _parse_kv_lines(data: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 1:
            out[parts[0]] = ""
        else:
            out[parts[0]] = parts[1].strip()
    return out


def try_parse_bg_tile_animation(data: str) -> Optional[BgTileAnimSpec]:
    """If `data` defines a tile BG animation (all three keys), return spec; else None."""
    kv = _parse_kv_lines(data)
    if not kv:
        return None
    keys = {
        "animation_rect_size",
        "animation_wait_time_updates",
        "animation_frames",
    }
    if not keys.issubset(kv.keys()):
        return None
    if any(k.startswith("animation_") for k in kv.keys() if k not in keys):
        raise ValueError(
            "Unknown animation_* key in tileset customData — supported keys are only: "
            + ", ".join(sorted(keys))
        )

    m_rect = re.fullmatch(r"(\d+)\s*,\s*(\d+)", kv["animation_rect_size"])
    if not m_rect:
        raise ValueError(
            f'Invalid animation_rect_size "{kv["animation_rect_size"]}" (expected "W,H")'
        )
    rect_w, rect_h = int(m_rect.group(1)), int(m_rect.group(2))
    if rect_w < 1 or rect_h < 1 or rect_w > 32 or rect_h > 32:
        raise ValueError(
            f"animation_rect_size out of range (1..32): {rect_w}x{rect_h}"
        )

    m_wait = re.fullmatch(r"(\d+)", kv["animation_wait_time_updates"])
    if not m_wait:
        raise ValueError(
            f'Invalid animation_wait_time_updates "{kv["animation_wait_time_updates"]}"'
        )
    wait_updates = int(m_wait.group(1))
    if wait_updates < 1 or wait_updates > 255:
        raise ValueError(f"animation_wait_time_updates out of range (1..255): {wait_updates}")

    # animation_frames: block index ``off`` from the anchor (see _bg_anim_frame_block_top_left_tile_id).
    frame_parts = [p.strip() for p in kv["animation_frames"].split(",") if p.strip() != ""]
    if len(frame_parts) < 2:
        raise ValueError("animation_frames must list at least two offsets")
    frame_offsets = tuple(int(p) for p in frame_parts)
    if len(frame_offsets) > 64:
        raise ValueError(f"Too many animation frames (max 64), got {len(frame_offsets)}")
    for o in frame_offsets:
        if o < -512 or o > 512:
            raise ValueError(f"animation_frames offset out of range (-512..512): {o}")

    return BgTileAnimSpec(rect_w, rect_h, wait_updates, frame_offsets)


def _bg_anim_frame_block_top_left_tile_id(
    tileset_ident: str,
    c_wid: int,
    c_hei: int,
    anchor: int,
    rect_w: int,
    rect_h: int,
    off: int,
) -> int:
    """LDtk id of the top-left tile of the animation rect for frame index ``off``.

    ``off == 0`` is the anchor. For ``off != 0``, move the block's top-left by ``off`` steps:
    each step shifts **right** by ``rect_w`` tiles (same as pixel step ``rect_w * tileGridSize``
    on a grid with no spacing). If the block would extend past the right edge, subtract
    ``c_wid`` from the column and add ``rect_h`` to the row; if ``cx < 0``, add ``c_wid`` and
    subtract ``rect_h``. This is applied ``|off|`` times from the anchor (see animation_frames).
    """
    if rect_w < 1 or rect_h < 1:
        raise AssertionError("rect_w / rect_h")
    if c_wid < 1 or c_hei < 1:
        raise ValueError(f'Tileset "{tileset_ident}": invalid tileset dimensions')
    if rect_w > c_wid:
        raise ValueError(
            f'Tileset "{tileset_ident}": tileset width {c_wid} < animation_rect width {rect_w}'
        )

    ax = anchor % c_wid
    ay = anchor // c_wid
    if ax + rect_w > c_wid or ay + rect_h > c_hei:
        raise ValueError(
            f'Tileset "{tileset_ident}": animation anchor {anchor} places {rect_w}x{rect_h} '
            f"rect outside tileset"
        )

    if off == 0:
        return anchor

    cx = ax
    cy = ay
    step = 1 if off > 0 else -1
    for _ in range(abs(off)):
        cx += step * rect_w
        while cx < 0:
            cx += c_wid
            cy -= rect_h
        while cx + rect_w > c_wid:
            cx -= c_wid
            cy += rect_h

    if cx < 0 or cx + rect_w > c_wid or cy < 0 or cy + rect_h > c_hei:
        raise ValueError(
            f'Tileset "{tileset_ident}": animation frame offset {off} ends outside tileset '
            f"{c_wid}x{c_hei} (top-left grid=({cx},{cy}), {rect_w}x{rect_h})"
        )
    return cx + cy * c_wid


def collect_bg_animation_tile_srcs_per_tileset(
    ldtk_project: LdtkJson.LdtkJSON,
    tilesets_level_used_tiles: List[Set[Point]],
) -> List[List[Point]]:
    """Per-tileset ordered list of atlas src Points for BG animation frames (separate grit).

    Level-used sets are unchanged. When a customData entry defines a BG animation and its
    anchor ``tile_id`` is on a level, every LDtk tile id touched by the animation is listed
    here once (order stable) for ``ldtk_gen_priv_tileset_*_anim`` export.
    """

    def tile_id_to_src(tileset_def: LdtkJson.TilesetDefinition, tile_id: int) -> Point:
        grid_x = tile_id % tileset_def.c_wid
        grid_y = tile_id // tileset_def.c_wid
        # Must match TilesetInfos.__get_tile_src (LDtk: stride = tileGridSize + spacing).
        square_diff = tileset_def.tile_grid_size + tileset_def.spacing
        return Point(
            tileset_def.padding + grid_x * square_diff,
            tileset_def.padding + grid_y * square_diff,
        )

    n = len(ldtk_project.defs.tilesets)
    seen: List[Set[Point]] = [set() for _ in range(n)]
    ordered: List[List[Point]] = [[] for _ in range(n)]

    for ts_idx, tileset_def in enumerate(ldtk_project.defs.tilesets):
        used = tilesets_level_used_tiles[ts_idx]
        max_tid = tileset_def.c_wid * tileset_def.c_hei

        for custom_data in tileset_def.custom_data:
            spec = try_parse_bg_tile_animation(custom_data.data)
            if spec is None:
                continue
            anchor = custom_data.tile_id
            anchor_src = tile_id_to_src(tileset_def, anchor)
            if anchor_src not in used:
                continue

            c_wid = tileset_def.c_wid
            c_hei = tileset_def.c_hei
            # Deterministic order: full anchor rect first (row-major), then each frame block
            # in animation_frames order with the same cell order. (Using a set only for ids
            # made the *_anim.bmp follow "per-cell then all frames", which is confusing and
            # looked like wrong tiles vs frame sequence.)
            lids_order: List[int] = []
            lids_seen: Set[int] = set()

            def add_lid(lid: int) -> None:
                if lid in lids_seen:
                    return
                lids_seen.add(lid)
                lids_order.append(lid)

            for ry in range(spec.rect_h):
                for rx in range(spec.rect_w):
                    add_lid(anchor + rx + ry * c_wid)
            for off in spec.frame_offsets:
                top = _bg_anim_frame_block_top_left_tile_id(
                    tileset_def.identifier,
                    c_wid,
                    c_hei,
                    anchor,
                    spec.rect_w,
                    spec.rect_h,
                    off,
                )
                for ry in range(spec.rect_h):
                    for rx in range(spec.rect_w):
                        add_lid(top + rx + ry * c_wid)

            for lid in lids_order:
                if lid < 0 or lid >= max_tid:
                    raise ValueError(
                        f'Tileset "{tileset_def.identifier}": BG animation references LDtk tile id {lid} '
                        f"outside tileset bounds (0..{max_tid - 1})"
                    )
                p = tile_id_to_src(tileset_def, lid)
                if p not in seen[ts_idx]:
                    seen[ts_idx].add(p)
                    ordered[ts_idx].append(p)

    return ordered


def assert_single_m_tile_cnt_for_tileset(
    ldtk_project: LdtkJson.LdtkJSON, tileset_uid: int, tileset_ident: str
) -> int:
    """All tile layers using this tileset must share the same grid_size-derived m."""
    m_values: List[int] = []
    for level in ldtk_project.levels:
        if level.layer_instances is None:
            continue
        for layer in level.layer_instances:
            if layer.tileset_def_uid != tileset_uid:
                continue
            if not layer.auto_layer_tiles and not layer.grid_tiles:
                continue
            m_values.append(layer.grid_size >> 3)
    if not m_values:
        return 1
    if min(m_values) != max(m_values):
        raise ValueError(
            f'Tileset "{tileset_ident}" tile animations require a single layer grid size '
            f"(grid_size/8) across all layers using this tileset; got {sorted(set(m_values))}"
        )
    return m_values[0]


def build_animation_group_payload(
    tileset_def: LdtkJson.TilesetDefinition,
    tileset_infos: TilesetInfos,
    ldtk_project: LdtkJson.LdtkJSON,
    anchor_ldtk_tile_id: int,
    spec: BgTileAnimSpec,
    metatile_dedupe_plan: Optional[MetatileDedupePlan] = None,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    Returns (slots, frame_template_tile_1based_frame_major).

    Each slot is (template_tile_index_1based, sub_tx, sub_ty) with sub_* in [0, m-1]
    for m = max layer grid_size / 8 among layers using this tileset.

    frame_template_tile_1based: for frame f, slot s: packed 1-based **anim-atlas** meta-tile
    index (separate ``*_anim`` grit); runtime resolves via ``anim_bg_item().map_item()``.
    Slot templates still use the level tileset (``bg_item()``). ``animation_frames`` are
    block indices in reading order (see ``_bg_anim_frame_block_top_left_tile_id``).
    """
    uid = tileset_def.uid
    m = assert_single_m_tile_cnt_for_tileset(ldtk_project, uid, tileset_def.identifier)
    subs = tileset_def.tile_grid_size >> 3
    if m != subs:
        raise ValueError(
            f'Tileset "{tileset_def.identifier}": for tile animations, every layer using this '
            f"tileset must have grid_size/8 ({m}) equal to tileGridSize/8 ({subs})"
        )

    c_wid = tileset_def.c_wid
    c_hei = tileset_def.c_hei
    # Validates anchor aligns to WxH block grid and c_wid/c_hei multiples.
    _bg_anim_frame_block_top_left_tile_id(
        tileset_def.identifier,
        c_wid,
        c_hei,
        anchor_ldtk_tile_id,
        spec.rect_w,
        spec.rect_h,
        0,
    )

    slot_templates: List[Tuple[int, int, int]] = []
    for ry in range(spec.rect_h):
        for rx in range(spec.rect_w):
            base_ldtk = anchor_ldtk_tile_id + rx + ry * c_wid
            if not tileset_infos.get_tileset_is_used_tile_id(uid, base_ldtk):
                raise ValueError(
                    f'Tileset "{tileset_def.identifier}": animation rect cell LDtk id {base_ldtk} '
                    "is not a used tile"
                )
            compact = tileset_infos.get_tileset_used_tile_id_to_idx(uid, base_ldtk)
            if metatile_dedupe_plan is not None:
                compact = metatile_dedupe_plan.old_to_new[compact]
            template_1based = 1 + compact
            for sub_ty in range(m):
                for sub_tx in range(m):
                    slot_templates.append((template_1based, sub_tx, sub_ty))

    slot_count = len(slot_templates)
    frame_count = len(spec.frame_offsets)
    frame_targets: List[int] = []

    for f in range(frame_count):
        off = spec.frame_offsets[f]
        frame_top = _bg_anim_frame_block_top_left_tile_id(
            tileset_def.identifier,
            c_wid,
            c_hei,
            anchor_ldtk_tile_id,
            spec.rect_w,
            spec.rect_h,
            off,
        )
        for ry in range(spec.rect_h):
            for rx in range(spec.rect_w):
                base_ldtk = anchor_ldtk_tile_id + rx + ry * c_wid
                frame_ldtk = frame_top + rx + ry * c_wid
                if frame_ldtk < 0 or not tileset_infos.get_tileset_anim_contains_ldtk_tile_id(
                    uid, frame_ldtk
                ):
                    raise ValueError(
                        f'Tileset "{tileset_def.identifier}": frame block index {off} '
                        f"(top-left id {frame_top}) cell (+{rx},+{ry}) from base {base_ldtk} "
                        f"yields LDtk id {frame_ldtk} missing from animation tile atlas"
                    )
                target_1based = 1 + tileset_infos.get_tileset_anim_tile_id_to_idx(
                    uid, frame_ldtk
                )
                for _sub_ty in range(m):
                    for _sub_tx in range(m):
                        frame_targets.append(target_1based)

    if len(frame_targets) != slot_count * frame_count:
        raise AssertionError("internal frame target table size mismatch")

    if len(frame_targets) > MAX_FRAME_RESOLVE_ENTRIES:
        raise ValueError(
            f'Tileset "{tileset_def.identifier}": animation keyframe table too large '
            f"({len(frame_targets)} entries, max {MAX_FRAME_RESOLVE_ENTRIES})"
        )

    return slot_templates, frame_targets
