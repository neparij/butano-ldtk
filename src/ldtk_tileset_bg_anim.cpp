// SPDX-FileCopyrightText: Copyright 2025-2026 Guyeon Yu <copyrat90@gmail.com>, Nikolai Laptev (neparij)
// SPDX-License-Identifier: Zlib

#include "ldtk_tileset_bg_anim.h"

#include "ldtk_tileset_definition.h"

#include <bn_assert.h>
#include <bn_regular_bg_map_item.h>
#include <bn_tile.h>

namespace ldtk
{

namespace
{

BN_CODE_IWRAM [[nodiscard]] constexpr bn::regular_bg_map_cell cell_with_palette_id(bn::regular_bg_map_cell cell, int palette_id)
{
    const std::uint16_t w = static_cast<std::uint16_t>(cell);
    return bn::regular_bg_map_cell((w & 0x0FFFu) | (std::uint16_t(unsigned(palette_id)) << 12));
}

} // namespace

void tileset_bg_anim_controller::set_context(const tileset_definition& tileset, bn::regular_bg_tiles_ptr tiles_ptr,
                                             int layer_grid_size, bn::regular_bg_map_cell* map_cells,
                                             int map_cell_count)
{
    BN_ASSERT(layer_grid_size > 0 && (layer_grid_size & 7) == 0, "Invalid layer_grid_size: ", layer_grid_size);
    BN_ASSERT(map_cells && map_cell_count > 0, "tileset_bg_anim_controller needs live map cells");

    _tileset = &tileset;
    _tiles_ptr = bn::move(tiles_ptr);
    _template_map_cells = tileset.bg_item().map_item().cells_ptr();
    _anim_template_map_cells =
        tileset.anim_bg_item() ? tileset.anim_bg_item()->map_item().cells_ptr() : nullptr;
    _live_map_cells = map_cells;
    _live_map_cell_count = map_cell_count;
    _m_tile_cnt = layer_grid_size >> 3;
    _hw_ready = false;

    const bn::span<const tileset_bg_anim_group> groups = tileset.bg_anim_groups();
    BN_ASSERT(groups.empty() == (tileset.anim_bg_item() == nullptr), "bg_anim_groups / anim_bg_item mismatch");
    if(!groups.empty())
    {
        BN_BASIC_ASSERT(tileset.anim_bg_item() != nullptr);
    }
    BN_BASIC_ASSERT(groups.size() <= BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS, "Too many tile BG anim groups");

    _group_count = static_cast<std::uint8_t>(groups.size());
    for(int g = 0; g < _group_count; ++g)
    {
        _wait_left[g] = groups[g].wait_updates;
        _frame_idx[g] = 0;
    }

    if(_group_count)
    {
        _ensure_hw_indices();
    }
}

void tileset_bg_anim_controller::reset_context()
{
    _tileset = nullptr;
    _tiles_ptr.reset();
    _template_map_cells = nullptr;
    _anim_template_map_cells = nullptr;
    _live_map_cells = nullptr;
    _live_map_cell_count = 0;
    _m_tile_cnt = 0;
    _hw_ready = false;
    _group_count = 0;
}

void tileset_bg_anim_controller::_ensure_hw_indices()
{
    BN_BASIC_ASSERT(_tileset && _tiles_ptr.has_value() && _template_map_cells);
    BN_BASIC_ASSERT(_tileset->anim_bg_item() && _anim_template_map_cells);

    for(int gi = 0; gi < _group_count; ++gi)
    {
        _palette_cache_frame_at[gi] = 255;
    }

    const bn::span<const tileset_bg_anim_group> groups = _tileset->bg_anim_groups();
    const int m = _m_tile_cnt;
    const int msq = m * m;
    const bn::regular_bg_map_item& tmpl_map = _tileset->bg_item().map_item();
    const int cells_count = tmpl_map.cells_count();
    const bn::regular_bg_map_item& anim_map = _tileset->anim_bg_item()->map_item();
    const int anim_cells_count = anim_map.cells_count();

    for(int i = 0; i < 1024; ++i)
    {
        _hw_tile_index_to_flat[i] = -1;
    }

    int flat = 0;
    for(int gi = 0; gi < _group_count; ++gi)
    {
        const tileset_bg_anim_group& g = groups[gi];
        _hw_base[gi] = static_cast<std::uint16_t>(flat);

        for(int s = 0; s < g.slot_count; ++s)
        {
            BN_BASIC_ASSERT(flat < BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS, "Too many tile BG anim slots");

            const tileset_bg_anim_slot_template& sl = g.slots[s];
            const int cell_idx =
                int(sl.template_tile_index) * msq + int(sl.sub_ty) * m + int(sl.sub_tx);
            BN_ASSERT(cell_idx >= 0 && cell_idx < cells_count, "Bad anim template cell_idx: ", cell_idx, " / ",
                      cells_count);

            const bn::regular_bg_map_cell_info cell_info(_template_map_cells[cell_idx]);
            const int hw_tile = cell_info.tile_index();
            BN_ASSERT(hw_tile >= 0 && hw_tile < 1024, "Bad hw tile index: ", hw_tile);

            // If grit dedupes two slots to the same VRAM tile index, last slot wins for palette lookup.
            _hw_tile_index_to_flat[hw_tile] = static_cast<std::int16_t>(flat);

            _hw_indices[flat] = static_cast<std::uint16_t>(hw_tile);
            _last_gfx[flat] = 0xFFFF;
            ++flat;
        }
    }

    int grit_write = 0;
    for(int gi = 0; gi < _group_count; ++gi)
    {
        const tileset_bg_anim_group& g = groups[gi];
        _resolved_grit_base[gi] = static_cast<std::uint16_t>(grit_write);

        const int sc = g.slot_count;
        const int fc = g.frame_count;
        const bn::span<const std::uint16_t> t1b = g.frame_template_tile_1based;

        BN_ASSERT(t1b.size() == sc * fc, "Bad frame_template_tile_1based size");

        for(int f = 0; f < fc; ++f)
        {
            for(int s = 0; s < sc; ++s)
            {
                BN_BASIC_ASSERT(grit_write < BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES,
                                "tile BG anim resolve overflow");

                const std::uint16_t packed_1based = t1b[(f * sc) + s];
                const tileset_bg_anim_slot_template& sl = g.slots[s];
                const int cell_idx =
                    int(packed_1based) * msq + int(sl.sub_ty) * m + int(sl.sub_tx);
                BN_ASSERT(cell_idx >= 0 && cell_idx < anim_cells_count, "Bad anim source cell_idx: ", cell_idx,
                          " / ", anim_cells_count);

                const bn::regular_bg_map_cell_info src(_anim_template_map_cells[cell_idx]);
                const int grit = src.tile_index();
                const bn::span<const bn::tile> tiles_ref = _tileset->anim_bg_item()->tiles_item().tiles_ref();
                BN_ASSERT(grit >= 0 && grit < tiles_ref.size(), "Invalid anim grit tile index: ", grit, " / ",
                          tiles_ref.size());

                _resolved_grit_indices[grit_write] = static_cast<std::uint16_t>(grit);
                _resolved_palette_ids[grit_write] = static_cast<std::uint8_t>(src.palette_id());
                ++grit_write;
            }
        }
    }

    _resolved_grit_base[_group_count] = static_cast<std::uint16_t>(grit_write);

    _hw_ready = true;

    for(int gi = 0; gi < _group_count; ++gi)
    {
        _apply_grit_for_group(gi, 0);
    }

    if(_live_map_cells)
    {
        (void)_patch_map_palettes_all(true);
    }
}

void tileset_bg_anim_controller::sync_map_palettes()
{
    if(!_hw_ready || !_live_map_cells)
    {
        return;
    }

    (void)_patch_map_palettes_all(true);
}

BN_CODE_IWRAM bool tileset_bg_anim_controller::_refresh_flat_target_palette_cache()
{
    const bn::span<const tileset_bg_anim_group> groups = _tileset->bg_anim_groups();
    bool any_target_palette_changed = false;

    for(int gi = 0; gi < _group_count; ++gi)
    {
        const std::uint8_t frame = _frame_idx[gi];
        if(_palette_cache_frame_at[gi] == frame)
        {
            continue;
        }

        const std::uint8_t prev_tag = _palette_cache_frame_at[gi];

        const tileset_bg_anim_group& g = groups[gi];
        const int hwb = int(_hw_base[gi]);
        const int sc = int(g.slot_count);
        const int row0 = int(_resolved_grit_base[gi]) + int(frame) * sc;

        for(int s = 0; s < sc; ++s)
        {
            const std::uint8_t new_p = _resolved_palette_ids[row0 + s];
            const int idx = hwb + s;
            const std::uint8_t old_p = _flat_target_palette[idx];
            _flat_target_palette[idx] = new_p;

            if(prev_tag == 255)
            {
                any_target_palette_changed = true;
            }
            else if(new_p != old_p)
            {
                any_target_palette_changed = true;
            }
        }

        _palette_cache_frame_at[gi] = frame;
    }

    return any_target_palette_changed;
}

BN_CODE_IWRAM bool tileset_bg_anim_controller::_patch_map_palettes_all(bool force_scan_map_cells)
{
    BN_BASIC_ASSERT(_live_map_cells);

    const bool targets_changed = _refresh_flat_target_palette_cache();

    if(!force_scan_map_cells && !targets_changed)
    {
        return false;
    }

    bn::regular_bg_map_cell* const cells = _live_map_cells;
    const int n = _live_map_cell_count;
    bool changed = false;

    for(int i = 0; i < n; ++i)
    {
        const std::uint16_t w = static_cast<std::uint16_t>(cells[i]);
        const int ti = int(w & 0x3FFu);
        const int flat = int(_hw_tile_index_to_flat[ti]);
        if(flat < 0)
        {
            continue;
        }

        const int target_pal = int(_flat_target_palette[flat]);
        const int cur_pal = int((w >> 12) & 0xFu);
        if(cur_pal != target_pal)
        {
            cells[i] = cell_with_palette_id(cells[i], target_pal);
            changed = true;
        }
    }

    return changed;
}

void tileset_bg_anim_controller::_apply_grit_for_group(int group_index, int frame_index)
{
    const bn::span<const tileset_bg_anim_group> groups = _tileset->bg_anim_groups();
    const tileset_bg_anim_group& g = groups[group_index];
    const int sc = g.slot_count;
    const int hwb = _hw_base[group_index];
    const int grit_row_base = _resolved_grit_base[group_index];
    const bn::span<const bn::tile> tiles_ref = _tileset->anim_bg_item()->tiles_item().tiles_ref();

    BN_ASSERT(frame_index >= 0 && frame_index < g.frame_count, "Bad frame_index");

    const int grit_row = grit_row_base + (frame_index * sc);
    for(int s = 0; s < sc; ++s)
    {
        const std::uint16_t grit = _resolved_grit_indices[grit_row + s];
        if(_last_gfx[hwb + s] != grit)
        {
            _last_gfx[hwb + s] = grit;
            _tiles_ptr->overwrite_tile(_hw_indices[hwb + s], tiles_ref[grit]);
        }
    }
}

bool tileset_bg_anim_controller::update()
{
    if(!_tileset || !_tiles_ptr.has_value() || !_template_map_cells)
    {
        return false;
    }

    BN_BASIC_ASSERT(_hw_ready, "tileset_bg_anim_controller not initialized");

    const bn::span<const tileset_bg_anim_group> groups = _tileset->bg_anim_groups();
    bool advanced_any = false;

    for(int gi = 0; gi < _group_count; ++gi)
    {
        const tileset_bg_anim_group& g = groups[gi];

        if(_wait_left[gi] != 0)
        {
            --_wait_left[gi];
            continue;
        }

        _wait_left[gi] = g.wait_updates;

        const int next_frame = (int(_frame_idx[gi]) + 1) % int(g.frame_count);
        _frame_idx[gi] = static_cast<std::uint8_t>(next_frame);
        _apply_grit_for_group(gi, next_frame);
        advanced_any = true;
    }

    if(advanced_any && _live_map_cells)
    {
        return _patch_map_palettes_all(false);
    }

    return false;
}

} // namespace ldtk