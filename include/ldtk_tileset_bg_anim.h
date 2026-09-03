// SPDX-FileCopyrightText: Copyright 2025-2026 Guyeon Yu <copyrat90@gmail.com>, Nikolai Laptev (neparij)
// SPDX-License-Identifier: Zlib

#pragma once

#include "ldtk_tile_index.h"

#include <bn_common.h>
#include <bn_optional.h>
#include <bn_regular_bg_map_cell_info.h>
#include <bn_regular_bg_tiles_item.h>
#include <bn_regular_bg_tiles_ptr.h>
#include <bn_span.h>

#include <cstdint>

#ifndef BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS
/**
 * @brief Max total 8x8 tile slots (sum over all groups) for ldtk::tileset_bg_anim_controller.
 */
#define BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS 96
#endif

#ifndef BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS
#define BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS 16
#endif

#ifndef BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES
/**
 * @brief Max (slot_count * frame_count) summed over all groups on one tileset; one-time resolve to grit indices.
 */
#define BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES 512
#endif

namespace ldtk
{

class tileset_definition;

/// @brief One map template cell for ldtk::tileset_bg_anim_group (layer grid_size / 8 = m).
struct tileset_bg_anim_slot_template final
{
    /// @brief Same 1-based packed index as ldtk tile grid cells (`tile_info.index`).
    tile_index template_tile_index;
    std::uint8_t sub_tx;
    std::uint8_t sub_ty;
};

/// @brief Prebuilt tileset animation: packed 1-based tile indices at codegen; grit indices resolved once on GBA.
struct tileset_bg_anim_group final
{
    std::uint8_t wait_updates;
    std::uint8_t slot_count;
    std::uint8_t frame_count;
    bn::span<const tileset_bg_anim_slot_template> slots;
    /// Per (frame, slot): packed 1-based meta-tile index into the ``*_anim`` map template.
    bn::span<const std::uint16_t> frame_template_tile_1based;
};

/// @brief Steps precomputed ldtk::tileset_bg_anim_group data using bn::regular_bg_tiles_ptr::overwrite_tile.
class tileset_bg_anim_controller final
{
public:
    tileset_bg_anim_controller() = default;

    /// @brief Binds controller to a tileset's animations and the live BG tiles.
    /// @param tileset Must outlive the controller; template map cells come from `tileset.bg_item().map_item()`.
    /// @param tiles_ptr Must reference the same tiles as the level background built from this tileset.
    /// @param layer_grid_size Layer instance grid_size in pixels (8 or 16, …).
    /// @param map_cells Same buffer as the layer's `regular_bg_map_item` (e.g. level_bgs_manager 32×32 cells).
    /// @param map_cell_count `width * height` of that map (e.g. 1024).
    void set_context(const tileset_definition& tileset, bn::regular_bg_tiles_ptr tiles_ptr, int layer_grid_size,
                     bn::regular_bg_map_cell* map_cells, int map_cell_count);

    void reset_context();

    /// @brief After scrolling / map regen, re-apply palette_id for current anim frames (cells were rebuilt from main map).
    void sync_map_palettes();

    /// @brief Call once per frame; advances timers and applies overwrite_tile when the graphic changes.
    /// @return True if map RAM cells were changed (caller should `reload_cells_ref()`).
    [[nodiscard]] bool update();

private:
    const tileset_definition* _tileset = nullptr;
    bn::optional<bn::regular_bg_tiles_ptr> _tiles_ptr;
    /// Grit tileset map (`tileset.bg_item().map_item()`), same indexing as ldtk_level_bgs_manager::reset_rows.
    const bn::regular_bg_map_cell* _template_map_cells = nullptr;
    /// Grit ``*_anim`` map for frame graphics (same m² layout per meta-tile as main tileset).
    const bn::regular_bg_map_cell* _anim_template_map_cells = nullptr;
    bn::regular_bg_map_cell* _live_map_cells = nullptr;
    int _live_map_cell_count = 0;
    int _m_tile_cnt = 0;
    bool _hw_ready = false;

    std::uint8_t _group_count = 0;
    std::uint8_t _wait_left[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS]{};
    std::uint8_t _frame_idx[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS]{};
    std::uint16_t _hw_base[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS]{};

    std::uint16_t _hw_indices[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS]{};
    std::uint16_t _last_gfx[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS]{};

    std::uint16_t _resolved_grit_indices[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES]{};
    std::uint8_t _resolved_palette_ids[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_RESOLVED_ENTRIES]{};
    std::uint16_t _resolved_grit_base[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS + 1]{};

    /// VRAM tile index (0–1023) → flat slot index, or -1 if not an animated tile.
    std::int16_t _hw_tile_index_to_flat[1024]{};
    /// Target palette_id per flat slot for the current animation frame (filled by _refresh_flat_target_palette_cache).
    std::uint8_t _flat_target_palette[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_SLOTS]{};
    /// Last `_frame_idx[gi]` reflected in `_flat_target_palette` for that group (0xFF = stale / force refresh).
    std::uint8_t _palette_cache_frame_at[BN_CFG_LDTK_TILESET_BG_ANIM_MAX_GROUPS]{};

    void _ensure_hw_indices();
    void _apply_grit_for_group(int group_index, int frame_index);
    /// @return True if any flat slot's target palette_id changed vs previous cache (or first fill for a group).
    [[nodiscard]] BN_CODE_IWRAM bool _refresh_flat_target_palette_cache();
    /// @param force_scan_map_cells If true, always reconcile live map cells (after scroll / init). If false, skip the
    ///        full map scan when no target palette changed (grit-only frame advance).
    [[nodiscard]] BN_CODE_IWRAM bool _patch_map_palettes_all(bool force_scan_map_cells);
};

} // namespace ldtk
