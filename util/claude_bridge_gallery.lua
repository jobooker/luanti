-- Gallery ops for the claude_bridge (2026-08-13): the interior
-- program's room gallery — interior-program-plan.md phase 1. APPEND to
-- the dev bridge init.lua after the scene ops (the assembled mod for
-- the Windows seat lives at mods/claude_bridge/init.lua; upstream
-- init.lua is jobooker/luanti-server mods/claude_bridge/init.lua).
--
-- The furnace and Cornell rooms are built from claude_bridge:* nodes
-- with UNIFORM single-color textures, so the tracer's cell albedo (the
-- minimap average color, game.cpp claudeVolumeSnapshot) is exact by
-- construction, and pathAlbedo() linearizes it as pow(c/255, 2.2):
--   texture 186 -> albedo 0.50        texture 221 -> albedo 0.73
-- CAVEAT (game.cpp, emissive forcing): cells with light_source > 0 are
-- warm-forced in the snapshot — r=255, g=max(g,200), b=max(b,120). For
-- the LIT variants that means only channels above the floor keep their
-- authored value: the BLUE channel carries the canonical rho for BOTH
-- variants (0.50 and 0.73); green carries it for 0.73 only; red is
-- always forced to 1.0. The analytic check is per-channel.

local function galnode(suffix, tex, light)
    core.register_node("claude_bridge:" .. suffix, {
        description = "claude gallery " .. suffix,
        tiles = { tex },
        is_ground_content = false,
        groups = { dig_immediate = 3 },
        light_source = light or 0,
    })
end
galnode("gray186", "claude_gray186.png", 0)      -- rho 0.50, matte
galnode("gray186_lit", "claude_gray186.png", 14) -- furnace A walls
galnode("gray221", "claude_gray221.png", 0)      -- rho 0.73 (Cornell white)
galnode("gray221_lit", "claude_gray221.png", 14) -- furnace B walls
galnode("red221", "claude_red221.png", 0)        -- Cornell left wall
galnode("green221", "claude_green221.png", 0)    -- Cornell right wall
galnode("white_lit", "claude_white255.png", 14)  -- Cornell ceiling emitter

local function box(p1, p2, name)  -- inclusive solid box fill
    for x = p1.x, p2.x do for y = p1.y, p2.y do for z = p1.z, p2.z do
        core.set_node({ x = x, y = y, z = z }, { name = name })
    end end end
end

-- Sealed, uniform-albedo, uniformly-emissive room — the analytic
-- referee (L = Le/(1-rho); interior-program-plan.md phase 1 item 2).
-- PURE by law: 1m cells only, no shape, no features. pos = exterior
-- SW floor corner; size = interior edge (default 5); variant = "050"
-- or "073". The 1x2 doorway (south wall center, floor level) is left
-- OPEN for walkability; plug it with OPS.door before measuring — the
-- plug is the wall node itself, so a shut furnace room has a perfectly
-- uniform interior.
function OPS.furnace(p)
    local o, s = p.pos, p.size or 5
    local wall = (p.variant == "073") and "claude_bridge:gray221_lit"
            or "claude_bridge:gray186_lit"
    local e = s + 1
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + e, y = o.y + e, z = o.z + e }, wall)
    box({ x = o.x + 1, y = o.y + 1, z = o.z + 1 },
        { x = o.x + s, y = o.y + s, z = o.z + s }, "air")
    local dx = o.x + 1 + math.floor((s - 1) / 2)
    box({ x = dx, y = o.y + 1, z = o.z },
        { x = dx, y = o.y + 2, z = o.z }, "air")
    return { door = { x = dx, y = o.y + 1, z = o.z }, wall = wall }
end

-- Classic Cornell box (phase 1 item 3): white floor/ceiling/back at
-- rho 0.73, red west (camera-left from the south door), green east,
-- one 3x3 ceiling area emitter, flush. size = interior edge (default
-- 7). Door as in OPS.furnace (white plug via OPS.door p.name).
function OPS.cornell(p)
    local o, s = p.pos, p.size or 7
    local e = s + 1
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + e, y = o.y + e, z = o.z + e }, "claude_bridge:gray221")
    box({ x = o.x, y = o.y + 1, z = o.z + 1 },
        { x = o.x, y = o.y + s, z = o.z + s }, "claude_bridge:red221")
    box({ x = o.x + e, y = o.y + 1, z = o.z + 1 },
        { x = o.x + e, y = o.y + s, z = o.z + s }, "claude_bridge:green221")
    box({ x = o.x + 1, y = o.y + 1, z = o.z + 1 },
        { x = o.x + s, y = o.y + s, z = o.z + s }, "air")
    local mid = o.x + 1 + math.floor((s - 1) / 2)
    local mz = o.z + 1 + math.floor((s - 1) / 2)
    box({ x = mid - 1, y = o.y + e, z = mz - 1 },
        { x = mid + 1, y = o.y + e, z = mz + 1 }, "claude_bridge:white_lit")
    -- two gray occluders (John, hand-placed on the first walkthrough,
    -- ratified 2026-08-13): the deep-shadow generators every classic
    -- Cornell render has — the region beside each block is lit only by
    -- bleed. Positions preserved exactly as placed.
    core.set_node({ x = o.x + 3, y = o.y + 1, z = o.z + 4 },
            { name = "claude_bridge:gray186" })
    core.set_node({ x = o.x + 6, y = o.y + 1, z = o.z + 6 },
            { name = "claude_bridge:gray186" })
    box({ x = mid, y = o.y + 1, z = o.z },
        { x = mid, y = o.y + 2, z = o.z }, "air")
    return { door = { x = mid, y = o.y + 1, z = o.z } }
end

-- The connecting hall (phase 1 item 4). Rooms sit in a row along the
-- z=0 line with their doors facing SOUTH; the hall runs east-west to
-- the south of them across a 3m daylight gap, and each room connects
-- through a 1-wide sealed stub corridor, so hall light cannot reach
-- any room's windows and no room leaks into another. Hall exterior
-- z -8..-4 (interior z -7..-5), height 3, neutral gray186 walls.
-- p = { y, x0, x1, doors = {x, ...}, torches = {x, ...} }
function OPS.hall(p)
    local y = p.y
    local wall = "claude_bridge:gray186"
    box({ x = p.x0, y = y, z = -8 }, { x = p.x1, y = y + 4, z = -4 }, wall)
    box({ x = p.x0 + 1, y = y + 1, z = -7 },
        { x = p.x1 - 1, y = y + 3, z = -5 }, "air")
    for _, dx in ipairs(p.doors or {}) do
        -- stub: sealed 1-wide corridor from hall north wall to the room
        box({ x = dx - 1, y = y, z = -4 }, { x = dx + 1, y = y + 3, z = -1 }, wall)
        box({ x = dx, y = y + 1, z = -4 }, { x = dx, y = y + 2, z = -1 }, "air")
    end
    -- walk-in entrances at both hall ends (the rooms are only
    -- reachable through the hall — caught by John on the first
    -- walkthrough, 2026-08-13: a gallery you cannot enter)
    box({ x = p.x0, y = y + 1, z = -6 }, { x = p.x0, y = y + 2, z = -6 }, "air")
    box({ x = p.x1, y = y + 1, z = -6 }, { x = p.x1, y = y + 2, z = -6 }, "air")
    local torch = core.registered_nodes["mcl_torches:torch_wall"]
            and "mcl_torches:torch_wall" or "default:torch_wall"
    local wm = core.dir_to_wallmounted({ x = 0, y = 0, z = -1 })
    for _, tx in ipairs(p.torches or {}) do
        core.set_node({ x = tx, y = y + 2, z = -7 },
                { name = torch, param2 = wm })
    end
    return { ok = true }
end

-- Open or shut a 1-wide x 2-high doorway. shut=true fills with p.name,
-- or with whatever node sits beside the doorway (so a furnace plug is
-- the furnace wall itself and the shut room is perfectly uniform).
function OPS.door(p)
    local d = p.pos
    local name = "air"
    if p.shut then
        name = p.name or core.get_node({ x = d.x - 1, y = d.y, z = d.z }).name
    end
    core.set_node({ x = d.x, y = d.y, z = d.z }, { name = name })
    core.set_node({ x = d.x, y = d.y + 1, z = d.z }, { name = name })
    return { pos = d, name = name }
end

-- Teleport by acting on the player object directly — the OPS.cmd
-- teleport chatcommand path fails silently without the bring priv
-- (worklog 2026-08-12). pitch is +up, matching OPS.look.
function OPS.tp(p)
    local pl = core.get_player_by_name(p.player or ADMIN)
    if not pl then error("not online: " .. tostring(p.player or ADMIN)) end
    pl:set_pos(p.pos)
    if p.yaw then pl:set_look_horizontal(math.rad(p.yaw)) end
    if p.pitch then pl:set_look_vertical(math.rad(-p.pitch)) end
    return { pos = pl:get_pos(), yaw = p.yaw, pitch = p.pitch }
end

-- Surface probe: the top-most non-air node in a column (loaded areas;
-- OPS.emerge_region first if needed).
function OPS.probe(p)
    for y = (p.ytop or 40), (p.ybot or -8), -1 do
        local n = core.get_node_or_nil({ x = p.x, y = y, z = p.z })
        if n and n.name ~= "air" and n.name ~= "ignore" then
            return { y = y, name = n.name }
        end
    end
    return { y = false, name = "all air or unloaded" }
end
