-- Gallery ops for the claude_bridge (2026-08-13): the interior
-- program's room gallery — interior-program-plan.md phase 1. APPEND to
-- the dev bridge init.lua after the scene ops (the assembled mod for
-- the Windows seat lives at mods/claude_bridge/init.lua; upstream
-- init.lua is jobooker/luanti-server mods/claude_bridge/init.lua).
--
-- The furnace and Cornell rooms are built from claude_bridge:* nodes
-- with UNIFORM single-color textures, so the tracer's cell albedo (the
-- minimap average color, game.cpp claudeTraceGridSnapshot) is exact by
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
galnode("white255", "claude_white255.png", 0)    -- white_lit's unlit twin
                                                  -- (OPS.lamps toggles them)

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

-- Sealed sky-referee cave (roadmap 1b): a uniform rho=0.50 box like
-- OPS.furnace but non-cubic (interior sx wide x sy tall x sz deep, no
-- lamps) with ONE 1x1 ceiling opening at centre -- open (glazed=false
-- misnomer aside: glazed=true means OPEN, sky-visible) for
-- cave-skylight, or plugged with the wall node itself for cave-glass
-- ("glass IS opaque today, so build what it is" -- John 2026-08-15; the
-- plug becomes an actual glass node only in 2b, as its own change, same
-- coordinates). Walk-in door as OPS.furnace; plug it with OPS.door
-- before measuring, same as every referee room.
-- in: { pos={x,y,z}, sx=7, sy=5, sz=7, glazed=true|false }
function OPS.skycave(p)
    local o = p.pos
    local sx, sy, sz = p.sx or 7, p.sy or 5, p.sz or 7
    local ex, ey, ez = sx + 1, sy + 1, sz + 1
    local wall = "claude_bridge:gray186"
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + ex, y = o.y + ey, z = o.z + ez }, wall)
    box({ x = o.x + 1, y = o.y + 1, z = o.z + 1 },
        { x = o.x + sx, y = o.y + sy, z = o.z + sz }, "air")
    local dx = o.x + 1 + math.floor((sx - 1) / 2)
    box({ x = dx, y = o.y + 1, z = o.z },
        { x = dx, y = o.y + 2, z = o.z }, "air")
    local cx = o.x + 1 + math.floor((sx - 1) / 2)
    local cz = o.z + 1 + math.floor((sz - 1) / 2)
    local hole = { x = cx, y = o.y + ey, z = cz }
    core.set_node(hole, { name = p.glazed and "air" or wall })
    return { door = { x = dx, y = o.y + 1, z = o.z }, hole = hole,
             glazed = p.glazed and true or false }
end

-- Open sky-furnace pad (roadmap 1b): a flat rho=0.50 slab, no walls, no
-- roof -- the L = rho*L_sky referee once 2b ships a uniform-sky test
-- dial. A low (1-node) fence at radius >= 12 from centre keeps the
-- referee's grazing-angle sky occlusion under 1% (measured/derived,
-- handoff 2026-08-15-gallery-phase2-referees.md) without walling in the
-- hemisphere the pad needs to see.
-- in: { center={x,y,z}, half=12, fence=13 }
function OPS.skypad(p)
    local o = p.center
    local half = p.half or 12
    local fdist = p.fence or 13
    for x = o.x - half, o.x + half - 1 do
        for z = o.z - half, o.z + half - 1 do
            core.set_node({ x = x, y = o.y, z = z },
                    { name = "claude_bridge:gray186" })
        end
    end
    local fence = (core.registered_nodes["mcl_fences:oak_fence"] and
                  "mcl_fences:oak_fence")
            or (core.registered_nodes["mcl_fences:spruce_fence"] and
               "mcl_fences:spruce_fence")
            or (core.registered_nodes["default:fence_wood"] and
               "default:fence_wood")
            or "claude_bridge:gray186"
    for x = o.x - fdist, o.x + fdist do
        core.set_node({ x = x, y = o.y + 1, z = o.z - fdist }, { name = fence })
        core.set_node({ x = x, y = o.y + 1, z = o.z + fdist }, { name = fence })
    end
    for z = o.z - fdist, o.z + fdist do
        core.set_node({ x = o.x - fdist, y = o.y + 1, z = z }, { name = fence })
        core.set_node({ x = o.x + fdist, y = o.y + 1, z = z }, { name = fence })
    end
    return { ok = true, fence = fence, half = half, fdist = fdist }
end

-- =====================================================================
-- TRANSPARENCY REFEREE ROOMS (roadmap coverage 4, 2026-08-18)
-- =====================================================================
--
-- WHAT THEY ARE FOR, before any coordinate. Until today every non-air
-- cell in this renderer was opaque: glass was a solid block and a lit
-- room seen through a window was black. These two rooms are the pair of
-- measurements that say whether that stopped being true and whether it
-- stopped being true WITHOUT inventing or losing light.
--
-- cave-glass IS NOT ONE OF THEM AND MUST NOT BECOME ONE. Despite its
-- name it is plugged with the WALL node, deliberately -- it is the
-- sealed control that caught the origin-cell leak, and it is the only
-- bit-exactly black instrument in this harness. These rooms are built
-- BESIDE it (in fact 85 nodes north of everything), not out of it.
--
-- WHY SO FAR NORTH. The trace grid is 128^3 centred on the camera, so a
-- room within 64 nodes of a CI vantage is inside that capture's bubble.
-- These rooms contain emissive walls, and an emissive cell inside the
-- bubble joins the area-emitter list the estimator samples from -- a
-- cap-16 list sorted by distance, so a new lit room could silently
-- displace a referee room's own lights. z >= 170 clears every vantage on
-- the z axis alone (the northernmost is the sky pad at z = 104).

-- A LIT CHAMBER AND A DARK ONE, SHARING ONE WALL -- the "does light get
-- through" measurement, and the whole point is that it is a DIFFERENCE.
--
-- Chamber A (west) has furnace-050 walls: uniform rho = 0.50, uniformly
-- emissive, so its interior radiance is the analytic Le/(1-rho). Chamber
-- B (east) has the same walls UNLIT, so with an opaque partition it is
-- honestly black -- and every photon it ever sees had to cross the
-- partition. The partition is the variable and nothing else is: same
-- geometry, same albedo, same vantage, one node changed.
--
--   p.pane = "glass" | "water" | "opaque" | "air"
--
-- "opaque" is the before-half of the A/B and "air" opens the two
-- chambers into one room, which is the control that says the lit side is
-- lit at all. Walk-in doors on the z = o.z wall for both chambers, in
-- the gallery's own idiom; plug them with OPS.door before measuring.
--
-- in: { pos={x,y,z}, s=5, pane="glass" }
function OPS.glasspair(p)
    local o = p.pos
    local s = p.s or 5              -- interior edge of each chamber
    local lit = "claude_bridge:gray186_lit"
    local dark = "claude_bridge:gray186"
    local panes = {
        glass = { "mcl_core:glass", "default:glass" },
        water = { "mcl_core:water_source", "default:water_source" },
        opaque = { dark },
        air = { "air" },
    }
    local want = panes[p.pane or "glass"]
    local pane = nil
    for _, n in ipairs(want) do
        if n == "air" or core.registered_nodes[n] then pane = n break end
    end
    if not pane then return { error = "no node for pane " .. tostring(p.pane),
                              tried = want } end
    -- exterior: wall + s + partition + s + wall on x; wall + s + wall on
    -- y and z
    local ex = 2 * s + 2
    local ey, ez = s + 1, s + 1
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + ex, y = o.y + ey, z = o.z + ez }, dark)
    -- chamber A is a FURNACE: every face of it emissive, including the
    -- partition's own A-side is NOT (the partition is the variable)
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + s + 1, y = o.y + ey, z = o.z + ez }, lit)
    box({ x = o.x + 1, y = o.y + 1, z = o.z + 1 },
        { x = o.x + s, y = o.y + s, z = o.z + s }, "air")
    box({ x = o.x + s + 2, y = o.y + 1, z = o.z + 1 },
        { x = o.x + ex - 1, y = o.y + s, z = o.z + s }, "air")
    -- THE PARTITION, one node thick, spanning the full interior cross
    -- section. Every path from A to B crosses exactly one of these cells.
    local px = o.x + s + 1
    box({ x = px, y = o.y + 1, z = o.z + 1 },
        { x = px, y = o.y + s, z = o.z + s }, pane)
    -- walk-in doors, one per chamber, on the south wall
    local dax = o.x + 1 + math.floor((s - 1) / 2)
    local dbx = o.x + s + 2 + math.floor((s - 1) / 2)
    box({ x = dax, y = o.y + 1, z = o.z }, { x = dax, y = o.y + 2, z = o.z },
        "air")
    box({ x = dbx, y = o.y + 1, z = o.z }, { x = dbx, y = o.y + 2, z = o.z },
        "air")
    return { pane = pane, partition_x = px,
             door_a = { x = dax, y = o.y + 1, z = o.z },
             door_b = { x = dbx, y = o.y + 1, z = o.z } }
end

-- A FURNACE WITH A LOSSLESS SLAB THROUGH THE MIDDLE OF IT -- the energy
-- measurement, and it is the sharpest one available because it needs no
-- new referee at all.
--
-- A sealed uniform-albedo uniformly-emissive box reads L = Le/(1-rho)
-- everywhere inside, and claude_furnace_check.py already scores that to
-- 0.3 %. Drop a LOSSLESS dielectric into it -- one that reflects R and
-- transmits 1-R and absorbs nothing -- and the equilibrium cannot move:
-- the slab neither adds energy nor removes it. So the pane material is
-- the only variable and the analytic answer is "no change", which is a
-- number this harness has been checking since August.
--
-- That is the honest form of the handoff's "furnace variant with a
-- transmissive SHELL". A transmissive shell would let the light out and
-- the room would stop being a furnace; a transmissive PARTITION leaves
-- it a furnace and puts the interface on every path.
--
-- The three arms:
--   air     the plain furnace-050, which must reproduce its own pin
--   opaque  the wall node -- two furnaces back to back, same answer
--   glass   the same room with a dielectric across it
--
-- in: { pos={x,y,z}, size=5, pane="glass" }
function OPS.glassfurnace(p)
    local o, s = p.pos, p.size or 5
    local wall = "claude_bridge:gray186_lit"
    local panes = {
        glass = { "mcl_core:glass", "default:glass" },
        water = { "mcl_core:water_source", "default:water_source" },
        opaque = { wall },
        air = { "air" },
    }
    local want = panes[p.pane or "glass"]
    local pane = nil
    for _, n in ipairs(want) do
        if n == "air" or core.registered_nodes[n] then pane = n break end
    end
    if not pane then return { error = "no node for pane " .. tostring(p.pane),
                              tried = want } end
    local e = s + 1
    box({ x = o.x, y = o.y, z = o.z },
        { x = o.x + e, y = o.y + e, z = o.z + e }, wall)
    box({ x = o.x + 1, y = o.y + 1, z = o.z + 1 },
        { x = o.x + s, y = o.y + s, z = o.z + s }, "air")
    -- the slab: one node thick, across the whole interior, at the
    -- half-depth plane so a camera at the south end looks THROUGH it at
    -- the north wall
    local pz = o.z + 1 + math.floor((s - 1) / 2)
    box({ x = o.x + 1, y = o.y + 1, z = pz },
        { x = o.x + s, y = o.y + s, z = pz }, pane)
    local dx = o.x + 1 + math.floor((s - 1) / 2)
    box({ x = dx, y = o.y + 1, z = o.z },
        { x = dx, y = o.y + 2, z = o.z }, "air")
    return { door = { x = dx, y = o.y + 1, z = o.z }, wall = wall,
             pane = pane, slab_z = pz }
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

-- Room integrity scan (2026-08-15, after a referee room was found dug
-- open). A referee room is only a referee while it is SEALED: one
-- missing node at (47,11,8) leaked daylight into the Cornell box for an
-- unknown period and invalidated a day of numbers before the strange
-- values gave it away. OPS.probe answers about a COLUMN; this answers
-- about a BOX, which is what a room is.
--
-- Returns every node name in the inclusive box, in a fixed x->y->z
-- order, so the caller can diff it against the builder's own spec and
-- name the exact offending positions; plus a cheap order-dependent
-- hash of that sequence, which is the room's identity as one number
-- (the formal room hash is roadmap 1b; this is its ancestor).
-- "unloaded" is reported as such and is never silently air.
function OPS.scan(p)
    local a, b = p.p1, p.p2
    local x0, x1 = math.min(a.x, b.x), math.max(a.x, b.x)
    local y0, y1 = math.min(a.y, b.y), math.max(a.y, b.y)
    local z0, z1 = math.min(a.z, b.z), math.max(a.z, b.z)
    local n = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
    if n > 20000 then error("scan too large: " .. n) end
    local names, counts, h = {}, {}, 2166136261
    for x = x0, x1 do for y = y0, y1 do for z = z0, z1 do
        local node = core.get_node_or_nil({ x = x, y = y, z = z })
        local name = node and node.name or "unloaded"
        names[#names + 1] = name
        counts[name] = (counts[name] or 0) + 1
        for i = 1, #name do
            h = (h + name:byte(i)) * 16777619 % 4294967296
        end
    end end end
    return { p1 = { x = x0, y = y0, z = z0 }, p2 = { x = x1, y = y1, z = z1 },
             total = n, counts = counts, hash = string.format("%08x", h),
             names = names }
end

-- /warp (roadmap 1b): the second way into every room, "reachable by
-- jump". The builtin /teleport is coordinates-only, so this reads
-- util/claude_vantages.json directly -- ONE SOURCE OF TRUTH, the same
-- file claude_ci.py and claude_lab.py already read -- rather than
-- keeping a second copy of the vantage list in the world. Not /tp: some
-- games (mineclonia) alias that chatcommand to their own teleport.
-- RUN_IN_PLACE=TRUE means the world lives at <repo>/worlds/<name>, so
-- worldpath/../../util is the repo's util dir on every seat this repo
-- scripts (environment-laws "Seat / build").
-- util/claude_vantages.json lives outside the world dir and outside
-- every mod dir, so the sandboxed `io` refuses it (checkPathWithGamedef
-- allows read/write under the world path and read-only under mod paths,
-- nothing else). claude_bridge is listed in secure.trusted_mods
-- (minetest.conf) for exactly this; request_insecure_environment() hands
-- back the pre-sandbox globals, whose `io` has no path check at all.
local claude_ie = core.request_insecure_environment()
local claude_io = claude_ie and claude_ie.io or io

local function claude_vantages_path()
    return core.get_worldpath() .. "/../../util/claude_vantages.json"
end

local function claude_load_vantages()
    local f = claude_io.open(claude_vantages_path(), "r")
    if not f then
        return nil, "cannot open claude_vantages.json (trusted_mods not "
                .. "set for claude_bridge? see minetest.conf secure.trusted_mods)"
    end
    local data = f:read("*a")
    f:close()
    local ok, parsed = pcall(core.parse_json, data)
    if not ok or type(parsed) ~= "table" then
        return nil, "cannot parse claude_vantages.json"
    end
    return parsed
end

local function claude_save_vantages(t)
    local f = claude_io.open(claude_vantages_path(), "w")
    if not f then return false, "cannot open claude_vantages.json for write" end
    f:write(core.write_json(t, true))
    f:close()
    return true
end

core.register_chatcommand("warp", {
    params = "<name> | next | prev | list | save <name>",
    description = "Jump to a saved CI vantage from claude_vantages.json, "
                  .. "or save the current pose as a new one.",
    func = function(playername, param)
        local player = core.get_player_by_name(playername)
        if not player then return false, "warp: not online" end
        local vs, err = claude_load_vantages()
        if not vs then return false, "warp: " .. tostring(err) end
        local names = {}
        for k in pairs(vs) do names[#names + 1] = k end
        table.sort(names)

        local args = {}
        for w in param:gmatch("%S+") do args[#args + 1] = w end
        local cmd = args[1]

        local function apply(vname)
            local v = vs[vname]
            if not v then
                return false, "warp: no such vantage: " .. tostring(vname)
            end
            -- time BEFORE the teleport (environment-laws: a sky change
            -- resets the accumulator; freeze/settle is the caller's job,
            -- this just applies the vantage's own recorded time).
            if v.time ~= nil then core.set_timeofday(v.time) end
            player:set_pos({ x = v.pos[1], y = v.pos[2], z = v.pos[3] })
            if v.yaw then player:set_look_horizontal(math.rad(v.yaw)) end
            if v.pitch then player:set_look_vertical(math.rad(-v.pitch)) end
            return true, "warped to " .. vname
                    .. (v.time ~= nil and (" (time " .. tostring(v.time) .. ")")
                        or "")
        end

        if not cmd or cmd == "list" then
            return true, "vantages (" .. #names .. "): "
                    .. table.concat(names, ", ")
        elseif cmd == "save" then
            local vname = args[2]
            if not vname then return false, "usage: /warp save <name>" end
            local pos = player:get_pos()
            vs[vname] = {
                set = "gallery",
                ci = true,
                pos = { pos.x, pos.y, pos.z },
                yaw = math.deg(player:get_look_horizontal()) % 360,
                pitch = -math.deg(player:get_look_vertical()),
                time = core.get_timeofday(),
                notes = "saved by /warp save (" .. playername .. ")",
            }
            local ok2, werr = claude_save_vantages(vs)
            if not ok2 then return false, "warp: " .. tostring(werr) end
            local msg = "saved vantage " .. vname .. " at ("
                    .. string.format("%.2f,%.2f,%.2f", pos.x, pos.y, pos.z)
                    .. ") -- stand still first, this IS the physics REST "
                    .. "position now on record"
            -- environment-laws: a moving sun resets the accumulator, so
            -- an unfrozen time_speed at save time means the RECORDED
            -- `time` is a snapshot of a value that was already drifting
            -- -- worth a warning, not a refusal (the position/look are
            -- still good, and freezing time is the caller's job at
            -- capture time regardless).
            local ts = tonumber(core.settings:get("time_speed") or "1")
            if ts and ts ~= 0 then
                msg = msg .. " -- WARNING: time_speed=" .. tostring(ts)
                        .. " (not frozen); the saved time=" .. tostring(vs[vname].time)
                        .. " was already drifting when this was recorded"
            end
            return true, msg
        elseif cmd == "next" or cmd == "prev" then
            if #names == 0 then return false, "warp: no vantages" end
            local meta = player:get_meta()
            local cur = meta:get_string("claude_warp_cur")
            local idx = 1
            for i, n in ipairs(names) do if n == cur then idx = i end end
            if cmd == "next" then idx = (idx % #names) + 1
            else idx = ((idx - 2) % #names) + 1 end
            local vname = names[idx]
            meta:set_string("claude_warp_cur", vname)
            return apply(vname)
        else
            player:get_meta():set_string("claude_warp_cur", cmd)
            return apply(cmd)
        end
    end,
})

-- Camera drift guard (2026-08-15). Turning the camera does NOT reset
-- the accumulator, so one stray mouse-look inside a 60 s settle blends
-- two views into one "converged" frame with a perfectly clean dial
-- state and nothing anywhere says so. OPS.player_state reports yaw but
-- not pitch; this reports BOTH plus position, so a capture can prove
-- the camera it was taken from. Degrees, and pitch is +up to match
-- OPS.tp / OPS.look and claude_vantages.json.
function OPS.aim(p)
    local pl = core.get_player_by_name(p.player or ADMIN)
    if not pl then error("not online: " .. tostring(p.player or ADMIN)) end
    local pos = pl:get_pos()
    return { pos = pos,
             yaw = math.deg(pl:get_look_horizontal()) % 360,
             pitch = -math.deg(pl:get_look_vertical()) }
end

-- Put the measurement seat back on its feet after a bad teleport
-- (2026-08-16). A harness that parks the camera by offsetting a vantage
-- can drop the player into a wall or off a roof, and a DEAD player is
-- not a recoverable state from this side of the channel: the death
-- formspec is client-side, `hp = 0` persists in the player file, and
-- rejoining does not clear it — a restarted client came back dead and
-- rendered "You died" over the capture. Mineclonia registers no heal
-- command, so there was no way back at all without this op.
--
-- Full HP and a teleport to a named position in one call, because those
-- are the two halves of "the seat is usable again". Idempotent.
-- AND THE DEATH SCREEN IS A SECOND, INDEPENDENT PROBLEM. Restoring hp
-- server-side does NOT take the "You died / Respawn" formspec off the
-- client: that dialog is client-side and closes when the client sends a
-- respawn, which a headless seat never does. Measured 2026-08-16: a
-- four-arm A/B ran to completion, printed stable numbers to four
-- decimals, and every one of its frames was a flat gray dialog over the
-- scene — `hp` read 20 the whole time. So respawn() FIRST (it is what
-- dismisses the dialog), then set_hp, then the teleport.
--
-- This is the "eyes on an actual frame before any number is believed"
-- law in its purest form: nothing in the stats, the dials or the conf
-- said anything was wrong.
function OPS.revive(p)
    local name = p.player or ADMIN
    local pl = core.get_player_by_name(name)
    if not pl then error("not online: " .. tostring(name)) end
    if pl:get_hp() <= 0 and pl.respawn then
        pl:respawn()
    end
    pl:set_hp(pl:get_properties().hp_max or 20, { type = "set_hp",
                                                  reason = "claude_bridge revive" })
    core.close_formspec(name, "")
    if p.pos then
        pl:set_pos(vector.new(p.pos.x, p.pos.y, p.pos.z))
    end
    return { hp = pl:get_hp(), pos = pl:get_pos() }
end

-- ABMs off/on, ASSERTED rather than set (2026-08-16, the "still the
-- world" step). claude_abm is a SERVER setting -- the grid, the
-- accumulator and the stats are all client-side, so putting it in the
-- client conf does nothing and looks exactly like the fix failing.
--
-- It is pushed at runtime and read back in the same call, because
-- spec/handoffs/2026-08-16-still-the-world-abm.md gate 2 asks for the
-- effective value, not for the write to have returned: the last session
-- found a stale claude_settings_patch.conf silently running
-- claude_grid_follow = 0 on a seat whose conf on disk said 1, and
-- grepping the conf was a blind instrument.
--
-- Runtime and NOT the conf file, deliberately. The dedicated server
-- returns from run_dedicated_server before main()'s
-- g_settings->updateConfigFile, so nothing set here is ever written to
-- minetest.conf -- which means a measurement run cannot leave a
-- gameplay seat with its world silently frozen. claude_ci turns it off
-- after seat start and back on in a finally, the same shape as the
-- doors and the lamps.
function OPS.abm(p)
    if p.on ~= nil then
        core.settings:set_bool("claude_abm", p.on and true or false)
    end
    return { claude_abm = core.settings:get_bool("claude_abm", true) }
end

-- Read any server setting back by name. The generic half of OPS.abm:
-- "what is this seat actually running" is a question the harness has
-- had to answer by grepping a conf file that the client rewrites on
-- exit (spec/environment-laws.md). Asking the server is the only
-- honest form of the question.
function OPS.setting(p)
    return { name = p.name, value = core.settings:get(p.name) }
end

-- Dig a node the way a player's hand does, not the way a mod does.
--
-- OPS.lamps and the room builders use core.set_node, which is the "mod
-- edit" path. The bug under test (roadmap: "world edits do not reach the
-- tracer while John drives") was reported for a HAND dig, and the whole
-- question was whether the two paths differ from the client's point of
-- view. core.node_dig runs the real thing: the protection check, the
-- node's on_dig / after_dig_node, the drops, and core.remove_node.
--
-- WHAT THIS CANNOT REPRODUCE, stated rather than glossed: a real hand
-- dig also runs the CLIENT-SIDE PREDICTION in game.cpp
-- (client->removeNode(nodepos) when node_dig_prediction is "air"). A
-- headless seat has no hand to press the mouse. It is the same
-- Client::removeNode that handleCommand_RemoveNode calls for the
-- server's packet, and both mark the same map block dirty, so the
-- prediction cannot be the difference -- but this op tests the server
-- half only, and the claim is scoped to that.
function OPS.dig(p)
    local pos = vector.new(p.x, p.y, p.z)
    local node = core.get_node(pos)
    if node.name == "air" or node.name == "ignore" then
        return { dug = false, was = node.name }
    end
    local digger = core.get_player_by_name(p.player or ADMIN)
    core.node_dig(pos, node, digger)
    return { dug = core.get_node(pos).name == "air", was = node.name,
             now = core.get_node(pos).name }
end

-- Everything about the player that can move a camera, in one answer.
--
-- Written 2026-08-16 after a Cornell capture came back framed ~3 degrees
-- differently from the golden at the SAME COMMIT, the same vantage, the
-- same room hash and the same server-reported pitch. The aim guard reads
-- pitch, yaw and position and nothing else, so a change in eye height,
-- physics override or player size is invisible to it -- and the client
-- owns the camera, which is the environment-laws trap in a new place.
-- properties.eye_height is the one a mod can move without touching a
-- single number the harness already prints.
function OPS.player_full(p)
    local name = p and p.player or ADMIN
    local pl = core.get_player_by_name(name)
    if not pl then error("not online: " .. tostring(name)) end
    local pr = pl:get_properties()
    local o1, o3 = pl:get_eye_offset()
    return {
        name = name,
        pos = pl:get_pos(),
        hp = pl:get_hp(),
        look_vertical = pl:get_look_vertical(),
        look_horizontal = pl:get_look_horizontal(),
        eye_height = pr.eye_height,
        collisionbox = pr.collisionbox,
        visual_size = pr.visual_size,
        eye_offset_first = o1,
        eye_offset_third = o3,
        physics = pl:get_physics_override(),
        attached = pl:get_attach() ~= nil,
        attributes = pl:get_meta():to_table().fields,
        timeofday = core.get_timeofday(),
    }
end
