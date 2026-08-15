-- claude_bridge: live co-admin channel over world-dir file RPC.
--
-- Request:  <worldpath>/claude_cmd.json   {"id": "<unique>", "op": "...", ...}
-- Response: <worldpath>/claude_out.json   {"id": ..., "ok": true, "result": ...}
--                                      or {"id": ..., "ok": false, "error": "..."}
-- A request is processed once per unique id (overwrite the file with a fresh
-- id for each call). Allowlisted ops only. Kill switch: set
-- load_mod_claude_bridge = false in world.mt.

local worldpath = core.get_worldpath()
local CMD = worldpath .. "/claude_cmd.json"
local OUT = worldpath .. "/claude_out.json"
local ADMIN = core.settings:get("name") or "ichthyroid"

local last_id = nil
local timer = 0

local function respond(id, ok, payload)
    local out = { id = id, ok = ok }
    if ok then out.result = payload else out.error = tostring(payload) end
    core.safe_file_write(OUT, core.write_json(out))
end

local OPS = {}

function OPS.ping()
    return { pong = true, version = core.get_version().string }
end

function OPS.players()
    local list = {}
    for _, pl in ipairs(core.get_connected_players()) do
        list[#list + 1] = {
            name = pl:get_player_name(),
            pos = pl:get_pos(),
            hp = pl:get_hp(),
        }
    end
    return list
end

function OPS.player_state(p)
    local pl = core.get_player_by_name(p.name or ADMIN)
    if not pl then error("not online: " .. tostring(p.name or ADMIN)) end
    return {
        pos = pl:get_pos(),
        hp = pl:get_hp(),
        look_yaw = pl:get_look_horizontal(),
        timeofday = core.get_timeofday(),
    }
end

-- {"op":"chat","msg":"hello"} broadcast; add "to":"<player>" for a whisper
function OPS.chat(p)
    local msg = "[claude] " .. (p.msg or "")
    if p.to then
        if not core.get_player_by_name(p.to) then error("not online: " .. p.to) end
        core.chat_send_player(p.to, msg)
    else
        core.chat_send_all(msg)
    end
    return true
end

-- Run a registered chatcommand, acting as the admin account by default.
function OPS.cmd(p)
    local def = core.registered_chatcommands[p.command or ""]
    if not def then error("unknown command: " .. tostring(p.command)) end
    local ok, msg = def.func(p.as or ADMIN, p.param or "")
    return { ok = ok, msg = msg }
end

function OPS.time(p)
    if p.set then core.set_timeofday(p.set) end
    return core.get_timeofday()
end

-- {"op":"look","yaw":90,"pitch":-30} degrees; also accepts "player" (default
-- ADMIN). Server-side camera aim so Claude can frame screenshot tours.
-- Yaw: 0 = +Z north, 90 = -X west (Minetest convention). Pitch: +up/-down.
function OPS.look(p)
    local player = core.get_player_by_name(p.player or ADMIN)
    if not player then error("not online: " .. (p.player or ADMIN)) end
    if p.yaw then player:set_look_horizontal(math.rad(p.yaw)) end
    if p.pitch then player:set_look_vertical(math.rad(-(p.pitch))) end
    return { yaw = math.deg(player:get_look_horizontal()),
             pitch = -math.deg(player:get_look_vertical()) }
end

-- {"op":"replace","p1":{...},"p2":{...},"from":"mcl_core:snow","to":"air"}
-- name-targeted swap in an inclusive box; leaves everything else alone
function OPS.replace(p)
    local p1 = vector.new(p.p1.x, p.p1.y, p.p1.z)
    local p2 = vector.new(p.p2.x, p.p2.y, p.p2.z)
    local nodes = core.find_nodes_in_area(p1, p2, p["from"])
    if #nodes > 60000 then error("replace too large") end
    for _, pos in ipairs(nodes) do
        core.set_node(pos, { name = p.to })
    end
    return { replaced = #nodes }
end

-- {"op":"fill","p1":{...},"p2":{...},"name":"mcl_core:stone"} inclusive box
-- fill; used to build deterministic diagnostic scenes for render testing.
function OPS.fill(p)
    local a, b = p.p1, p.p2
    local n = { name = p.name }
    local count = 0
    for x = math.min(a.x, b.x), math.max(a.x, b.x) do
    for y = math.min(a.y, b.y), math.max(a.y, b.y) do
    for z = math.min(a.z, b.z), math.max(a.z, b.z) do
        core.set_node({ x = x, y = y, z = z }, n)
        count = count + 1
        if count > 60000 then error("fill too large") end
    end end end
    return { filled = count }
end

function OPS.get_node(p)
    return core.get_node_or_nil(p.pos) or "unloaded"
end

function OPS.set_node(p)
    core.set_node(p.pos, { name = p.name })
    return true
end

function OPS.stats()
    return {
        uptime_s = core.get_server_uptime(),
        max_lag_s = core.get_server_max_lag(),
        players = #core.get_connected_players(),
        timeofday = core.get_timeofday(),
    }
end

-- /dial <key> [value]: console-native tuning, no new permissions anywhere
-- — it only ever touches files this mod already owns inside the world
-- dir. <key> gets "claude_" prefixed unless already present. With a
-- value, merges "key = value" into <worldpath>/claude_dial.conf
-- (replacing that key's existing line, or appending). The client polls
-- this file at ~1 Hz via the claude_dial_file setting (same parse/apply
-- path as claude_settings_patch.conf) and applies it live. With no
-- value, reports the key's current line from the file, or "not set".
local DIAL = worldpath .. "/claude_dial.conf"

local function dial_read()
    local f = io.open(DIAL, "r")
    if not f then return "" end
    local content = f:read("*a") or ""
    f:close()
    return content
end

local function dial_key_of(line)
    return line:match("^%s*([%w_]+)%s*=")
end

core.register_chatcommand("dial", {
    params = "<key> [value]",
    description = "Read or set a claude_* client dial via claude_dial.conf "
        .. "(the console->client channel; client applies within ~1s)",
    func = function(_name, param)
        local key, value = param:match("^(%S+)%s+(.+)$")
        if not key then key = param:match("^(%S+)$") end
        if not key or key == "" then
            return false, "usage: /dial <key> [value]"
        end
        if not key:match("^claude_") then
            key = "claude_" .. key
        end

        local content = dial_read()
        if not value then
            for line in content:gmatch("[^\n]+") do
                if dial_key_of(line) == key then
                    return true, line
                end
            end
            return true, key .. ": not set"
        end

        local newline = key .. " = " .. value
        local lines = {}
        local found = false
        for line in content:gmatch("[^\n]+") do
            if dial_key_of(line) == key then
                lines[#lines + 1] = newline
                found = true
            else
                lines[#lines + 1] = line
            end
        end
        if not found then
            if #lines == 0 then
                lines[#lines + 1] =
                    "# claude_dial.conf: console-set client dials (/dial),"
                lines[#lines + 1] =
                    "# merged in by the client's claude_dial_file poll (~1 Hz)"
            end
            lines[#lines + 1] = newline
        end
        core.safe_file_write(DIAL, table.concat(lines, "\n") .. "\n")
        return true, newline .. " (client applies within ~1s)"
    end,
})

core.register_globalstep(function(dtime)
    timer = timer + dtime
    if timer < 0.5 then return end
    timer = 0
    local f = io.open(CMD, "r")
    if not f then return end
    local raw = f:read("*a")
    f:close()
    local req = raw and core.parse_json(raw)
    if not req or not req.id or req.id == last_id then return end
    last_id = req.id
    local op = OPS[req.op or ""]
    if not op then
        respond(req.id, false, "unknown op: " .. tostring(req.op))
        return
    end
    local ok, res = pcall(op, req)
    respond(req.id, ok, res)
end)

core.log("action", "[claude_bridge] active, polling " .. CMD)
-- Far-data feed ops for claude_bridge (2026-08-12). APPEND these two
-- functions to /opt/luanti/config/mods/claude_bridge/init.lua on the
-- beelink (before the register_globalstep block), then restart the
-- server. Orchestrated by util/claude_far_fetch.py from the client Mac.

-- Weather lock (John, 2026-08-12): keep the sky clear until the art
-- pass gets to weather. Checks every 60 s; delete this block to give
-- the weather cycle back.
local weather_timer = 0
core.register_globalstep(function(dtime)
    weather_timer = weather_timer + dtime
    if weather_timer < 60 then return end
    weather_timer = 0
    if mcl_weather and mcl_weather.state and mcl_weather.state ~= "none" then
        mcl_weather.change_weather("none", nil, "claude_bridge")
    end
end)

-- Kick an async emerge so ungenerated terrain exists before sampling.
-- Returns immediately; poll with sample_columns (ungenerated blocks
-- simply come back empty until the emerge lands).
function OPS.emerge_region(p)
    core.emerge_area(p.p1, p.p2)
    return { started = true }
end

-- Sample block columns into per-block 4^3-subcell summaries:
-- in : { cols = {{bx,bz},...}, ybot = <block y>, ytop = <block y> }
-- out: { blocks = {{ p={bx,by,bz},
--                    sub={{idx, occ, water, "node:name", param2},...}},...}}
-- Skips the same decoration drawtypes the client fold skips. Dominant
-- node name per subcell carries the color; param2 carries biome tint.
function OPS.sample_columns(p)
    local defmemo = {}
    local function classify(cid)
        local m = defmemo[cid]
        if m == nil then
            local name = core.get_name_from_content_id(cid)
            local def = core.registered_nodes[name]
            if not def then
                m = false
            else
                local dt = def.drawtype
                local skip = (def.light_source or 0) == 0
                    and (dt == "plantlike" or dt == "plantlike_rooted"
                        or dt == "firelike" or dt == "signlike"
                        or dt == "raillike" or dt == "torchlike")
                -- thin leveled layers (snow) are air, not a +1m wall
                if def.paramtype2 == "leveled" then skip = true end
                if not skip and (def.light_source or 0) > 0
                        and dt == "airlike" then
                    skip = true
                end
                if skip then
                    m = false
                else
                    m = { name = name,
                          water = def.liquidtype ~= nil
                              and def.liquidtype ~= "none" }
                end
            end
            defmemo[cid] = m
        end
        return m
    end

    local out = {}
    for _, col in ipairs(p.cols) do
        local bx, bz = col[1], col[2]
        for by = p.ytop, p.ybot, -1 do
            local minp = { x = bx * 16, y = by * 16, z = bz * 16 }
            local maxp = { x = minp.x + 15, y = minp.y + 15, z = minp.z + 15 }
            local vm = core.get_voxel_manip(minp, maxp)
            local emin, emax = vm:get_emerged_area()
            local data = vm:get_data()
            local p2d = vm:get_param2_data()
            local va = VoxelArea:new{ MinEdge = emin, MaxEdge = emax }
            local subs = {}
            local any = false
            for z = 0, 15 do
                for y = 0, 15 do
                    local base = va:index(minp.x, minp.y + y, minp.z + z)
                    for x = 0, 15 do
                        local cid = data[base + x]
                        if cid ~= core.CONTENT_AIR
                                and cid ~= core.CONTENT_IGNORE then
                            local m = classify(cid)
                            if m then
                                local sub = math.floor(z / 4) * 16
                                    + math.floor(y / 4) * 4
                                    + math.floor(x / 4)
                                local s = subs[sub]
                                if not s then
                                    s = { o = 0, w = 0, names = {}, p2 = 0 }
                                    subs[sub] = s
                                end
                                if m.water then
                                    s.w = s.w + 1
                                else
                                    s.o = s.o + 1
                                end
                                s.names[m.name] = (s.names[m.name] or 0) + 1
                                if s.p2 == 0 then
                                    s.p2 = p2d[base + x]
                                end
                                any = true
                            end
                        end
                    end
                end
            end
            if any then
                local sublist = {}
                for idx, s in pairs(subs) do
                    local best, bn = "", -1
                    for n, c in pairs(s.names) do
                        if c > bn then best, bn = n, c end
                    end
                    sublist[#sublist + 1] = { idx, s.o, s.w, best, s.p2 }
                end
                out[#out + 1] = { p = { bx, by, bz }, sub = sublist }
            end
        end
    end
    return { blocks = out }
end

-- ==== appended: util/claude_bridge_scene.lua ====
-- Scene-builder ops for the claude_bridge (2026-08-12): John's lighting
-- test world — "an Everest, a cave, a log cabin with lights right next
-- to me, and a lighthouse far away." APPEND to the dev bridge.

-- One y-slab of a noisy cone mountain per call (VoxelManip bulk).
-- in: { center={x,z}, radius=n, height=n, ybase=n, y0=n, y1=n, seed=n }
function OPS.mountain(p)
    local cx, cz = p.center.x, p.center.z
    local r, h, yb = p.radius, p.height, p.ybase
    local minp = { x = cx - r, y = p.y0, z = cz - r }
    local maxp = { x = cx + r, y = p.y1, z = cz + r }
    local vm = core.get_voxel_manip(minp, maxp)
    local emin, emax = vm:get_emerged_area()
    local data = vm:get_data()
    local va = VoxelArea:new{ MinEdge = emin, MaxEdge = emax }
    local c_stone = core.get_content_id("default:stone")
    local c_snow = core.get_content_id("default:snowblock")
    local c_air = core.CONTENT_AIR
    local n = 0
    for z = minp.z, maxp.z do
        for x = minp.x, maxp.x do
            local dx, dz = x - cx, z - cz
            local d = math.sqrt(dx * dx + dz * dz)
            if d < r then
                -- ridged cone: two noise octaves carve spurs and gullies
                local a = math.atan2(dz, dx)
                local ridge = 0.72 + 0.18 * math.abs(math.sin(a * 5 + d * 0.02))
                        + 0.10 * math.sin(x * 0.11) * math.cos(z * 0.13)
                local surf = yb + h * ((1 - d / r) ^ 1.45) * ridge
                local snowline = yb + h * 0.45
                for y = p.y0, p.y1 do
                    if y < surf then
                        local vi = va:index(x, y, z)
                        if data[vi] == c_air then
                            data[vi] = (y > snowline and surf - y < 6)
                                    and c_snow or c_stone
                            n = n + 1
                        end
                    end
                end
            end
        end
    end
    vm:set_data(data)
    vm:write_to_map()
    return { placed = n }
end

-- Log cabin with warm interior light, door gap south.
-- in: { pos = {x,y,z} } (floor corner)
function OPS.cabin(p)
    local o = p.pos
    local W, D, H = 9, 7, 4
    local function set(x, y, z, name)
        core.set_node({ x = o.x + x, y = o.y + y, z = o.z + z },
                { name = name })
    end
    for x = 0, W - 1 do for z = 0, D - 1 do
        set(x, 0, z, "default:wood")               -- floor
        set(x, H, z, "default:wood")               -- flat roof
        for y = 1, H - 1 do
            local wall = (x == 0 or x == W - 1 or z == 0 or z == D - 1)
            if wall then
                local door = (z == 0 and (x == 4 or x == 5) and y <= 2)
                local win = (y == 2 and ((x == 2 or x == W - 3)
                        and (z == 0 or z == D - 1)
                        or (z == 3 and (x == 0 or x == W - 1))))
                if door then set(x, y, z, "air")
                elseif win then set(x, y, z, "default:glass")
                else set(x, y, z, "default:tree") end
            else
                for y2 = 1, H - 1 do set(x, y2, z, "air") end
            end
        end
    end end
    -- warm lights: two mese lamps in the ceiling corners + torch wall
    set(2, H - 1, 2, "default:meselamp")
    set(W - 3, H - 1, D - 3, "default:meselamp")
    return { ok = true }
end

-- Lighthouse: banded tower + glowing lamp room.
-- in: { pos = {x,z}, ybase = n, height = n }
function OPS.lighthouse(p)
    local cx, cz, yb = p.pos.x, p.pos.z, p.ybase
    local h = p.height or 36
    local function set(x, y, z, name)
        core.set_node({ x = x, y = y, z = z }, { name = name })
    end
    for y = yb, yb + h do
        local band = (math.floor((y - yb) / 4) % 2 == 0)
                and "default:silver_sandstone_block" or "default:brick"
        for dx = -2, 2 do for dz = -2, 2 do
            local edge = math.abs(dx) == 2 or math.abs(dz) == 2
            local corner = math.abs(dx) == 2 and math.abs(dz) == 2
            if edge and not corner then
                set(cx + dx, y, cz + dz, band)
            elseif not edge then
                set(cx + dx, y, cz + dz, "air")
            end
        end end
    end
    -- lamp room: glass walls, mese lamp core, roof
    for y = yb + h + 1, yb + h + 3 do
        for dx = -2, 2 do for dz = -2, 2 do
            local edge = math.abs(dx) == 2 or math.abs(dz) == 2
            if y == yb + h + 3 then
                set(cx + dx, y, cz + dz, "default:stone_block")
            elseif edge then
                set(cx + dx, y, cz + dz, "default:glass")
            else
                set(cx + dx, y, cz + dz, "air")
            end
        end end
        if y < yb + h + 3 then
            set(cx, y, cz, "default:meselamp")
        end
    end
    return { ok = true }
end

-- Cozy interior — the acceptance scene for the interior lighting
-- program. One room, four emitter classes at distinct intensities
-- (campfire, floor lantern, wall torches, lit furnace), one
-- deliberately dark sleeping corner (NE) with no emitter within 4 m,
-- high-frequency albedo clutter (bookshelves, chests, carpet, hearth
-- stone), and windows placed for raking sun: south pair for midday,
-- west slot for sunset across the hearth. Gable roof = cathedral
-- ceiling, so indirect light has real volume to work in.
-- Node names resolve Mineclonia-first with minetest_game fallbacks;
-- unresolvable names are skipped and listed in the return value.
-- NOTE for the tracer: campfire/lantern/torch/chest are non-full-cube
-- nodes — how the classifier folds them (glowing cube vs empty-with-
-- light) is part of what this scene is built to expose.
-- in: { pos = {x,y,z} } (floor corner, south-west)
function OPS.cozy(p)
    local o = p.pos
    local W, D = 11, 9  -- x 0..10, z 0..8; walls y 1..4, ridge y 9
    local missing, placed = {}, 0
    local function R(...)
        for _, n in ipairs({ ... }) do
            if core.registered_nodes[n] then return n end
        end
        missing[#missing + 1] = (select(1, ...))
        return nil
    end
    local function set(x, y, z, name, param2)
        if not name then return end
        core.set_node({ x = o.x + x, y = o.y + y, z = o.z + z },
                { name = name, param2 = param2 })
        placed = placed + 1
    end
    local planks = R("mcl_trees:wood_oak", "mcl_core:wood", "default:wood")
    local wallwd = R("mcl_trees:wood_spruce", "mcl_core:sprucewood",
            "default:junglewood") or planks
    local log = R("mcl_trees:tree_oak", "mcl_core:tree", "default:tree")
    local glass = R("mcl_core:glass", "default:glass")
    local cobble = R("mcl_core:cobble", "default:cobble")
    local stair = R("mcl_stairs:stair_oak", "mcl_stairs:stair_wood_oak",
            "stairs:stair_wood") or planks
    local furnace = R("mcl_furnaces:furnace_active", "mcl_furnaces:furnace",
            "default:furnace")
    local campfire = R("mcl_campfires:campfire_lit")
    local lantern = R("mcl_lanterns:lantern_floor", "default:meselamp")
    local torch_w = R("mcl_torches:torch_wall", "default:torch_wall")
    local chest = R("mcl_chests:chest_small", "mcl_chests:chest",
            "default:chest")
    local shelf = R("mcl_books:bookshelf", "default:bookshelf")
    local craft = R("mcl_crafting_table:crafting_table")
    local carpet = R("mcl_wool:white_carpet")
    local bed_b = R("mcl_beds:bed_red_bottom", "beds:bed_bottom")
    local bed_t = R("mcl_beds:bed_red_top", "beds:bed_top")
    local pot = R("mcl_flowerpots:flower_pot")

    -- shell: floor, walls (log corners, plank fill), window + door gaps
    for x = 0, W - 1 do for z = 0, D - 1 do
        set(x, 0, z, planks)
        for y = 1, 4 do
            local wall = (x == 0 or x == W - 1 or z == 0 or z == D - 1)
            if wall then
                local corner = (x == 0 or x == W - 1)
                        and (z == 0 or z == D - 1)
                local door = (z == 0 and x == 5 and y <= 2)
                local win = (z == 0 and (x == 3 or x == 7)
                                and (y == 2 or y == 3))
                        or (x == 0 and z >= 3 and z <= 5 and y == 2)
                        or (x == W - 1 and z == 2 and y == 2)
                if door then set(x, y, z, "air")
                elseif win then set(x, y, z, glass)
                elseif corner then set(x, y, z, log)
                else set(x, y, z, wallwd) end
            else
                set(x, y, z, "air")
            end
        end
    end end
    -- gable roof: stair rows climbing from both eaves, plank ridge,
    -- plank gable ends; interior below stays open (cathedral ceiling)
    local face_n = core.dir_to_facedir({ x = 0, y = 0, z = 1 })
    local face_s = core.dir_to_facedir({ x = 0, y = 0, z = -1 })
    for s = 0, 3 do
        local y = 5 + s
        for x = 0, W - 1 do
            set(x, y, s, stair, face_n)
            set(x, y, D - 1 - s, stair, face_s)
        end
        for z = s + 1, D - 2 - s do
            set(0, y, z, wallwd)
            set(W - 1, y, z, wallwd)
            for x = 1, W - 2 do set(x, y, z, "air") end
        end
    end
    for x = 0, W - 1 do set(x, 9, 4, planks) end

    -- hearth zone (west): stone pad, campfire, lit furnace facing east
    for x = 1, 3 do for z = 3, 5 do set(x, 0, z, cobble) end end
    set(2, 1, 4, campfire)
    set(1, 1, 6, furnace, core.dir_to_facedir({ x = 1, y = 0, z = 0 }))
    -- torch pair on the east wall (attached to +X)
    local wm_e = core.dir_to_wallmounted({ x = 1, y = 0, z = 0 })
    set(9, 3, 3, torch_w, wm_e)
    set(9, 3, 5, torch_w, wm_e)
    -- floor lantern by the door
    set(4, 1, 1, lantern)
    -- dark corner (NE): bed + chests, no emitter within 4 m
    set(8, 1, 6, bed_b, face_n)
    set(8, 1, 7, bed_t, face_n)
    set(9, 1, 7, chest)
    set(9, 1, 1, chest)
    -- clutter: bookshelf wall (N), crafting table + pot at the S window
    for x = 3, 5 do for y = 1, 2 do set(x, y, 7, shelf) end end
    set(6, 1, 1, craft)
    set(7, 1, 1, pot)
    -- white carpet patch center — a bounce brightener on the floor
    if carpet then
        for x = 4, 6 do for z = 3, 5 do set(x, 1, z, carpet) end end
    end
    return { placed = placed, missing = missing }
end

-- Cave: chain of overlapping air spheres into a hillside, torches.
-- in: { start = {x,y,z}, dir = {x,z}, length = n }
function OPS.cave(p)
    local x, y, z = p.start.x, p.start.y, p.start.z
    local dx, dz = p.dir.x, p.dir.z
    local L = p.length or 40
    local n = 0
    for i = 0, L, 3 do
        local r = 3 + math.floor(math.sin(i * 0.4) * 1.5 + 1.5)
        local cxx = math.floor(x + dx * i)
        local czz = math.floor(z + dz * i)
        local cy = y - math.floor(i * 0.15)
        for ox = -r, r do for oy = -r, r do for oz = -r, r do
            if ox * ox + oy * oy + oz * oz <= r * r then
                core.set_node({ x = cxx + ox, y = cy + oy, z = czz + oz },
                        { name = "air" })
                n = n + 1
            end
        end end end
        if i % 9 == 0 and i > 0 then
            core.set_node({ x = cxx, y = cy - r + 1, z = czz },
                    { name = "default:torch" })
        end
    end
    return { carved = n }
end

-- ==== appended: util/claude_bridge_gallery.lua ====
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
