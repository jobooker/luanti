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
        mcl_weather.change_weather("none")
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
