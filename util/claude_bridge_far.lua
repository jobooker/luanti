-- Far-data feed ops for claude_bridge (2026-08-12). APPEND these two
-- functions to /opt/luanti/config/mods/claude_bridge/init.lua on the
-- beelink (before the register_globalstep block), then restart the
-- server. Orchestrated by util/claude_far_fetch.py from the client Mac.

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
