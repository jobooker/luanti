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
