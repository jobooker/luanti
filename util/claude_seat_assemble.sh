#!/bin/sh
# Assemble the claude_bridge mod for a local seat from its fragments.
#
# The bridge mod is one Lua file built from pieces that live in this
# repo: a base (the RPC channel + generic world ops, upstream in
# jobooker/luanti-server) plus the scene-builder and gallery op
# fragments. They append because OPS is a file-local table and every
# fragment only adds functions to it.
#
# Usage: util/claude_seat_assemble.sh [<base-init.lua>]
# Writes to "$CLAUDE_MT_DIR"/mods/claude_bridge/init.lua. Defaults to the
# REPO ROOT, matching the RUN_IN_PLACE=TRUE seat convention that
# claude_lab.py also assumes. Keep a pristine copy of the base as
# init.base.lua so re-assembly is idempotent.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
MT=${CLAUDE_MT_DIR:-$(dirname "$HERE")}
MOD="$MT/mods/claude_bridge"
BASE=${1:-$MOD/init.base.lua}

[ -d "$MOD" ] || { echo "no bridge mod at $MOD" >&2; exit 1; }

# First run: the installed init.lua IS the base; preserve it.
if [ ! -f "$MOD/init.base.lua" ] && [ "$BASE" = "$MOD/init.base.lua" ]; then
    cp "$MOD/init.lua" "$MOD/init.base.lua"
fi

# KNOWN-BAD LINE IN THE BASE (fixed in-repo by e6260e8d1, but that fix
# landed in util/claude_bridge_far.lua while the deployed base kept the
# bug): mineclonia's change_weather logs debug.getinfo(2).name, which is
# nil inside an anonymous globalstep, and concatenating nil takes the
# whole server down on the first weather roll (~15-30 min in). Patch it
# here so any seat assembled from an old base is immune.
patch_weather() {
    sed 's/mcl_weather\.change_weather("none")/mcl_weather.change_weather("none", nil, "claude_bridge")/'
}

if grep -q 'change_weather("none")' "$BASE"; then
    echo "patching base: weather lock changer name (mineclonia crash)"
fi

{
    cat "$BASE" | patch_weather
    echo
    echo "-- ==== appended: util/claude_bridge_scene.lua ===="
    cat "$HERE/claude_bridge_scene.lua"
    echo
    echo "-- ==== appended: util/claude_bridge_gallery.lua ===="
    cat "$HERE/claude_bridge_gallery.lua"
} > "$MOD/init.lua.new"

mv "$MOD/init.lua.new" "$MOD/init.lua"

# The gallery's uniform-albedo node textures are generated, not stored —
# they are what makes the furnace/Cornell albedo exact by construction.
python3 "$HERE/claude_gallery_textures.py" "$MOD/textures" >/dev/null

echo "assembled $MOD/init.lua"
grep -c '^function OPS\.' "$MOD/init.lua" | sed 's/^/ops: /'
ls "$MOD/textures" | wc -l | sed 's/^/textures: /'
