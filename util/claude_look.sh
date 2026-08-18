#!/bin/sh
# claude_look.sh — start a LOOK seat: the traced renderer, driveable by a human.
#
# This is NOT the measurement seat. `claude_ci` builds its own seat with
# input locked, HUD and chat hidden and photo-mode dials; this one is the
# opposite of that on purpose — you can walk, fly, type and change dials.
#
# Usage:   util/claude_look.sh            # real sky, NEE on (play settings)
#          util/claude_look.sh --photo    # NEE off: photo mode, the "truth"
#          util/claude_look.sh --stop     # shut the seat down
#
# Nothing here can leak into a measurement: claude_ci.pin_conf() forces
# PINNED_CONF (free_move, fast_move, screen size, fps caps...) and pushes
# CANONICAL_DIALS (input lock, HUD, chat, nee, descend...) at the start of
# every run, so a look seat's settings are overwritten before any capture.
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(dirname "$HERE")
cd "$REPO"

SERVER_PAT="bin/luantiserver --world worlds/gallery"
CLIENT_PAT="bin/luanti --address 127.0.0.1"

stop_seat() {
    pkill -f "$CLIENT_PAT" 2>/dev/null || true
    sleep 1
    pkill -f "$SERVER_PAT" 2>/dev/null || true
    sleep 1
    echo "seat stopped."
}

if [ "${1:-}" = "--stop" ]; then stop_seat; exit 0; fi

NEE=1
if [ "${1:-}" = "--photo" ]; then NEE=0; fi

# ONE SEAT AT A TIME. The bridge is single-slot (measured), and
# `bin/luanti --go` logs in as `claude`, which KICKS whoever is on that
# account -- including a CI run mid-capture. Refuse rather than evict.
if pgrep -f "$SERVER_PAT" >/dev/null 2>&1 || pgrep -f "$CLIENT_PAT" >/dev/null 2>&1; then
    echo "A seat is ALREADY RUNNING. If it is a CI run, leave it alone." >&2
    echo "If it is yours and you want it back: util/claude_look.sh --stop" >&2
    exit 1
fi

# The bridge mod is ASSEMBLED from fragments, and until 2026-08-17 nothing
# ever re-assembled it -- an edit to a fragment silently did nothing.
sh "$HERE/claude_seat_assemble.sh" >/dev/null

# Live dial channels persist across seat restarts and are applied on the
# client's FIRST poll, so a leftover line silently reconfigures a fresh
# seat. Clear both, exactly as claude_ci does.
: > claude_settings_patch.conf
: > worlds/gallery/claude_dial.conf

python3 - "$NEE" <<'PY'
import re, sys
nee = sys.argv[1]
look = {
    "claude_grid_debug": "3",     # 3 = traced (0 = plain raster)
    "claude_descend":    "1",     # stairs/beds are real sub-metre shapes
    "claude_rng":        "1",     # counter-based RNG (roadmap 1a)
    "claude_sky_uniform":"0",     # 0 = the REAL sky; >0 is the flat test sky
    "claude_view":       "0",     # 0 = photo; U cycles the debug views
    "claude_nee":        nee,
    "claude_stats":      "1",
    # A human seat is not a measurement seat:
    "claude_input_lock": "0",     # CI locks input; you need it unlocked
    "claude_show_chat":  "1",     # /warp and /dial are typed
    "claude_show_hud":   "1",
    "free_move":         "true",  # fly (K), reset by CI's pin_conf
    "fast_move":         "true",
    # time_speed 0 freezes the sun. With a sky now, a MOVING sun resets
    # accumulation every frame (roadmap 3c), so leave it frozen and move
    # time deliberately with /time.
    "time_speed":        "0",
}
s = open("minetest.conf").read()
for k, v in look.items():
    if re.search(r"(?m)^%s\s*=" % re.escape(k), s):
        s = re.sub(r"(?m)^%s\s*=.*$" % re.escape(k), "%s = %s" % (k, v), s)
    else:
        s += "%s = %s\n" % (k, v)
open("minetest.conf", "w").write(s)
PY

mkdir -p /tmp/claude_look
nohup ./bin/luantiserver --world worlds/gallery --port 30000 \
    > /tmp/claude_look/server.log 2>&1 &
sleep 8
nohup ./bin/luanti --address 127.0.0.1 --port 30000 --name claude --go \
    > /tmp/claude_look/client.log 2>&1 &
sleep 10

echo
echo "  LOOK SEAT UP.  nee=$NEE   logs: /tmp/claude_look/"
echo
echo "  /warp list          the saved viewpoints"
echo "  /warp cozy-night    the cabin  |  exterior-ci  outdoors"
echo "  /time 6000          noon   |  /time 18000  midnight"
echo "  /dial claude_nee 0  photo mode, live (~1 s to apply)"
echo "  K fly   J fast   U cycles debug views (says which, in words)"
echo
echo "  stop:  util/claude_look.sh --stop"
echo
[ -f claude_stats.json ] && cat claude_stats.json
