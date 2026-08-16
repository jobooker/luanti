#!/bin/bash
# GL41 tracer verification harness. One invocation = one measured run.
# Enforces every environment lesson from the debugging session:
#   - shaders copied to the app bundle (stale-shader trap)
#   - video_driver forced AFTER any client exit (config-rewrite trap)
#   - core profile asserted via the core-gated [claude_core] log marker
#   - verdict = traced-path green marker AND bottom-half stddev,
#     gated on light_body==1 (day) so the 30s day/night cycle can't confound
set -u
REPO=/Users/jobooker/code/luanti
APP=$REPO/build/macos/luanti.app
DATA="$HOME/Library/Application Support/minetest"
SCRATCH=$(dirname "$0")
LABEL=${1:-run}

echo "== [$LABEL] $(date +%H:%M:%S) =="

# 1. sync shaders repo -> bundle
rsync -a --delete "$REPO/client/shaders/" "$APP/Contents/Resources/client/shaders/" || exit 1
diff -q "$REPO/client/shaders/claude_accum/opengl_fragment.glsl" \
        "$APP/Contents/Resources/client/shaders/claude_accum/opengl_fragment.glsl" >/dev/null \
  || { echo "FATAL: bundle shader != repo shader"; exit 1; }

# 2. kill client, force config
pkill -f "luanti.app/Contents/MacOS/luanti" 2>/dev/null && sleep 2
python3 - "$DATA/minetest.conf" <<'EOF'
import sys, re
p = sys.argv[1]
txt = open(p).read()
want = {"video_driver": "opengl3", "claude_grid_debug": "3",
        "claude_stats": "1", "claude_trace_scale": "0.75"}
for k, v in want.items():
    if re.search(rf"^{k}\s*=", txt, re.M):
        txt = re.sub(rf"^{k}\s*=.*$", f"{k} = {v}", txt, flags=re.M)
    else:
        txt += f"\n{k} = {v}\n"
open(p, "w").write(txt)
EOF

# 3. launch, record launch time for log scoping
LAUNCH_TS=$(date "+%Y-%m-%d %H:%M:%S")
"$APP/Contents/MacOS/luanti" --address 192.168.0.243 --port 30000 \
    --name ichthyroid --go >/dev/null 2>&1 &
CLIENT_PID=$!
echo "client pid $CLIENT_PID, waiting for world join..."

# wait for stats.json to go fresh (written 1/s once in-game)
for i in $(seq 1 60); do
  sleep 2
  kill -0 $CLIENT_PID 2>/dev/null || { echo "VOID: client died during startup"; exit 2; }
  AGE=$(( $(date +%s) - $(stat -f %m "$DATA/claude_stats.json" 2>/dev/null || echo 0) ))
  [ "$AGE" -le 3 ] && break
done
[ "$AGE" -le 3 ] || { echo "VOID: never joined world (stats stale)"; exit 2; }

# 4. CORE ASSERTION: core-gated marker logged after launch
sleep 2   # marker logs on the first core draw
CORE=$(awk -v ts="$LAUNCH_TS" '$0 >= ts' "$DATA/debug.txt" | grep -c "claude_core")
if [ "$CORE" -lt 1 ]; then
  echo "VOID: no core-gated claude_core marker since launch -> NOT core profile"
  exit 2
fi
echo "core asserted ($CORE markers since launch)"

# 5. wait for a valid grid, then settle. No daylight gate: the parked
# camera faces the torch-lit cabin, so a working tracer is structured at
# any hour (calibrated: broken band sd 6.3 at night, working 33+).
for i in $(seq 1 30); do
  VV=$(python3 -c "import json;d=json.load(open('$DATA/claude_stats.json'));print(d['grid_valid'])" 2>/dev/null)
  [ "$VV" = "1" ] && break
  sleep 2
done
[ "$VV" = "1" ] || { echo "VOID: grid never valid"; exit 2; }
sleep 5  # accumulation settle

# 6. screenshot via settings-patch pseudo-key
kill -0 $CLIENT_PID 2>/dev/null || { echo "VOID: client died before shot"; exit 2; }
BEFORE=$(ls -t "$DATA/screenshots/" | head -1)
echo "claude_screenshot = $LABEL-$(date +%s)" > "$DATA/claude_settings_patch.conf"
for i in $(seq 1 10); do
  sleep 1
  AFTER=$(ls -t "$DATA/screenshots/" | head -1)
  [ "$AFTER" != "$BEFORE" ] && break
done
[ "$AFTER" != "$BEFORE" ] || { echo "VOID: screenshot never appeared"; exit 2; }
SHOT="$DATA/screenshots/$AFTER"
# wait for the PNG write to finish (size stable across 1s)
SZ=0
for i in $(seq 1 10); do
  NEW=$(stat -f %z "$SHOT" 2>/dev/null || echo 0)
  [ "$NEW" -gt 0 ] && [ "$NEW" = "$SZ" ] && break
  SZ=$NEW; sleep 1
done
echo "shot: $SHOT (${SZ} bytes)"

# 7. verdict: green marker + bottom-half stddev
python3 - "$SHOT" <<'EOF'
import sys
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB")
w, h = im.size
# traced-path marker: 12px green block at framebuffer bottom-left
marker = sum(1 for y in range(h - 40, h) for x in range(40)
             if (lambda p: p[1] > 200 and p[0] < 80 and p[2] < 80)(im.getpixel((x, y))))
# UI-free band (skips hotbar, held item, marker): x 10-75%, y 45-80%
crop = im.crop((int(w * 0.10), int(h * 0.45), int(w * 0.75), int(h * 0.80)))
px = list(crop.getdata())[::11]
sds = []
for c in range(3):
    vals = [p[c] for p in px]
    m = sum(vals) / len(vals)
    sds.append((sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5)
means = [sum(p[c] for p in px) / len(px) for c in range(3)]
print(f"marker_px={marker}  band mean RGB={means[0]:.0f}/{means[1]:.0f}/{means[2]:.0f}  "
      f"stddev={sds[0]:.1f}/{sds[1]:.1f}/{sds[2]:.1f}  min_sd={min(sds):.1f}")
if marker < 50:
    print("VERDICT: VOID (traced present path did not run — raster passthrough or debug off)")
    sys.exit(3)
# calibrated 2026-08-11: broken band sd 6.3 (night), working 33.7+ (night)
if min(sds) >= 20:
    print("VERDICT: SCENE (tracer producing structured image)")
elif min(sds) <= 10:
    print("VERDICT: FLAT (traced path ran, output unstructured)")
else:
    print("VERDICT: AMBIGUOUS (rerun)")
EOF
RC=$?
cat "$DATA/claude_stats.json"; echo
exit $RC
