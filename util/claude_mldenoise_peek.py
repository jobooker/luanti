"""one frame of each real-time dump of a playtest run, side by side, to look at
(a leaked claude_view would show here)"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claude_judge as J  # noqa: E402

run = sys.argv[1]
meta = json.load(open(os.path.join(run, "meta.json")))
tiles = []
for name, sc in meta["scenarios"].items():
    for k in ("realtime", "reference"):
        frames, rows = J.load_dump(sc[k])
        f = frames[len(frames) // 2]
        print(name, k, len(frames), "frames; middle frame mean %.1f, std %.1f" % (f.mean(), f.std()))
        tiles.append(Image.fromarray(np.asarray(f)).resize((480, 270)))
sheet = Image.new("RGB", (960, 270 * (len(tiles) // 2)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 2) * 480, (i // 2) * 270))
out = os.path.join(run, "peek.jpg")
sheet.save(out)
print(out)
