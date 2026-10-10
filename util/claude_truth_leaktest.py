"""claude_truth_leaktest -- truth mode must be immune to the display dials
(DECISIONS 0y, 2026-10-09).

One pose in the doorway room, 256 frames, each arm arriving from another
pose so it starts fresh. Arms: truth mode at defaults twice (the
renderer's own run-to-run spread: it is NOT bit-repeatable at one seed,
0.003 pixel RMS here), truth mode with every display dial on, and the same
dials outside truth mode (a real change, which must be far larger, or the
test proves nothing). Run from the engine checkout: python3 util/claude_truth_leaktest.py"""
import glob, hashlib, json, os, subprocess, sys
import numpy as np
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(WT); sys.path.insert(0, "util")
import claude_gpu_lock
claude_gpu_lock.hold("truth mode leak test")
import claude_truth_store as TS
pin = "/tmp/leak_pin.txt"
open(pin, "w").write("0 241.7 8.5 231.7 315 -12\n")      # the doorway room: bounce light dominates
# each arm starts at another pose and moves to the test pose, so its frame
# is dumped at exactly 256 frames from a fresh start (2026-10-09: back to
# back at one pose, the second arm kept counting from the first: frame 512+)
two = "/tmp/leak_two.txt"
open(two, "w").write("0 241.7 8.5 229.7 135 -12\n1 241.7 8.5 231.7 315 -12\n")
ON = ["claude_denoise=1", "claude_denoise_learned=2", "claude_ledger=5", "claude_boost=1", "claude_raw_frame=1",
      "claude_reproject=1", "claude_bounces=4", "claude_rng=0", "claude_split=16"]
DEF = ["claude_denoise=1", "claude_denoise_learned=2", "claude_ledger=0", "claude_boost=0", "claude_raw_frame=0",
       "claude_reproject=1", "claude_bounces=24", "claude_rng=2", "claude_split=0"]
EYE0 = ["claude_white_balance=0", "claude_night_vision=0"]
arms = [("def-1", ["claude_truth=1"] + DEF + EYE0), ("def-2", ["claude_truth=1"] + DEF + EYE0),
        ("all-on", ["claude_truth=1"] + ON + EYE0), ("play-all-on", ["claude_truth=0"] + ON + EYE0)]
out = {}
# warm up like the playtest: a fresh seat loads far terrain for ~60 s at a
# low frame rate, and 256 held frames then miss claude_motion's budget
subprocess.run(["python3", "util/claude_motion.py", "--play", "--path", pin, "--nodump", "--frames", "120",
                "--name", "leak-warmup"], capture_output=True, text=True)
import time
time.sleep(60)
first = False
for name, dials in arms:
    cmd = ["python3", "util/claude_motion.py", "--play", "--path", two, "--frames", "2", "--scale", "2",
           "--name", "leak-" + name, "--dial", "claude_path_hold=256", "--dial", "claude_auto_exposure=0",
           "--dial", "claude_exposure=1.9", "--dial", "claude_rng_seed=0"] + sum([["--dial", d] for d in dials], [])
    if not first:
        cmd.append("--skip-seat")
    res = subprocess.run(cmd, capture_output=True, text=True)
    r = res.stdout.strip().splitlines()
    d = r[-1] if r else ""
    if not os.path.isdir(d):
        print(name, "NO DUMP. stdout:", r[-6:], "stderr:", res.stderr.strip().splitlines()[-6:], flush=True)
        sys.exit(1)
    rows = TS._rows(d)
    raw = open(os.path.join(d, "%05d.rgba" % rows[-1]["i"]), "rb").read()
    out[name] = (hashlib.sha256(raw).hexdigest()[:16], rows[-1].get("features"), TS._frame(d, rows[-1]).mean(),
                 "still_frames %s path_frame %s" % (rows[-1]["still_frames"], rows[-1]["path_frame"]))
    print(name, out[name], flush=True)
FR = {}
for name, _ in arms:
    d = sorted(glob.glob("screenshots/dump/leak-%s_*" % name), key=os.path.getmtime)[-1]
    FR[name] = TS._frame(d, TS._rows(d)[-1])
rms = lambda a, b: float(np.sqrt(((FR[a] - FR[b]) ** 2).mean()))
rep, leak, real = rms("def-1", "def-2"), max(rms("def-1", "all-on"), rms("def-2", "all-on")), rms("def-1", "play-all-on")
print("pixel RMS: same render twice %.5f | display dials on, in truth mode %.5f | outside truth mode %.5f"
      % (rep, leak, real))
# TUNED: 1.5x the repeat spread; a real change at least 5x | learn by: the
# spread of rep over repeated runs of this test
ok = leak <= 1.5 * rep and real >= 5 * rep
print("truth mode immune to the display dials:", "PASS" if ok else "FAIL",
      "" if real >= 5 * rep else "(VACUOUS: the dials do not change the picture outside truth mode)")
sys.exit(0 if ok else 1)
