"""Moving light, step 1: the reference (each mover step converged) and the
real-time arm, same camera, same light path."""
import os, subprocess, sys, json, time
sys.path.insert(0, os.path.expanduser("~/code/luanti/util"))
import claude_lab as lab
REPO = os.path.expanduser("~/code/luanti")
D = ["--dial", "claude_cascades=1", "--dial", "claude_exposure=1"]
subprocess.run(["python3", "util/claude_shoot.py", "--pos", "241.7", "9", "231.7", "--yaw", "315",
                "--pitch", "-12", "--frames", "30", "--name", "warm"] + D, cwd=REPO, capture_output=True)
print(subprocess.run(["python3", "util/claude_scene_lights.py"], cwd=REPO, capture_output=True, text=True).stdout[-300:])
lab.rpc("abm", on=False)
out = {}
for name, extra in (("ref", ["--dial", "claude_path_hold=512"]), ("realtime", []),
                    ("realtime-denoise", ["--dial", "claude_denoise=1"])):
    r = subprocess.run(["python3", "util/claude_motion.py", "--skip-seat", "--path", "util/paths/torch-park.txt",
                        "--mover", "util/paths/glow-walk.txt", "--scale", "1", "--name", "glow-" + name] + D + extra,
                       cwd=REPO, capture_output=True, text=True).stdout.strip().splitlines()
    print(name, [l for l in r if "dumped" in l or "REFUSED" in l], flush=True)
    out[name] = r[-1]
lab.rpc("abm", on=True)
json.dump(out, open("/tmp/dyn_dumps.json", "w"), indent=1)
print(out)
