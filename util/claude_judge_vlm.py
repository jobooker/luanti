#!/usr/bin/env python3
"""claude_judge_vlm — Gemini as a SECOND judge: names artifacts, and is
graded on known answers before any of its votes count.

The research (luanti-docs research/perceptual-judge.md) is blunt: VLMs
have position bias, call slight differences "the same", and are weakest
on noise. So: always pairwise, always with the reference shown, always
both orders (a vote counts only if it survives the swap), and its
accuracy on the known-answer ladders is reported next to every use.

Runs where the key is (the beelink: ~/.config/gemini/api-key). John,
2026-10-06: "Use Gemini or yourself to grade the images" -- Luanti renders
only.

  python3 claude_judge_vlm.py pair REF A B        -> JSON verdict
  python3 claude_judge_vlm.py calibrate PAIRS.json -> accuracy on known pairs
     PAIRS.json: [{"ref":..,"better":..,"worse":..,"what":..}, ..]
"""
import base64
import io
import json
import os
import sys
import urllib.request

from PIL import Image

MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-pro-preview")
KEY = open(os.path.expanduser("~/.config/gemini/api-key")).read().strip()

PROMPT = """You are grading a real-time path-traced renderer of a world made of
axis-aligned blocks. Image R is the REFERENCE: the converged, correct render.
Images 1 and 2 are two renders of the same view. Which is closer to R as a
person would see it: fewer visible differences such as grain/noise, blur,
wrong brightness or colour, missing or changed geometry (trees, terrain),
smearing or ghosting? Look carefully at fine detail and at distant terrain.
Answer ONLY with JSON: {"closer": 1 or 2 or 0 (0 = no visible difference),
"confidence": 0..1, "artifacts_1": [short phrases], "artifacts_2": [short phrases]}"""


def part(path):
    im = Image.open(path).convert("RGB")
    b = io.BytesIO()
    im.save(b, "JPEG", quality=95)
    return {"inline_data": {"mime_type": "image/jpeg",
                            "data": base64.b64encode(b.getvalue()).decode()}}


def ask(ref, a, b):
    body = {"contents": [{"parts": [
        {"text": PROMPT}, {"text": "R:"}, part(ref), {"text": "1:"}, part(a),
        {"text": "2:"}, part(b)]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "temperature": 0}}
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % MODEL,
        data=json.dumps(body).encode(),
        headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    txt = r["candidates"][0]["content"]["parts"][-1]["text"]
    v = json.loads(txt)
    if isinstance(v, list):      # sometimes wrapped in a list
        v = v[0] if v else {}
    return v


def ask_retry(ref, a, b, tries=3):
    for t in range(tries):
        try:
            return ask(ref, a, b)
        except Exception as e:
            err = e
    return {"closer": None, "error": str(err)}


def pair(ref, a, b):
    """both orders; the verdict stands only if it survives the swap"""
    v1 = ask_retry(ref, a, b)
    v2 = ask_retry(ref, b, a)
    c1 = {1: "A", 2: "B"}.get(v1.get("closer"), "same")
    c2 = {1: "B", 2: "A"}.get(v2.get("closer"), "same")
    verdict = c1 if c1 == c2 else "inconsistent"
    return {"verdict": verdict, "orders": [c1, c2],
            "artifacts_A": v1.get("artifacts_1", []) + v2.get("artifacts_2", []),
            "artifacts_B": v1.get("artifacts_2", []) + v2.get("artifacts_1", [])}


def calibrate(pairs_file):
    pairs = json.load(open(pairs_file))
    rows = []
    for p in pairs:
        v = pair(p["ref"], p["better"], p["worse"])
        ok = v["verdict"] == "A"
        rows.append({"what": p["what"], "verdict": v["verdict"],
                     "orders": v["orders"], "correct": ok})
        print("%-34s %-12s %s" % (p["what"], v["verdict"], "OK" if ok else "--"),
              flush=True)
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    inc = sum(r["verdict"] == "inconsistent" for r in rows) / n
    print("accuracy %.2f (%d pairs), inconsistent across order swap %.2f"
          % (acc, n, inc))
    return rows


if __name__ == "__main__":
    if sys.argv[1] == "pair":
        print(json.dumps(pair(*sys.argv[2:5]), indent=1))
    else:
        calibrate(sys.argv[2])
