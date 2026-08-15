#!/usr/bin/env python3
"""Generate the gallery's uniform-color node textures.

The furnace and Cornell rooms are built from claude_bridge:* nodes whose
textures are FLAT single-color 16x16 PNGs — that is what makes the
tracer's per-cell albedo (the minimap average color) exact by
construction rather than a measurement. pathAlbedo() linearizes as
pow(c/255, 2.2), so:

    186 -> albedo 0.50        221 -> albedo 0.73        255 -> 1.0

The Cornell side walls are PURE channels (red 221,0,0 / green 0,221,0):
any red landing on the white wall then unambiguously arrived by bleed.

These were a hand-made deployment artifact on the first seat and never
committed, so the gallery could not be rebuilt from the repo alone.
Written by hand as a minimal PNG so this has no Pillow dependency.

Usage: util/claude_gallery_textures.py [<mod-textures-dir>]
"""
import binascii
import os
import struct
import sys

COLORS = {
    "claude_gray186.png": (186, 186, 186),
    "claude_gray221.png": (221, 221, 221),
    "claude_white255.png": (255, 255, 255),
    "claude_red221.png": (221, 0, 0),
    "claude_green221.png": (0, 221, 0),
}
SIZE = 16


def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", binascii.crc32(tag + data) & 0xFFFFFFFF))


def solid_png(rgb, size=SIZE):
    import zlib
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB
    row = b"\x00" + bytes(rgb) * size                          # filter 0
    raw = row * size
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.join(os.path.dirname(here), "mods",
                           "claude_bridge", "textures")
    out = sys.argv[1] if len(sys.argv) > 1 else default
    os.makedirs(out, exist_ok=True)
    for name, rgb in COLORS.items():
        with open(os.path.join(out, name), "wb") as f:
            f.write(solid_png(rgb))
        print("wrote %s %s" % (name, rgb))


if __name__ == "__main__":
    main()
