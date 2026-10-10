#!/usr/bin/env python3
"""Ask Gemini for a 16x16 relief height map per texture face (carve2).
Only game textures go out. Key read from file, never printed."""
import base64, hashlib, io, json, sys, time, datetime, urllib.request, concurrent.futures as cf
from PIL import Image, ImageDraw
import numpy as np

MODEL = "gemini-3.1-pro-preview"
KEY = open('/home/jobooker/.config/gemini/api-key').read().strip()
TEX = '/tmp/carve2/tex/'
OUT = '/tmp/carve2/hm/'
GRASS_TINT = (145, 189, 89)   # a plains-like green, for showing the grayscale tile only

def load(stem):
    if stem == 'mcl_core_grass_block_top':
        g = np.asarray(Image.open(TEX + stem + '.png').convert('RGB'), dtype=np.float32)
        return np.clip(g * np.array(GRASS_TINT) / 255.0, 0, 255).astype(np.uint8)
    if stem == 'default_dirt^mcl_dirt_grass_shadow+mcl_core_grass_block_side_overlay':
        base = Image.open(TEX + 'default_dirt.png').convert('RGBA')
        base.alpha_composite(Image.open(TEX + 'mcl_dirt_grass_shadow.png').convert('RGBA'))
        ov = np.asarray(Image.open(TEX + 'mcl_core_grass_block_side_overlay.png').convert('RGBA'), dtype=np.float32)
        b = np.asarray(base.convert('RGB'), dtype=np.float32)
        a = ov[:, :, 3:4] / 255.0
        o = ov[:, :, :3] * np.array(GRASS_TINT) / 255.0
        return np.clip(b * (1 - a) + o * a, 0, 255).astype(np.uint8)
    return np.asarray(Image.open(TEX + stem + '.png').convert('RGB'))

def grid_image(rgb, s=24):
    m = 28
    im = Image.new('RGB', (16 * s + m, 16 * s + m), (255, 255, 255))
    im.paste(Image.fromarray(rgb).resize((16 * s, 16 * s), Image.NEAREST), (m, m))
    d = ImageDraw.Draw(im)
    for i in range(17):
        d.line([(m + i * s, m), (m + i * s, m + 16 * s)], fill=(255, 0, 255) if i % 4 == 0 else (90, 90, 90))
        d.line([(m, m + i * s), (m + 16 * s, m + i * s)], fill=(255, 0, 255) if i % 4 == 0 else (90, 90, 90))
    for i in range(16):
        d.text((m + i * s + 6, 8), str(i), fill=(0, 0, 0))
        d.text((4, m + i * s + 6), str(i), fill=(0, 0, 0))
    return im

def tiled(rgb, nx, ny, s=8):
    t = np.tile(rgb, (ny, nx, 1))
    return Image.fromarray(t).resize((16 * nx * s, 16 * ny * s), Image.NEAREST)

def png_b64(im):
    b = io.BytesIO(); im.save(b, 'PNG'); return base64.b64encode(b.getvalue()).decode()

COMMON = """You are helping turn a 16x16 pixel-art game texture (Mineclonia, a Minecraft-like game) into REAL carved geometry for a path-traced voxel renderer. Each texel of the face becomes a 1/16 m square column of the block's surface, and you decide how far each column is recessed into the block. The renderer then shades the recesses itself with real light, so the depth must describe the PHYSICAL OBJECT the texture depicts.

Block: {block}
Face: {face}
What the texture depicts: {depicts}
Physical expectation for this object: {expect}
Tiling: {tiling}

Return a 16x16 integer HEIGHT MAP (really a depth map): 0 = the outer surface, flush with the block face; 1 = recessed 1/16 m; 2 = recessed 2/16 m{deep3}. Allowed values: {allowed}.
Row 0 is the TOP row of the image and column 0 the LEFT column -- the same indexing as the RGB grid and the labelled grid image.

Rules:
- Depth follows the real object, not pixel brightness alone. The artist painted shadows into grooves, which is a useful clue, but a texel that is dark because the MATERIAL is darker (a dark knot, a darker stone, a dark-covered book, a darker clod) stays at the depth of the part of the object it belongs to.
- Keep it physically plausible: connected features (a groove is a continuous line, a furrow a continuous channel), no isolated single-texel pits or pillars unless the object really has one.
- Respect the tiling: a feature that crosses an edge of the tile must meet itself at the opposite edge where the tiling rule says the tiles join.

Image 1: the texture, enlarged, with a grid (row numbers on the left, column numbers on top; magenta lines every 4 texels).
Image 2: the texture tiled {tilenote}, to show how it joins its neighbours.

RGB grid, row 0 first, sRGB hex per texel, columns 0..15 left to right:
{grid}

Reply with JSON only, exactly this shape:
{{"height": [[16 integers], ... 16 rows ...], "explanation": "one paragraph: what you identified as raised and recessed, and why"}}"""

BOTH = "The texture tiles seamlessly in both directions: identical blocks sit side by side and on top of each other, so column 15 meets column 0 of the next block and row 15 meets row 0 of the block below."

FACES = {
 'default_tree': dict(block='Oak log (mcl_trees:tree_oak), upright', face='side (all four sides of the log are this texture)',
   depicts='oak bark on the side of a vertical tree trunk',
   expect='Real oak bark is long vertical ridges (plates) separated by deep vertical furrows. Furrows run VERTICALLY and are CONTINUOUS from block to block, because a trunk is a stack of these blocks. Ridge faces are the surface (0); shallow cracks and the edges of plates 1; the deepest furrows 2.',
   tiling='A trunk is a vertical stack of identical logs: row 0 directly continues row 15 of the log above, so a furrow that reaches the bottom edge in some column must continue at the same column from the top edge. The bark also wraps around the trunk, so column 15 meets column 0.',
   allowed='0, 1, 2', tile=(3, 3)),
 'default_tree_top': dict(block='Oak log (mcl_trees:tree_oak), upright', face='top and bottom (the sawn end of the log)',
   depicts='the sawn end of an oak log seen from above: end grain with growth rings, and the bark ring around the outside',
   expect='A sawn log end is close to flat. The bark ring at the perimeter is the outside of the trunk seen end-on (its outer edge is flush with the sides). Real relief here is small: perhaps radial drying cracks (checks), a slight step where bark meets wood. Growth rings are colour, not grooves, unless you think a real sawn end would show them recessed.',
   tiling='Logs stack vertically, so this face usually touches the next log and is only seen at the top of a trunk or on a fallen log; it does not need to continue across its edges.',
   allowed='0, 1, 2', tile=(2, 2)),
 'default_wood': dict(block='Oak planks (mcl_trees:wood_oak); also the top and bottom of the bookshelf', face='all six faces',
   depicts='oak wooden planks: horizontal boards laid edge to edge',
   expect='Seams between boards are narrow grooves (the dark lines between boards). Short vertical butt joints where one board ends and the next begins are grooves too. Board faces are flat (0). Grain and knots are colour, not depth (at most a slight 1 for a very pronounced line; use sparingly). Use 2 only if a seam is genuinely deep.',
   tiling=BOTH, allowed='0, 1, 2', tile=(3, 3)),
 'mcl_core_planks_spruce': dict(block='Spruce planks (mcl_trees:wood_spruce)', face='all six faces',
   depicts='spruce wooden planks: horizontal boards laid edge to edge',
   expect='Seams between boards are narrow grooves (the dark lines between boards). Short vertical butt joints where one board ends and the next begins are grooves too. Board faces are flat (0). Grain and knots are colour, not depth (at most a slight 1 for a very pronounced line; use sparingly). Use 2 only if a seam is genuinely deep.',
   tiling=BOTH, allowed='0, 1, 2', tile=(3, 3)),
 'default_cobble': dict(block='Cobblestone (mcl_core:cobble)', face='all six faces',
   depicts='cobblestone: irregular rounded stones set in mortar',
   expect='Each stone\'s face is the surface (0); stones are rounded, so their edges can fall away (1) toward the joints; the mortar joints between stones are recessed (1 or 2, 2 where the gap is wide and deep). Different stones have different colours -- a darker stone is still a stone at the surface.',
   tiling=BOTH, allowed='0, 1, 2', tile=(3, 3)),
 'default_stone_brick': dict(block='Stone bricks (mcl_core:stonebrick)', face='all six faces',
   depicts='stone bricks: rectangular dressed stone blocks in a bond pattern, with mortar joints',
   expect='Brick faces are the surface (0), maybe with a slightly bevelled edge (1) along a brick\'s border; the mortar joints between bricks are recessed grooves (1 or 2). Joints are straight continuous lines.',
   tiling=BOTH, allowed='0, 1, 2', tile=(3, 3)),
 'default_bookshelf': dict(block='Bookshelf (mcl_books:bookshelf)', face='the four sides (the shelf front)',
   depicts='the front of a wooden bookshelf: a frame of boards (a top board, a bottom board, a shelf board across the middle) holding two rows of books standing upright with their spines facing out',
   expect='The boards of the frame and the shelf board are the front surface (0). The books stand recessed behind the front of the shelf: each book spine is a VERTICAL strip of constant depth (1), the dark gaps between books and the dark space above shorter books are deeper (2). The depth must follow the books: spines are upright strips, a leaning book is a slanted strip, and every book sits back from the shelf boards.',
   tiling='Bookshelves stack vertically and stand side by side; the frame boards meet the next shelf\'s boards. Within one face the layout is a single shelf unit.',
   allowed='0, 1, 2', tile=(3, 2)),
 'crafting_workbench_front': dict(block='Crafting table (mcl_crafting_table:crafting_table)', face='front (the side faces of the table use an identical texture)',
   depicts='the side of a wooden crafting table/workbench, as drawn by this texture pack (look at the image and decide what wooden parts it shows: a top board, legs or posts, panels, decorative carving, tools)',
   expect='Identify the real wooden construction. Parts nearest the viewer (the table top\'s edge board, posts/legs, raised carved shapes) are the surface (0); panels set back between posts, and the background of any carving, are recessed (1); deep gaps (2) only where wood is genuinely cut away.',
   tiling='A single block; faces meet neighbouring blocks at their edges but the picture is one object.',
   allowed='0, 1, 2', tile=(2, 1)),
 'crafting_workbench_side': dict(block='Crafting table (mcl_crafting_table:crafting_table)', face='side',
   depicts='the side of a wooden crafting table/workbench, as drawn by this texture pack (look at the image and decide what wooden parts it shows: a top board, legs or posts, panels, decorative carving, tools)',
   expect='Identify the real wooden construction. Parts nearest the viewer (the table top\'s edge board, posts/legs, raised carved shapes) are the surface (0); panels set back between posts, and the background of any carving, are recessed (1); deep gaps (2) only where wood is genuinely cut away.',
   tiling='A single block; faces meet neighbouring blocks at their edges but the picture is one object.',
   allowed='0, 1, 2', tile=(2, 1)),
 'crafting_workbench_top': dict(block='Crafting table (mcl_crafting_table:crafting_table)', face='top',
   depicts='the top of a crafting table: a wooden board with a border frame and a 3x3 crafting grid',
   expect='The border frame and the top surface are the surface (0). The 3x3 grid is inlaid: either the grid lines are grooves (1) or the grid cells are shallow recessed squares (1) -- decide from the picture what the real object would be. Use 2 only for genuinely deep cuts.',
   tiling='A single block top; it does not need to continue across its edges.',
   allowed='0, 1, 2', tile=(2, 2)),
 'default_furnace_front_active': dict(block='Furnace, lit (mcl_furnaces:furnace_active)', face='front',
   depicts='the front of a stone furnace: masonry of stone blocks with mortar joints around a fire mouth opening in the lower middle, fire burning inside',
   expect='Stone block faces are the surface (0); mortar joints between the stones are recessed (1); the fire mouth is a genuinely deep opening (3) -- the fire and the dark interior sit at the back of it. The glowing orange-lit stones around the opening are the surround of the mouth, part of the masonry.',
   tiling='A single block; it does not need to continue across its edges.',
   allowed='0, 1, 2, 3', deep3=', 3 = recessed 3/16 m (ONLY for the fire mouth)', tile=(2, 1)),
 'default_furnace_side': dict(block='Furnace (mcl_furnaces:furnace)', face='sides and back',
   depicts='the side of a stone furnace: masonry of stone blocks with mortar joints',
   expect='Stone block faces are the surface (0), perhaps with bevelled edges (1); the mortar joints between the stones are recessed (1 or 2).',
   tiling='Furnaces are single blocks but often stand side by side; the masonry need not continue across edges.',
   allowed='0, 1, 2', tile=(2, 1)),
 'default_furnace_top': dict(block='Furnace (mcl_furnaces:furnace)', face='top',
   depicts='the top of a stone furnace: a stone slab with a lighter border around it',
   expect='Decide from the picture: e.g. a raised rim (0) around a slightly lower slab face (1), or a flat slab with a chamfered edge. Stone speckle is colour, not depth.',
   tiling='A single block top.', allowed='0, 1, 2', tile=(2, 2)),
 'default_furnace_bottom': dict(block='Furnace (mcl_furnaces:furnace)', face='bottom',
   depicts='the bottom of a stone furnace: a stone slab with a lighter border around it',
   expect='Decide from the picture: e.g. a raised rim (0) around a slightly lower slab face (1), or a flat slab with a chamfered edge. Stone speckle is colour, not depth.',
   tiling='A single block bottom.', allowed='0, 1, 2', tile=(2, 2)),
 'default_dirt': dict(block='Dirt (mcl_core:dirt); also the bottom of grass blocks', face='all faces',
   depicts='bare soil: clods, crumbs and small stones of earth',
   expect='Ground relief is shallow: lumps and clods stand at the surface (0), the soil between them sits slightly lower (1). Only 0 and 1.',
   tiling=BOTH, allowed='0, 1', tile=(3, 3)),
 'mcl_core_grass_block_top': dict(block='Grass block (mcl_core:dirt_with_grass)', face='top',
   depicts='the top of a grass block: short grass turf seen from directly above (the grayscale tile is tinted green by the biome; shown here with a plains green)',
   expect='Shallow: clumps and tufts of grass blades are the surface (0), the gaps between clumps slightly lower (1). Only 0 and 1.',
   tiling=BOTH, allowed='0, 1', tile=(3, 3)),
 'default_dirt^mcl_dirt_grass_shadow+mcl_core_grass_block_side_overlay': dict(block='Grass block (mcl_core:dirt_with_grass)', face='side',
   depicts='the side of a grass block: soil, with the edge of the grass turf hanging over the top (a green fringe along the top rows with an irregular lower edge, and a dark shadow under it)',
   expect='Shallow: the grass fringe is turf at the top edge of the block and stands at the surface (0); the soil below sits slightly lower, with its lumps and clods at 0 and the soil between them at 1. Only 0 and 1.',
   tiling='Grass blocks stand side by side (column 15 meets column 0). Row 0 is the top edge, where the side meets the grass top; the bottom edge meets dirt below.',
   allowed='0, 1', tile=(3, 1)),
}

def build_prompt(stem, extra=''):
    f = FACES[stem]
    rgb = load(stem)
    grid = '\n'.join('row %2d: ' % r + ' '.join('%02x%02x%02x' % tuple(rgb[r, c]) for c in range(16)) for r in range(16))
    nx, ny = f['tile']
    p = COMMON.format(block=f['block'], face=f['face'], depicts=f['depicts'], expect=f['expect'], tiling=f['tiling'],
                      deep3=f.get('deep3', ''), allowed=f['allowed'], tilenote='%d wide x %d high' % (nx, ny), grid=grid)
    return p + extra, rgb

def call(stem, extra='', tag=''):
    prompt, rgb = build_prompt(stem, extra)
    nx, ny = FACES[stem]['tile']
    body = dict(contents=[dict(parts=[dict(text=prompt),
        dict(inline_data=dict(mime_type='image/png', data=png_b64(grid_image(rgb)))),
        dict(inline_data=dict(mime_type='image/png', data=png_b64(tiled(rgb, nx, ny))))])],
        generationConfig=dict(responseMimeType='application/json', temperature=0.2))
    url = 'https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent' % MODEL
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'x-goog-api-key': KEY, 'Content-Type': 'application/json'})
            t0 = time.time()
            resp = json.load(urllib.request.urlopen(req, timeout=600))
            break
        except Exception as e:
            print(stem, 'attempt', attempt, 'failed:', type(e).__name__, str(e)[:200], flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        return stem, None
    raw = ''.join(p.get('text', '') for p in resp['candidates'][0]['content']['parts'] if not p.get('thought'))
    try:
        j = json.loads(raw)
        h = np.array(j['height'], dtype=int)
        assert h.shape == (16, 16), h.shape
        allowed = [int(a) for a in FACES[stem]['allowed'].split(', ')]
        assert set(np.unique(h)) <= set(allowed), np.unique(h)
    except Exception as e:
        print(stem, 'bad reply:', e, flush=True)
        j = None
    rec = dict(texture=stem, height=j['height'] if j else None, explanation=j.get('explanation') if j else None,
               provenance=dict(model=MODEL, model_version=resp.get('modelVersion'), date=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                               seconds=round(time.time() - t0, 1), temperature=0.2,
                               images=['16x16 texture at 24x nearest with labelled grid', 'texture tiled %dx%d at 8x nearest' % (nx, ny)],
                               texture_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                               grass_tint_shown=list(GRASS_TINT) if 'grass' in stem else None,
                               prompt=prompt, raw_reply=raw))
    json.dump(rec, open(OUT + stem + tag + '.json', 'w'), indent=1)
    return stem, j is not None

if __name__ == '__main__':
    stems = sys.argv[1:] or list(FACES)
    with cf.ThreadPoolExecutor(6) as ex:
        for stem, ok in ex.map(call, stems):
            print(stem, 'OK' if ok else 'FAILED', flush=True)
