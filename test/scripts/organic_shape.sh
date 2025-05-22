#!/usr/bin/env bash
# ---------------------------------------------------------------
# Organischer Blob aus Metaball-Ellipsoiden – Protrusion skaliert
# usage: ./organic_ellipsoids.sh CHAOS TARGET OUT.stl [ax ay az]
# ---------------------------------------------------------------

CHAOS=${1:-0.0}
TARGET=${2:-8000}
OUT=${3:-organic_ellipsoid.stl}
AX=${4:-1.0}
AY=${5:-1.0}
AZ=${6:-1.0}

# ─── Eingaben absichern ─────────────────────────────────────────
clamp () { echo "$1" | LC_NUMERIC=C awk '{if($1<0) print 0; else if($1>1) print 1; else printf "%.4f", $1}'; }
CHAOS=$(clamp "$CHAOS"); AX=$(clamp "$AX"); AY=$(clamp "$AY"); AZ=$(clamp "$AZ")

blender -b -noaudio --python-expr "
import bpy, random, math, mathutils, os

# ───── Parameter ───────────────────────────────────────────────
chaos   = float('$CHAOS')
target  = int('$TARGET')
outfile = os.path.abspath('$OUT')
ax,ay,az = map(float, ('$AX','$AY','$AZ'))
BUBBLES  = 12                     # konstante Blasen-Anzahl

random.seed()

# ───── Szene reset ─────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)

# ───── Haupt-Ellipsoid (Metaball) ──────────────────────────────
bpy.ops.object.metaball_add(type='BALL', radius=1.0)
meta_obj = bpy.context.active_object
meta     = meta_obj.data

# Auflösung in Abhängigkeit vom Ziel-Mesh
meta_res = max(0.02, 3.0 / (target ** 0.5))
meta.resolution        = meta_res
meta.render_resolution = meta_res
meta.threshold         = 0.6

# Element-0 in Ellipsoid verwandeln
e0 = meta.elements[0]
e0.type   = 'ELLIPSOID'
e0.size_x, e0.size_y, e0.size_z = ax, ay, az

# Mittelhalbachse (für Distanz-Berechnung)
avg_axis = (ax + ay + az) / 3.0

# ───── Blasen-Ellipsoide anlegen ───────────────────────────────
r_min, r_max = 0.10, 0.70          # min/max Radius
dist_factor  = 0.80 + 0.15*chaos   # 0.80 … 0.95

for _ in range(BUBBLES):
    # Richtung auf Einheitskugel
    vec = mathutils.Vector((
        random.uniform(-1,1),
        random.uniform(-1,1),
        random.uniform(-1,1)
    )).normalized()

    # Radius abhängig von Chaos
    radius = (r_min + (r_max - r_min)*chaos) * random.uniform(0.8,1.2)

    # Zentrum: etwas innerhalb/auf Oberfläche
    pos = vec * (dist_factor * avg_axis)

    # Neues Metaball-Element
    e = meta.elements.new()
    e.type   = 'ELLIPSOID'
    e.co     = pos
    e.radius = radius

    # Individuelle Ellipsoid-Streckung  (Chaos erhöht Exzentrizität)
    def rand_axis(): return random.uniform(0.7,1.3) ** (1 + chaos)
    e.size_x, e.size_y, e.size_z = rand_axis(), rand_axis(), rand_axis()

# ───── Metaball → Mesh, glätten ────────────────────────────────
bpy.ops.object.convert(target='MESH')
obj = bpy.context.active_object
bpy.ops.object.shade_smooth()

# ───── Polygonzahl anpassen ────────────────────────────────────
def faces(o): return len(o.data.polygons)
cur = faces(obj)

# A) Unter Ziel: Subdivision
if cur < target:
    import math
    steps = math.ceil(math.log(target/cur, 4))
    bpy.ops.object.modifier_add(type='SUBSURF')
    sub = obj.modifiers[-1]
    sub.levels = steps
    sub.render_levels = steps
    bpy.ops.object.modifier_apply(modifier=sub.name)
    cur = faces(obj)

# B) Über Ziel: Decimate
if cur > target:
    ratio = target / cur
    bpy.ops.object.modifier_add(type='DECIMATE')
    dec = obj.modifiers[-1]
    dec.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=dec.name)

# ───── STL-Export ──────────────────────────────────────────────
bpy.ops.export_mesh.stl(filepath=outfile, ascii=False)
print(f'Wrote STL to {outfile} (faces ≈ {faces(obj)})')
"
echo "✓ STL exportiert nach: $OUT"

