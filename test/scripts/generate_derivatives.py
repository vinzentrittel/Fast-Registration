# generate_derivative.py
# Aufruf in Blender via: blender --background --python generate_derivative.py -- --input in.stl --output out.stl --seed 42

import bpy, bmesh, argparse, random, math
from mathutils import Vector, Matrix, noise

def parse_args():
    import sys
    argv = sys.argv[sys.argv.index("--")+1:]
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Pfad zur Atlas-STL")
    p.add_argument("--output", required=True, help="Pfad zur Ausgabedatei")
    p.add_argument("--seed",   type=int, default=0, help="Zufallsseed (z.B. Durchlaufindex)")
    return p.parse_args(argv)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def load_mesh(path):
    bpy.ops.import_mesh.stl(filepath=path)
    obj = bpy.context.selected_objects[0]
    return obj

def bbox_max_extent(obj):
    # Weltkoordinaten aller Eckpunkte des Bounding-Box
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))

def apply_multifreq_noise(obj, max_disp, seed):
    # Rauschen auf 4 Oktaven, Lacunarity=2, Gain=0.5
    random.seed(seed)
    me = obj.data
    bm = bmesh.new(); bm.from_mesh(me)
    base_freq = 1.0
    octaves = 2
    lacunarity = 2.0
    gain = 0.5

    for v in bm.verts:
        p = v.co.copy()
        amp = 1.0
        freq = base_freq
        n_total = 0.0
        for iteration in range(octaves):
            if iteration != 0:
                n_total += noise.noise(p * freq) * amp
            freq *= lacunarity
            amp  *= gain
        disp = n_total * max_disp
        v.co += v.normal * disp

    bm.to_mesh(me)
    bm.free()

def apply_linear_transform(obj, seed):
    random.seed(seed)
    # Skalierung
    sx = random.uniform(0.75, 1.25)
    sy = random.uniform(0.75, 1.25)
    sz = random.uniform(0.75, 1.25)
    S = Matrix.Diagonal((sx, sy, sz, 1.0))
    # Rotation um X, Y, Z
    rx = random.uniform(0, 2*math.pi)
    ry = random.uniform(0, 2*math.pi)
    rz = random.uniform(0, 2*math.pi)
    R = Matrix.Rotation(rx, 4, 'X') @ Matrix.Rotation(ry, 4, 'Y') @ Matrix.Rotation(rz, 4, 'Z')
    # Translation
    tx = random.uniform(-5, 5)
    ty = random.uniform(-5, 5)
    tz = random.uniform(-5, 5)
    T = Matrix.Translation(Vector((tx, ty, tz)))
    # Gesamte Transformation M = S · R · T
    M = S @ R @ T
    obj.matrix_world = M

def export_mesh(obj, path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_mesh.stl(filepath=path, use_selection=True)

def main():
    args = parse_args()
    clear_scene()
    obj = load_mesh(args.input)

    # Bestimme maximale Breite für das Displacement-Limit
    extent = bbox_max_extent(obj)
    max_disp = 0.1 * extent

    apply_multifreq_noise(obj, max_disp, args.seed)
    apply_linear_transform(obj, args.seed)
    export_mesh(obj, args.output)

if __name__ == "__main__":
    main()

