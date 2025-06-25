# generate_derivative.py
# Aufruf in Blender via: blender --background --python generate_derivative.py -- --input in.stl --output out.stl --seed 42
from pathlib import Path

import bpy, bmesh, argparse, random, math
from mathutils import Vector, Matrix, noise
from numpy import array

def parse_args():
    import sys
    argv = sys.argv[sys.argv.index("--")+1:]
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Pfad zur Atlas-STL")
    p.add_argument("--output", required=True, help="Pfad zur Ausgabedatei")
    return p.parse_args(argv)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def load_mesh(path):
    bpy.ops.import_mesh.stl(filepath=path)
    obj = bpy.context.selected_objects[0]
    return obj

def create_lattice(mesh):
    bpy.ops.object.add(type="LATTICE")
    obj = bpy.context.selected_objects[0]
    #matrix = Matrix.LocRotScale(Vector(mesh.location), None, Vector(mesh.dimensions))
    #obj.data.transform(matrix)
    return obj

def setup_modifier(mesh, lattice):
    lattice.select_set(False)
    mesh.select_set(True)
    mod = mesh.modifiers.new("Lattice", "LATTICE")
    mod.object = lattice

def lattice_modification_generator(lattice, relative_offsets=(-0.1, 0.0, 0.1)):
    lattice.data.points[0].select = True

    for point in lattice.data.points:
        point.select = True
        tmp = point.co_deform
        for offset in [
            array((x, y, z,))
            for x in relative_offsets
            for y in relative_offsets
            for z in relative_offsets
        ]:
            tmp = Vector(array(point.co_deform))
            point.co_deform = Vector(array(point.co_deform) + offset)
            yield
            point.co_deform = tmp
        point.co_deform = tmp
        point.select = False
    bpy.ops.object.mode_set(mode="EDIT")

def export_mesh(obj, path):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_mesh.stl(filepath=path, use_selection=True)

def main():
    args = parse_args()
    clear_scene()
    mesh = load_mesh(args.input)
    lattice = create_lattice(mesh)
    setup_modifier(mesh, lattice)
    for n, _ in enumerate(lattice_modification_generator(lattice, relative_offsets=(-0.5, 0, 0.5,))):
        export_mesh(mesh, str(Path(args.output, f"derivative_{n:04d}.stl")))

    return

if __name__ == "__main__":
    main()
