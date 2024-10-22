# !/usr/bin/python3.8
from __future__ import annotations

from argparse import ArgumentParser
from itertools import count
from typing import Tuple

from numpy import array, identity
from vtk import vtkPolyData, vtkSTLReader, vtkSTLWriter, vtkMatrix4x4, vtkTransform, vtkTransformPolyDataFilter

from features import Voxelizer
from rotation_factory import RotationFactory

Tuple3PolyData = Tuple[vtkPolyData, vtkPolyData, vtkPolyData]

def load_stl(filename: str) -> vtkPolyData:
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def save_stl(geometry: vtkPolyData, filename: str) -> vtkPolyData:
    writer = vtkSTLWriter()
    writer.SetFileName(filename)
    writer.SetInputData(geometry)
    writer.Update()

def main() -> None:
    parser = ArgumentParser(
        prog='Axis Registration',
        description="Normalize a vertebra's orientation. x pointing towards the right. y pointing upwards. z pointing to the back.",
    )

    parser.add_argument('filename')
    arguments = parser.parse_args()
    vertebra = load_stl(arguments.filename)
    for id_, rotation_matrix in enumerate(RotationFactory.Rotations):
        rotation_matrix_4x4 = identity(4)
        rotation_matrix_4x4[:3, :3] = rotation_matrix
        transform = vtkTransform()
        transform.SetMatrix(rotation_matrix_4x4.flatten())
        transform_filter =  vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(vertebra)
        transform_filter.Update()

        save_stl(transform_filter.GetOutput(), f'{arguments.filename[:-4]}.{id_}.stl')

    return
    voxelization = Voxelizer(vertebra)

    writer = vtkSTLWriter()
    poi = voxelization.points

    with open('../data/centers_template.mrk.json', 'r') as template:
        file_content = str().join(template.readlines()) % tuple(f'[{_1}, {_2}, {_3}]' for _1, _2, _3 in poi)
    print(file_content)

if __name__ == '__main__':
    main()
