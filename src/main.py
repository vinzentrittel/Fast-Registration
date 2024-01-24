# !/usr/bin/python3.8
from __future__ import annotations

from argparse import ArgumentParser
from itertools import count
from typing import Tuple

from numpy import array
from vtk import vtkPolyData, vtkSTLReader, vtkSTLWriter

from features import Voxelizer

Tuple3PolyData = Tuple[vtkPolyData, vtkPolyData, vtkPolyData]

def load_stl(filename: str) -> vtkPolyData:
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def main() -> None:
    parser = ArgumentParser(
        prog='Axis Registration',
        description="Normalize a vertebra's orientation. x pointing towards the right. y pointing upwards. z pointing to the back.",
    )

    parser.add_argument('filename')
    arguments = parser.parse_args()

    vertebra = load_stl(filename=arguments.filename)
    voxelization = Voxelizer(vertebra)

    writer = vtkSTLWriter()
    poi = voxelization.points

    with open('../data/centers_template.mrk.json', 'r') as template:
        file_content = str().join(template.readlines()) % tuple(f'[{_1}, {_2}, {_3}]' for _1, _2, _3 in poi)
    print(file_content)

if __name__ == '__main__':
    main()
