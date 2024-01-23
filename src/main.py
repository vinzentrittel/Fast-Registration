# !/usr/bin/python3.8
from __future__ import annotations

from argparse import ArgumentParser
from itertools import count
from typing import Tuple

from numpy import array
from vtk import vtkPolyData, vtkSTLReader, vtkSTLWriter

from features import slice_geometry, demo_poc

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
    demo_poc(vertebra)
    return
    slices = slice_geometry(
        vertebra,
        axes=(array((1,0,0,)), array((0,1,0,)), array((0,0,1,)),),
        number_of_slices=3
    )

    centers = list()
    writer = vtkSTLWriter()
    for no, slice_ in zip(count(1), slices):
        geometry, center = slice_
        if geometry.GetNumberOfPoints() < 1:
            continue
        centers.append(center)
        writer.SetFileName(f'../data/simple.{no}.stl')
        writer.SetInputData(geometry)
        writer.Update()

    with open('../data/centers_template.mrk.json', 'r') as template:
        file_content = str().join(template.readlines()) % tuple(f'[{_1}, {_2}, {_3}]' for _1, _2, _3 in centers)
    with open('../data/centers.mrk.json', 'w') as out_file:
        out_file.write(file_content)

if __name__ == '__main__':
    main()
