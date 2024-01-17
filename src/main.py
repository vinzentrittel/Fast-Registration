# !/usr/bin/python3.8
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
from itertools import count
from typing import Tuple

from numpy import (
    array,
    dot,
    inf as Infinity,
    ndarray,
)
from numpy.linalg import norm
from vtk import (
    vtkPolyData,
    vtkSTLReader,
    vtkCenterOfMass,
    vtkPlane,
    vtkClipPolyData,
    vtkSTLWriter,
)

Tuple3PolyData = Tuple[vtkPolyData, vtkPolyData, vtkPolyData]

@dataclass
class Dimension:
    frontal = 1.0
    longitudinal = 1.0
    sagittal = 1.0

@dataclass(frozen=True)
class SlicerConfig:
    input_data: vtkPolyData
    direction: ndarray
    number_of_slices: int

    center_of_mass: ndarray = None
    length: float = None
    center: ndarray = None

@dataclass(frozen=True)
class SlicerState:
    clippers: Tuple[vtkClipPolyData]
    centers: Tuple[ndarray]

class NewSlicer:       
    def __init__(self, config: SlicerConfig) -> None:
        self._config = config
        self._config = SlicerConfig(
            self.input_data,
            self.direction,
            self.number_of_slices,
            self.center_of_mass,
            self.length,
            self.center,
        )
        self._state_change = True
        self._state: SlicerState = None

    def calculate_length_in_direction(self, direction: ndarray):
        minimum_projection = Infinity
        maximum_projection = -Infinity

        for n in range(self.input_data.GetNumberOfPoints()):
            point = array(self.input_data.GetPoint(n))
            projection = dot(point, direction)

            minimum_projection = min(minimum_projection, projection)
            maximum_projection = max(maximum_projection, projection)

        return maximum_projection - minimum_projection

    @property
    def output(self) -> Tuple[vtkPolyData]:
        slices = list()
        for clipper in self.clippers:
            slices.append(clipper.GetOutput())
        slices.append(self.clippers[-1].GetClippedOutput())
        return tuple(slices)

    @property
    def center_of_mass(self) -> ndarray:
        if not self.config.center_of_mass is None:
            return self.config.center_of_mass

        center_of_mass_filter = vtkCenterOfMass()
        center_of_mass_filter.SetUseScalarsAsWeights(False)
        center_of_mass_filter.SetInputData(self.input_data)
        center_of_mass_filter.Update()
        center_of_mass = center_of_mass_filter.GetCenter()
        
        return array(center_of_mass)

    @property
    def length(self) -> float:
        if not self.config.length is None:
            return self.config.length

        return self.calculate_length_in_direction(self.direction)

    def update(self) -> None:
        clippers = tuple(self.make_inverse_clipper() for _ in range(self.number_of_slices - 1))

        # setup cutting planes
        start = self.center_of_mass - self.length * self.direction / 2.0
        step = self.length / self.number_of_slices * self.direction
        for iteration, clipper in enumerate(clippers, start=1):
            plane = vtkPlane()
            plane.SetOrigin(start + iteration * step)
            plane.SetNormal(self.direction)
            clipper.SetClipFunction(plane)

        # setup sub-geometry centers
        start = self.center - self.length * self.direction / 2.0
        centers = tuple(start + (n + 0.5) * step for n in range(self.number_of_slices))

        # setup chain
        clippers[0].SetInputData(self.input_data)
        for previous_clipper, next_clipper in zip(clippers, clippers[1:]):
            next_clipper.SetInputConnection(previous_clipper.GetClippedOutputPort())
        clippers[-1].Update()

        self._state = SlicerState(clippers, centers)
        self._state_change = False
        return clippers


    @property
    def clippers(self) -> Tuple[vtkClipPolyData]:
        if self._state_change:
            self.update()

        return self._state.clippers

    @property
    def centers(self) -> Tuple[ndarray]:
        if self._state_change:
            self.update()

        return self._state.centers

    @property
    def center(self) -> ndarray:
        return self.center_of_mass if self.config.center is None else self.config.center

    @property
    def input_data(self) -> vtkPolyData:
        return self.config.input_data

    @property
    def direction(self) -> ndarray:
        return self.config.direction

    @property
    def number_of_slices(self) -> int:
        return self.config.number_of_slices

    @property
    def config(self) -> SlicerConfig:
        return self._config

    @config.setter
    def config(self, new_config: SlicerConfig) -> None:
        self._config = new_config
        self._state_change = True

    @staticmethod
    def make_inverse_clipper() -> vtkClipPolyData:
        clipper = vtkClipPolyData()
        clipper.GenerateClippedOutputOn()
        clipper.InsideOutOn()
        return clipper

    @classmethod
    def make(
        cls,
        input_data: vtkPolyData,
        direction: ndarray,
        number_of_slices: int,
        center_of_mass: ndarray = None,
        length: float = None,
        center: ndarray = None,
    ) -> NewSlicer:
        config = SlicerConfig(
            input_data, direction, number_of_slices, center_of_mass, length, center
        )
        return NewSlicer(config)

def slice_geometry(geometry: vtkPolyData, axes: Tuple[ndarray], number_of_slices: int) -> Tuple[vtkPolyData]:
    def slice_rec(slicer, axes, lengths):
        if len(axes) == 0:
            return tuple(zip(slicer.output, slicer.centers))

        current_axis, *remaining_axes = axes
        current_length, *remaining_lengths = lengths
        slicers = tuple(
            NewSlicer.make(
                input_data=sub_geometry,
                direction=current_axis,
                number_of_slices=slicer.number_of_slices,
                center_of_mass=slicer.center_of_mass,
                length=current_length,
                center=sub_center,
            ) 
            for sub_geometry, sub_center in zip(slicer.output, slicer.centers)
        )

        return reduce(
            lambda x, y: x + y,
            tuple(slice_rec(s, axes=remaining_axes, lengths=remaining_lengths) for s in slicers),
            tuple(),
        )

    slicer = NewSlicer.make(geometry, direction=axes[0], number_of_slices=number_of_slices)
    lengths = tuple(slicer.calculate_length_in_direction(a) for a in axes)
    return slice_rec(
        slicer,
        axes=axes[1:],
        lengths=lengths[1:]
    )

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
    slices = slice_geometry(
        vertebra,
        axes=(array((1,0,0,)), array((0,1,0,)), array((0,0,1,)),),
        number_of_slices=3
    )
    #slices = flatten_nested_slices(slice_rec(vertebra, number_of_cuts=2, axes=(array((1,0,0,)), array((0,1,0,)), array((0,0,1,)),)))

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
