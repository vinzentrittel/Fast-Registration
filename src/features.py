from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
from typing import Tuple

from numpy import abs, argmax, array, dot, inf as Infinity, newaxis, ndarray, outer, zeros
from numpy.linalg import norm
from vtk import (
    vtkPolyData,
    vtkClipPolyData,
    vtkCenterOfMass,
    vtkIdList,
    vtkOBBTree,
    vtkPlane,
)
from vtk.util.numpy_support import vtk_to_numpy

@dataclass(frozen=True)
class SlicerConfig:
    """
    Immutable configuration for a Slicer object.

    Properties:
    input_data - geometry to slice up
    direction - numpy array of shape (3,)
    number_of_slices - nomen est omen

    Optional properties:
    center_of_mass - origin for the algorithm, to construct the cuts around
    length - total length of the geometry along 'direction' axis, slice
             thickness depends on this
    center - center of the middle slice. If not provided, CoM is used.
    """
    input_data: vtkPolyData
    direction: ndarray
    number_of_slices: int

    center_of_mass: ndarray = None
    length: float = None
    center: ndarray = None

@dataclass(frozen=True)
class SlicerState:
    """
    Intermediate results fomr a Slicer object. It is calculated from the
    data provided by SlicerConfig.

    Properties:
    clippers - vtk representations of all cuts through a geometry.
    centers - individual center coordinates for all slices.
    """
    clippers: Tuple[vtkClipPolyData]
    centers: Tuple[ndarray]

class Slicer:
    """
    Algorithm class to slice divide a vtkPolyData geometry into a number
    of pieces.
    """
    def __init__(self, config: SlicerConfig) -> None:
        """
        Initialize by SlicerConfig object.

        Arguments:
        config - see SlicerConfig documentation.
        """
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

    def calculate_length_in_direction(self, direction: ndarray) -> ndarray:
        """
        For the 'input_data' provided, calculate the length along
        the 'direction' axis.
        """
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
        """
        Return the calculated slices.
        """
        slices = list()
        for clipper in self.clippers:
            slices.append(clipper.GetOutput())
        slices.append(self.clippers[-1].GetClippedOutput())
        return tuple(slices)

    @property
    def center_of_mass(self) -> ndarray:
        """
        Return the source "input_data"'s center of mass, or
        the user-defined center of mass, if provided.
        """
        if not self.config.center_of_mass is None:
            return self.config.center_of_mass

        return self.calculate_center_of_mass(self.input_data)

    @property
    def length(self) -> float:
        """
        Return the source "input_data"'s length along the
        'direction_axis' or the user-defined length, if
        provided.
        """
        if not self.config.length is None:
            return self.config.length

        return self.calculate_length_in_direction(self.direction)

    def update(self) -> None:
        """
        Recalculate the slices. This may produce new values for
        properties 'clippers' and 'centers'.
        """
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
        """
        Return a tuple of vtkClipPolyData with size being 'number_of_slices' - 1.
        Each object represents one of the equidistant cuts along the 'input_data'
        geometry.
        """
        if self._state_change:
            self.update()

        return self._state.clippers

    @property
    def centers(self) -> Tuple[ndarray]:
        """
        Return a tuple of numpy arrays, the tuple size is 'number_of_slices'. Each
        array is of shape (3,) and represents one slices center. The centers are
        either arranged around a line along the 'direction' axis, passing through
        'center_of_mass' or passing through user-defined 'center', if provided.
        """
        if self._state_change:
            self.update()

        return self._state.centers

    @property
    def center(self) -> ndarray:
        """
        Return eiter "input_data"'s 'center_of_mass' or user-defined 'center', if
        provided.
        """
        return self.center_of_mass if self.config.center is None else self.config.center

    @property
    def input_data(self) -> vtkPolyData:
        """
        Return the user-provided 'input_data' object.
        """
        return self.config.input_data

    @property
    def direction(self) -> ndarray:
        """
        Return the user-provided 'direction' axis.
        """
        return self.config.direction

    @property
    def number_of_slices(self) -> int:
        """
        Return the user-provided 'number_of_slices' setting.
        """
        return self.config.number_of_slices

    @property
    def config(self) -> SlicerConfig:
        """
        Return the user-provided SlicerConfig object.
        """
        return self._config

    @config.setter
    def config(self, new_config: SlicerConfig) -> None:
        """
        Set new data for this Slicer object. Due to lazy initialization, the
        new data is only calculated, on need.
        """
        self._config = new_config
        self._state_change = True

    @staticmethod
    def make_inverse_clipper() -> vtkClipPolyData:
        """
        Factory function to create a vtkClipPolyData object, that switches
        output with clipped output. This is only an aesthetic choice.
        """
        clipper = vtkClipPolyData()
        clipper.GenerateClippedOutputOn()
        clipper.InsideOutOn()
        return clipper

    @staticmethod
    def calculate_center_of_mass(geometry: vtkPolyData) -> ndarray:
        """
        Return the center of mass for a vtkPolyData geometry.

        Arguments:
        geometry - surface mesh to use for calculation.
        """
        center_of_mass_filter = vtkCenterOfMass()
        center_of_mass_filter.SetUseScalarsAsWeights(False)
        center_of_mass_filter.SetInputData(geometry)
        center_of_mass_filter.Update()
        center_of_mass = center_of_mass_filter.GetCenter()

        return array(center_of_mass)

    @classmethod
    def make(
        cls,
        input_data: vtkPolyData,
        direction: ndarray,
        number_of_slices: int,
        center_of_mass: ndarray = None,
        length: float = None,
        center: ndarray = None,
    ) -> Slicer:
        """
        Factory function to create an object from function arguments.

        Arguments:
        input_data - see SlicerConfig documentation
        direction - see SlicerConfig documentation
        number_of_slices - see SlicerConfig documentation

        Optional arguments:
        center_of_mass - see SlicerConfig documentation
        length - see SlicerConfig documentation
        center - see SlicerConfig documentation
        """
        config = SlicerConfig(
            input_data, direction, number_of_slices, center_of_mass, length, center
        )
        return Slicer(config)

@dataclass(frozen=True)
class Cube:
    geometry: vtkPolyData
    center: ndarray
    normal: ndarray
    point: ndarray

class Voxelizer:
    SlicesPerDimension = 3
    NumberOfCubes = 3 * 3 * 3

    def __init__(self, geometry: vtkPolyData, directions: ndarray=None) -> None:
        self.directions: ndarray
        self.axes: ndarray
        self.center_of_mass: ndarray
        self.directions: ndarray
        self._points: ndarray
        self._cubes: Tuple[Cube]

        self.create_cubes(geometry, directions)

    def create_cubes(self, geometry: vtkPolyData, directions: ndarray) -> None:
        self.directions = self.calc_obb(geometry)[1:] if directions is None else directions
        direction_vector_lengths = norm(self.directions, axis=1)
        self.axes = self.directions / direction_vector_lengths[:, newaxis]
        self.center_of_mass = Slicer.calculate_center_of_mass(geometry)

        slicing_data = slice_geometry(geometry, axes=self.axes, number_of_slices=self.SlicesPerDimension)
        geometries, centers = zip(*slicing_data)
        self.directions = self.calculate_directions(self.center_of_mass, targets=centers)
        self._points = array([
            self.calculate_projections(g, origin=self.center_of_mass, direction=d).tolist()
            for g, d in slicing_data
        ])

        self._cubes = tuple(Cube(*parameters) for parameters in zip(
            geometries, centers, self.directions, self._points
        ))

    @property
    def cubes(self) -> Tuple[Cube]:
        return self._cubes

    @property
    def points(self) -> ndarray:
        return self._points

    @classmethod
    def make_normalized(cls, voxelizer: Voxelizer) -> Voxelizer:
        return cls()

    @staticmethod
    def calculate_directions(origin: ndarray, targets: Tuple[ndarray]) -> Tuple[ndarray]:
        vectors = array(targets - origin)
        lengths = norm(vectors, axis=1)
        normals = vectors / lengths[:, newaxis]
        return normals

    @classmethod
    def calculate_projections(cls, geometry: vtkPolyData, origin: ndarray, direction: ndarray) -> ndarray:
        points = cls.extract_valid_vertices(geometry)
        if points.shape[1] == 0:
            return array([0, 0, 0])

        points -= origin
        axis = direction / norm(direction)

        dot_products = points.dot(axis)
        projections = outer(dot_products, axis)
        maximum_projection_id = argmax(dot_products)

        return points[0][maximum_projection_id] + origin

    @staticmethod
    def extract_valid_vertices(geometry: vtkPolyData) -> Tuple[int]:
        polygons: vtkCellArray = geometry.GetPolys()
        point_ids = set()
        new_point_ids = vtkIdList()
        for cell_id in range(polygons.GetNumberOfCells()):
            polygons.GetCellAtId(cell_id, new_point_ids)
            point_ids |= set(new_point_ids.GetId(id) for id in range(new_point_ids.GetNumberOfIds()))

        vertices = list(geometry.GetPoint(id) for id in point_ids)
        return array([vertices])

    @staticmethod
    def calc_obb(geometry: vtkPolyData) -> ndarray:
        """
        Return oriented bounding box as a tuple (corner, vector1, vector2, vector3,).
        """
        obb = zeros((5, 3))
        obb_tree = vtkOBBTree()
        obb_tree.SetDataSet(geometry)
        obb_tree.SetMaxLevel(0)
        obb_tree.BuildLocator()
        obb_tree.ComputeOBB(geometry, *[obb[i] for i in range(5)])
        return array(tuple(obb[i] for i in range(4)))


def slice_geometry(geometry: vtkPolyData, axes: Tuple[ndarray], number_of_slices: int) -> Tuple[vtkPolyData]:
    """
    Slice an vtkPolyData object along multiple axis. Return a tuple of vtkPolyData and numpy array pairs.
    The vtkPolyData represents the sublice and the array represents each respective slice center.

    Arguments:
    geometry - object to slice.
    axes - all axes to along. Cuts are produced in order.
    number_of_slices - refers to the number of slices to be produced per axis.
    """
    def slice_rec(slicer, axes, lengths):
        if len(axes) == 0:
            return tuple(zip(slicer.output, slicer.centers))

        current_axis, *remaining_axes = axes
        current_length, *remaining_lengths = lengths
        slicers = tuple(
            Slicer.make(
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

    slicer = Slicer.make(geometry, direction=axes[0], number_of_slices=number_of_slices)
    lengths = tuple(slicer.calculate_length_in_direction(a) for a in axes)
    return slice_rec(
        slicer,
        axes=axes[1:],
        lengths=lengths[1:]
    )

def demo_1d(geometry: vtkPolyData) -> Tuple[vtkPolyData]:
    " Boring example for slicing along a single axis. "
    return Slicer.make(geometry, direction=array([1,0,0]), number_of_slices=3).output

def demo_3d(geometry: vtkPolyData) -> Tuple[vtkPolyData]:
    " Little more exciting example for subdividing a gemetry in all three dimensions "
    return tuple(sub for sub, _ in slice_geometry(
        geometry,
        axes=(array([1,0,0]), array([0,1,0]), array([0,0,1]),),
        number_of_slices=3
    ))

def demo_poc(geometry: vtkPolyData) -> None:
    tester = Voxelizer(geometry)
