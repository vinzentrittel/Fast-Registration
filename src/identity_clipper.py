from enum import IntEnum
from typing import List, Tuple
from vtk import (
    vtkAlgorithmOutput,
    vtkClipPolyData,
    vtkPlane,
    vtkPolyData,
    vtkSTLWriter,
)

from plane_factory import (
    AnteriorPlane,
    InferiorPlane,
    LeftPlane,
    PlaneFactory,
    PosteriorPlane,
    RightPlane,
    SuperiorPlane,
)

class Axis(IntEnum):
    """ Enum to target all body axes and there center. """
    Center = 0
    Left = -1
    Right = 1
    Posterior = -1
    Anterior = 1
    Inferior = -1
    Superior = 1

class IdentityClipper:
    """
    Splits an identity cube of widths 2 with it's origin in (0, 0, 0).
    Inside that cube, any vtkPolyData geometry can be positioned.
    The underlying cube is then split in 27 equally sized parts.
    The assigned vtkPolyData is split at the same planes as the identity cube.

    Usage:
    my_clipper = IdentityClipper()
    my_clipper.input_data = vtkPolyData() # makes little sense, better provide
                                          # a populated vtkPolyData object
    subsection = my_clipper(Axis.Left, Axis.Posterior, Axis.Inferior)
    """
    class FrontalLvlAxis(IntEnum):
        " Indices for first level split "
        Left = 0
        Right = 1
        Slices = 2

    class SagittalLvlAxis(IntEnum):
        " Indices for second level split "
        Left = -1
        Right = 1
        Posterior = 0
        Anterior = 1
        Center = 0
        Slices = 3

    class LongitudinalLvlAxis(IntEnum):
        " Indices for third level split "
        Left = -1
        Right = 1
        Posterior = -1
        Anterior = 1
        Inferior = 0
        Superior = 1
        Center = 0
        Slices = 3

    def __init__(self):
        " Constructor "
        NumberOfCuttingPlanes = 6
        self.base_clipper: vtkClipPolyData    
        self.subclippers: List[List[List[vtkClipPolyData]]]
        self.planes = tuple(PlaneFactory.make_plane(axis) for axis in range(NumberOfCuttingPlanes))
        self.setup_subclippers()

    def setup_subclippers(self) -> None:
        """
        A construct is setup. That construct consists of two plane at -0.33.. and 0.33.. per dimension. Their normals
        are identical to the world axis.
        The VTK pipeline dictates 18 vtkClipPolyData objects to represent all neccessary cuts.
        """
        # create clippers
        frontal_clippers = self.make_frontal_clippers()
        sagittal_clippers = self.make_sagittal_clippers()
        longitudinal_clippers = self.make_longitudinal_clippers()

        # setup clippers
        frontal_clippers[self.FrontalLvlAxis.Right].SetInputConnection(
            frontal_clippers[self.FrontalLvlAxis.Left].GetClippedOutputPort()
        )
        self.setup_sagittal_clippers(
            sagittal_clippers, frontal_clippers=frontal_clippers
        )
        self.setup_longitudinal_clippers(
            longitudinal_clippers, sagittal_clippers=sagittal_clippers
        )

        # store clippers
        self.base_clipper = frontal_clippers[self.FrontalLvlAxis.Left]
        self.subclippers = longitudinal_clippers

    @property
    def input_data(self) -> vtkPolyData:
        " Returns the current vtkPolyData stored for dissection. "
        return self.base_clipper.GetInput()

    @input_data.setter
    def input_data(self, geometry: vtkPolyData) -> None:
        """
        Assign a vtkPolyData for dissection. By assigning it, the dissection is
        performed automatically, immediately after.
        """
        self.base_clipper.SetInputData(geometry)
        self.update()

    def update(self) -> None:
        """
        Updates all necessary end point of all VTK objects.

        When assigning a new vtkPolyData as 'input_data', this procedure
        is called automatically. If that object is modified, you may need
        to manually call update() again.
        """
        for frontal in (self.LongitudinalLvlAxis.Left, self.LongitudinalLvlAxis.Center, self.LongitudinalLvlAxis.Right,):
            for sagittal in (self.LongitudinalLvlAxis.Anterior, self.LongitudinalLvlAxis.Center, self.LongitudinalLvlAxis.Posterior,):
                self.subclippers[frontal][sagittal][self.LongitudinalLvlAxis.Superior].Update()

    def __call__(self, frontal: Axis, sagittal: Axis, longitudinal: Axis) -> vtkPolyData:
        """
        Return the subsection described by the 3 axes values.

        Arguments:
        frontal - left, right or center subsection - see Axis enum (-1, 1, 0)
        sagittal - posterior, anterior or center subsection - see Axis enum (-1, 1, 0)
        longitudinal - inferior, superior or center subsection - see Axis enum (-1, 1, 0)

        Usage:
        identity_clipper(Axis.Center, Axis.Center, Axis.Inferior)
        """
        clippers = self.subclippers[frontal][sagittal]
        if longitudinal == Axis.Inferior:
            return clippers[self.LongitudinalLvlAxis.Inferior].GetOutput()
        if longitudinal == Axis.Center:
            return clippers[self.LongitudinalLvlAxis.Superior].GetOutput()
        if longitudinal == Axis.Superior:
            return clippers[self.LongitudinalLvlAxis.Superior].GetClippedOutput()


    @staticmethod
    def make_clipper(plane: vtkPlane) -> vtkClipPolyData:
        """
        Return a clipper, that returns the part behind the plane as GetOutput().
        """
        clipper = vtkClipPolyData()
        clipper.GenerateClippedOutputOn()
        clipper.InsideOutOn()
        clipper.SetClipFunction(plane)
        return clipper

    def make_frontal_clippers(self) -> Tuple[vtkClipPolyData]:
        """
        Return a tuple of clippers for left and right cutting plane.
        """
        return tuple(
            self.make_clipper(p)
            for p
            in (self.planes[LeftPlane], self.planes[RightPlane],)
        )

    def make_sagittal_clippers(self) -> List[List[vtkClipPolyData]]:
        """
        Return a nested list of clippers for posterior and anterior cutting plane.
        The are proviced for each left, center and right frontal section.
        """
        sagittal_clippers = [2 * [None] for _ in range(3)]
        for nested_sagittal_clippers in sagittal_clippers:
            nested_sagittal_clippers[self.SagittalLvlAxis.Posterior] = self.make_clipper(self.planes[PosteriorPlane])
            nested_sagittal_clippers[self.SagittalLvlAxis.Anterior] = self.make_clipper(self.planes[AnteriorPlane])
        return sagittal_clippers

    def make_longitudinal_clippers(self) -> List[List[List[vtkClipPolyData]]]:
        """
        Return a nested list of clippers for inferior and superior cutting plane.
        The are proviced for each left, frontal center, right as well as posterior,
        sagittal center and anterior section.
        """
        longitudinal_clippers = [[2 * [None] for _ in range(3)] for _ in range(3)]
        for nested_longitudinal_clippers in longitudinal_clippers:
            for doubly_nested_longitudinal_clippers in nested_longitudinal_clippers:
                doubly_nested_longitudinal_clippers[self.LongitudinalLvlAxis.Inferior] = self.make_clipper(self.planes[InferiorPlane])
                doubly_nested_longitudinal_clippers[self.LongitudinalLvlAxis.Superior] = self.make_clipper(self.planes[SuperiorPlane])
        return longitudinal_clippers


    @classmethod
    def setup_nested_clippers(
        cls,
        clippers: Tuple[Tuple[vtkClipPolyData]],
        positions: Tuple[int],
        parents: Tuple[vtkClipPolyData],
    ) -> None:
        """
        Connect nested clipper output ports to another's input connections.
        """
        for position, connection in zip(
            positions,
            cls.generate_ports(*parents),
        ):
            nested_clippers = clippers[position]
            nested_clippers[0].SetInputConnection(connection)
            nested_clippers[1].SetInputConnection(
                nested_clippers[0].GetClippedOutputPort()
            )

    @classmethod
    def setup_sagittal_clippers(
        cls,
        clippers: Tuple[Tuple[vtkClipPolyData]],
        frontal_clippers: Tuple[vtkClipPolyData],
    ) -> None:
        """
        Connect nested clipper output ports to another's input connections.
        """
        cls.setup_nested_clippers(
            clippers,
            positions=(
                cls.SagittalLvlAxis.Left,
                cls.SagittalLvlAxis.Center,
                cls.SagittalLvlAxis.Right,
            ),
            parents=(
                frontal_clippers[cls.FrontalLvlAxis.Left],
                frontal_clippers[cls.FrontalLvlAxis.Right],
            ),
        )


    @classmethod
    def setup_longitudinal_clippers(
        cls,
        longitudinal_clippers: Tuple[Tuple[Tuple[vtkClipPolyData]]],
        sagittal_clippers: Tuple[Tuple[vtkClipPolyData]],
    ):
        """
        Connect nested clipper output ports to another's input connections.
        """
        frontal_positions = (
            cls.LongitudinalLvlAxis.Left,
            cls.LongitudinalLvlAxis.Center,
            cls.LongitudinalLvlAxis.Right,
        )
        sagittal_positions = (
            cls.LongitudinalLvlAxis.Posterior,
            cls.LongitudinalLvlAxis.Center,
            cls.LongitudinalLvlAxis.Anterior,
        )
        for position in frontal_positions:
            nested_sagittal_clippers = sagittal_clippers[position]
            cls.setup_nested_clippers(
                longitudinal_clippers[position],
                positions=sagittal_positions,
                parents=(
                    nested_sagittal_clippers[cls.FrontalLvlAxis.Left],
                    nested_sagittal_clippers[cls.FrontalLvlAxis.Right],
                ),
            )

    @staticmethod
    def generate_ports(*clippers: Tuple[vtkClipPolyData]) -> Tuple[vtkAlgorithmOutput]:
        """
        Connect nested clipper output ports to another's input connections.
        """
        return tuple(c.GetOutputPort() for c in clippers) + (clippers[-1].GetClippedOutputPort(),)

def demo() -> vtkPolyData:
    " Stupid little demo "
    from main import load_stl

    Clipper = IdentityClipper()
    Clipper.input_data = load_stl("../data/L5_normalized.stl")

    return Clipper(Axis.Center, Axis.Center, Axis.Center)

if __name__ == "__main__":
    demo()
