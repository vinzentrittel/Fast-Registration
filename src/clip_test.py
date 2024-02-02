from enum import IntEnum
from vtk import vtkPlane, vtkSTLWriter, vtkClipPolyData, vtkPolyData

def make_clipper(plane: vtkPlane) -> vtkClipPolyData:
    clipper = vtkClipPolyData()
    clipper.GenerateClippedOutputOn()
    clipper.InsideOutOn()
    clipper.SetClipFunction(plane)
    return clipper

def write_stl(geometry: vtkPolyData, filename: str) -> None:
    writer = vtkSTLWriter()
    writer.SetFileName(filename)
    writer.SetInputData(geometry)
    writer.Update()

from main import load_stl

Vertebra = load_stl("../data/L5_normalized.stl")
LeftPlane = vtkPlane()
LeftPlane.SetOrigin(-1./3., 0, 0)
LeftPlane.SetNormal(1, 0, 0)
RightPlane = vtkPlane()
RightPlane.SetOrigin(1./3., 0, 0)
RightPlane.SetNormal(1, 0, 0)
AnteriorPlane = vtkPlane()
AnteriorPlane.SetOrigin(0, -1./3., 0)
AnteriorPlane.SetNormal(0, 1, 0)
PosteriorPlane = vtkPlane()
PosteriorPlane.SetOrigin(0, 1./3., 0)
PosteriorPlane.SetNormal(0, 1, 0)
InferiorPlane = vtkPlane()
InferiorPlane.SetOrigin(0, 0, -1./3.)
InferiorPlane.SetNormal(0, 0, 1)
SuperiorPlane = vtkPlane()
SuperiorPlane.SetOrigin(0, 0, 1./3.)
SuperiorPlane.SetNormal(0, 0, 1)


class FirstLevel(IntEnum):
    Left = 0
    Right = 1
    Slices = 2

class SecondLevel(IntEnum):
    Left = -1
    Right = 1
    Anterior = 0
    Posterior = 1
    Center = 0
    Slices = 3

class ThirdLevel(IntEnum):
    Left = -1
    Right = 1
    Anterior = -1
    Posterior = 1
    Inferior = 0
    Superior = 1
    Center = 0
    Slices = 3

FirstLvlClip = [make_clipper(p) for p in (LeftPlane, RightPlane,)]
SecondLvlClip = [ FirstLevel.Slices * [None] for _ in range(SecondLevel.Slices) ]
for lcr in range(SecondLevel.Slices):
    for axis, plane in enumerate((AnteriorPlane, PosteriorPlane,)):
        SecondLvlClip[lcr][axis] = make_clipper(plane)

Clip = [[FirstLevel.Slices * [None] for _ in range(SecondLevel.Slices)] for _ in range(ThirdLevel.Slices)]
for frontal in range(ThirdLevel.Slices):
    for axial in range(SecondLevel.Slices):
        for longitudinal, plane in enumerate((InferiorPlane, SuperiorPlane,)):
            Clip[frontal][axial][longitudinal] = make_clipper(plane)

FirstLvlClip[FirstLevel.Left].SetInputData(Vertebra)
FirstLvlClip[FirstLevel.Right].SetInputConnection(
    FirstLvlClip[FirstLevel.Left].GetClippedOutputPort()
)

# Second Level
Left = SecondLevel.Left
Right = SecondLevel.Right
Anterior = SecondLevel.Anterior
Posterior = SecondLevel.Posterior
Center = SecondLevel.Center

for frontal, connection in zip(
    [Left, Center, Right],
    [
        FirstLvlClip[FirstLevel.Left].GetOutputPort(),
        FirstLvlClip[FirstLevel.Right].GetOutputPort(),
        FirstLvlClip[FirstLevel.Right].GetClippedOutputPort(),
    ]
):
    SecondLvlClip[frontal][Anterior].SetInputConnection(connection)
    SecondLvlClip[frontal][Posterior].SetInputConnection(
        SecondLvlClip[frontal][Anterior].GetClippedOutputPort()
    )

# Final level
Anterior = ThirdLevel.Anterior
Posterior = ThirdLevel.Posterior
Inferior = ThirdLevel.Inferior
Superior = ThirdLevel.Superior

for frontal in (Left, Center, Right,):
    for sagittal, connection in zip(
        [Anterior, Center, Posterior],
        [
            SecondLvlClip[frontal][SecondLevel.Anterior].GetOutputPort(),
            SecondLvlClip[frontal][SecondLevel.Posterior].GetOutputPort(),
            SecondLvlClip[frontal][SecondLevel.Posterior].GetClippedOutputPort(),
        ],
    ):
        Clip[frontal][sagittal][Inferior].SetInputConnection(connection)
        Clip[frontal][sagittal][Superior].SetInputConnection(
            Clip[frontal][sagittal][Inferior].GetClippedOutputPort()
        )

from timeit import default_timer
start = default_timer()
for frontal in (Left, Center, Right,):
    for sagittal in (Anterior, Center, Posterior,):
        Clip[frontal][sagittal][Superior].Update()
end = default_timer()
print(end -start)

write_stl(Clip[Center][Center][Superior].GetOutput(), "L5.middle.stl")
