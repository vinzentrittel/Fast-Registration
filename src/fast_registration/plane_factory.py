from vtk import vtkPlane

from .axis import AxisIndex

def make_plane(axis: AxisIndex) -> vtkPlane:
    """
    Return a ready to use cutting plane.
    A plane's origin will always reside at one of the body axes
    - frontal, sagittal, longitudinal. It's normal will face
    the same direction as the respective body axis:

    Left/Right - positive frontal axis direction
    Posterior/Anterior - positive sagittal axis direction
    Inferior/Superior - positive longitudinal axis direction

    Parameter:
    axis - define, where the planes origin will lie
    """
    plane = vtkPlane()
    if axis == AxisIndex.Left:
        plane.SetOrigin(-1./3., 0, 0)
        plane.SetNormal(1, 0, 0)
    elif axis == AxisIndex.Right:
        plane.SetOrigin(1./3., 0, 0)
        plane.SetNormal(1, 0, 0)
    elif axis == AxisIndex.Posterior:
        plane.SetOrigin(0, -1./3., 0)
        plane.SetNormal(0, 1, 0)
    elif axis == AxisIndex.Anterior:
        plane.SetOrigin(0, 1./3., 0)
        plane.SetNormal(0, 1, 0)
    elif axis == AxisIndex.Inferior:
        plane.SetOrigin(0, 0, -1./3.)
        plane.SetNormal(0, 0, 1)
    elif axis == AxisIndex.Superior:
        plane.SetOrigin(0, 0, 1./3.)
        plane.SetNormal(0, 0, 1)
    return plane

# some harmless globals
LeftPlane = AxisIndex.Left
RightPlane = AxisIndex.Right
PosteriorPlane = AxisIndex.Posterior
AnteriorPlane = AxisIndex.Anterior
InferiorPlane = AxisIndex.Inferior
SuperiorPlane = AxisIndex.Superior
