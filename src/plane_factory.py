from enum import auto, IntEnum

from vtk import vtkPlane

class PlaneFactory:
    """
    Helper class to construct vtkPlanes as parallel planes to the
    three body planes - frontal, sagittal, axial.
    """
    class Axis(IntEnum):
        """ Enum to target all body axes and there center. """
        Left = 0
        Right = auto()
        Posterior = auto()
        Anterior = auto()
        Inferior = auto()
        Superior = auto()
        NumberOfAxes = auto()

    @classmethod
    def make_plane(cls, axis: Axis) -> vtkPlane:
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
        if axis == cls.Axis.Left:
            plane.SetOrigin(-1./3., 0, 0)
            plane.SetNormal(1, 0, 0)
        elif axis == cls.Axis.Right:
            plane.SetOrigin(1./3., 0, 0)
            plane.SetNormal(1, 0, 0)
        elif axis == cls.Axis.Posterior:
            plane.SetOrigin(0, -1./3., 0)
            plane.SetNormal(0, 1, 0)
        elif axis == cls.Axis.Anterior:
            plane.SetOrigin(0, 1./3., 0)
            plane.SetNormal(0, 1, 0)
        elif axis == cls.Axis.Inferior:
            plane.SetOrigin(0, 0, -1./3.)
            plane.SetNormal(0, 0, 1)
        elif axis == cls.Axis.Superior:
            plane.SetOrigin(0, 0, 1./3.)
            plane.SetNormal(0, 0, 1)
        return plane

# some harmless globals
LeftPlane = PlaneFactory.Axis.Left
RightPlane = PlaneFactory.Axis.Right
PosteriorPlane = PlaneFactory.Axis.Posterior
AnteriorPlane = PlaneFactory.Axis.Anterior
InferiorPlane = PlaneFactory.Axis.Inferior
SuperiorPlane = PlaneFactory.Axis.Superior
