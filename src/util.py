from pathlib import Path
from typing import Any

from numpy import multiply, newaxis
from slic3r_display import Slic3rBoxRepresentable
from vtk import vtkPolyData, vtkSTLReader, vtkSTLWriter

def load_stl(filename: str) -> vtkPolyData:
    reader = vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def write(obj: Any, filename: Path) -> None:
    from bounding_box import BoundingBox

    if isinstance(obj, BoundingBox):
        Slic3rBoxRepresentable.write_from(
            origin=Slic3rBoxRepresentable.center_to_origin(obj.center, obj.axes, obj.length),
            axes=multiply(obj.axes, obj.length[:, newaxis]),
            filename=filename,
        )
    elif isinstance(obj, vtkPolyData):
        writer = vtkSTLWriter()
        writer.SetFileName(str(filename))
        writer.SetInputData(obj)
        writer.Update()
    else:
        raise NotImplementedError(type(obj), "has no appropriate write function")
