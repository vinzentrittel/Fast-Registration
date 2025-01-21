"""
This module provides an interface for a user to provide landmarks for a given STL geometry.
"""
from csv import DictReader, DictWriter
from enum import auto, Enum
from os.path import isfile
from pathlib import Path
from pdb import set_trace as break_
from re import compile as Regex
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk import (
    vtkActor,
    vtkCellArray,
    vtkCellPicker,
    vtkInteractorStyleTrackballCamera,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkIdList,
)

from .util import load_points, load_stl, PointMode, POINTS_HEADER

class MainWindow(QMainWindow):
    # pylint: disable=attribute-defined-outside-init
    def __init__(self):
        super().__init__()

        self.setup_window()
        self.setup_vertex_picking()
        self.filename = ""

        # add elements for geometry display
        self.geometry_mapper = vtkPolyDataMapper()
        geometry_actor = vtkActor()
        geometry_actor.SetMapper(self.geometry_mapper)
        self.renderer.AddActor(geometry_actor)

        self.vtk_widget.GetRenderWindow().Render()
        self.vtk_widget.Start()

    @property
    def geometry(self) -> vtkPolyData:
        return self.geometry_mapper.GetInput()

    @geometry.setter
    def geometry(self, new_geometry: vtkPolyData) -> None:
        self.geometry_mapper.SetInputData(new_geometry)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    @property
    def points(self):
        return self._points[self.current_mode.value]

    @property
    def point_mapper(self):
        return self._point_mappers[self.current_mode.value]

    @property
    def current_mode(self) -> PointMode:
        return self._current_mode

    @current_mode.setter
    def current_mode(self, new_mode: PointMode) -> None:
        self.point_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        self._current_mode = new_mode
        self.point_actor.GetProperty().SetColor(1.0, 0.0, 0.0)

    @property
    def point_actor(self) -> vtkActor:
        return self._point_actors[self.current_mode.value]

    def setup_window(self) -> None:
        self.setWindowTitle("VTK with PyQt: Mode Toggle Button")
        self.resize(800, 600)
        self.setAcceptDrops(True)
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.mouse_position = (-1, -1)
        self._current_mode = PointMode.POI

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.vtk_widget)

        self.poi_mode_button = QPushButton("POI Mode")
        self.poi_mode_button.setCheckable(True)
        self.poi_mode_button.setChecked(True)
        self.poi_mode_button.clicked.connect(lambda: self.toggle_mode(self.poi_mode_button))
        #layout.addWidget(self.poi_mode_button)
        self.handle_mode_button = QPushButton("Scale Handle Mode")
        self.handle_mode_button.setCheckable(True)
        self.handle_mode_button.clicked.connect(lambda: self.toggle_mode(self.handle_mode_button))
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.poi_mode_button)
        button_layout.addWidget(self.handle_mode_button)
        layout.addLayout(button_layout)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)  # Background color
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

    def setup_vertex_picking(self):
        picker = vtkCellPicker()
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        interactor.SetPicker(picker)
        interactor.AddObserver("LeftButtonPressEvent", lambda _, event: self.on_click(
            event,
            interactor,
        ))
        interactor.AddObserver("RightButtonPressEvent", lambda _, event: self.on_click(
            event,
            interactor,
        ))
        interactor.AddObserver(
            "EndInteractionEvent",
            lambda obj, event: self.on_release(obj, event, picker),
        )

        self._points = []
        self._point_mappers = []
        self._point_actors = []
        for mode in PointMode:
            self._points.append(vtkPolyData())
            self._points[-1].SetVerts(vtkCellArray())
            self._points[-1].SetPoints(vtkPoints())

            self._point_mappers.append(vtkPolyDataMapper())
            self._point_mappers[-1].SetInputData(self._points[-1])

            self._point_actors.append(vtkActor())
            self._point_actors[-1].SetMapper(self._point_mappers[-1])
            self._point_actors[-1].GetProperty().SetPointSize(10)
            self._point_actors[-1].GetProperty().SetColor(0.5, 0.5, 0.5)
            self._point_actors[-1].GetProperty().RenderPointsAsSpheresOn()
            self.renderer.AddActor(self._point_actors[-1])

        self.current_mode = PointMode.POI
        self.renderer.GetRenderWindow().Render()

    def on_click(self, event, interactor):
        if "Left" in event:
            self.left_button_pressed = True
        elif "Right" in event:
            self.right_button_pressed = True
        self.mouse_position = interactor.GetEventPosition()

    def on_release(self, obj, event, picker):
        if self.geometry and self.mouse_position == obj.GetEventPosition():
            picker.Pick(*self.mouse_position, 0, self.renderer)

            if self.left_button_pressed and picker.GetMapper() == self.geometry_mapper:
                point_id = picker.GetPointId()
                self.add_point(self.geometry.GetPoint(point_id))
            elif self.right_button_pressed                  \
                and self.points.GetNumberOfPoints() > 0     \
            :
                position = picker.GetPickPosition()
                self.points.BuildPointLocator()
                id_to_delete = self.points.GetPointLocator().FindClosestPoint(position)
                self.remove_point(id_to_delete)

        self.left_button_pressed = False
        self.right_button_pressed = False

    def add_point(self, new_point: Tuple[float, float, float]) -> None:
        new_point_id = self.points.GetPoints().InsertNextPoint(new_point)
        self.points.GetVerts().InsertNextCell(1)
        self.points.GetVerts().InsertCellPoint(new_point_id)
        self.points.GetVerts().Modified()
        self.points.GetPoints().Modified()
        self.points.Modified()
        self.points.BuildCells()
        self.points.BuildLinks()
        self.point_mapper.Update()
        self.renderer.GetRenderWindow().Render()
        self.write()

    def remove_point(self, point_id: int) -> None:
        self.set_points([
            self.points.GetPoint(id_)
            for id_ in range(self.points.GetNumberOfPoints())
            if id_ != point_id
            ])
        self.write()

    def set_points(self, points: List[Tuple[float, float, float]]) -> None:
        new_points = vtkPoints()
        new_verts = vtkCellArray()
        for point in points:
            point_id = new_points.InsertNextPoint(point)
            new_verts.InsertNextCell(1)
            new_verts.InsertCellPoint(point_id)
        self.points.SetPoints(new_points)
        self.points.SetVerts(new_verts)
        self.points.GetPoints().Modified()
        self.points.GetVerts().Modified()
        self.points.Modified()
        self.points.BuildCells()
        self.points.BuildLinks()
        self.point_mapper.Update()
        self.renderer.GetRenderWindow().Render()

    def write(self) -> None:
        previous_mode = self.current_mode
        with open(self.filename + ".csv", "w", encoding="utf-8") as point_file:
            csv = DictWriter(point_file, POINTS_HEADER)
            csv.writeheader()
            for mode in PointMode:
                self.current_mode = mode
                for id_ in range(self.points.GetNumberOfPoints()):
                    row = dict(zip(POINTS_HEADER, self.points.GetPoint(id_) + (mode.name,)))
                    csv.writerow(row)
        self.current_mode = previous_mode

    def read(self) -> None:
        previous_mode = self.current_mode
        for mode in PointMode:
            self.current_mode = mode
            self.set_points(load_points(self.filename + ".csv", mode))
        self.current_mode = previous_mode
        self.renderer.GetRenderWindow().Render()

    def append_points(self, new_points: List[Tuple[float, float, float]]) -> None:
        old_points = [self.points.GetPoint(id_) for id_ in range(self.points.GetNumberOfPoints())]
        self.set_points(old_points + new_points)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            filename, *_ = event.mimeData().urls()
            filename = filename.path()
            if filename.endswith(".stl"):
                self.filename = filename
                self.geometry = load_stl(filename)
                self.read()
                event.accept()
            elif filename.endswith(".mrk.json"):
                regex = Regex("""\"position\":\s*\[([^,]+),\s*([^,]+),\s*([^,]+)],""")
                with open(filename, "r", encoding="utf-8") as markups_file:
                    json = "".join(markups_file.readlines())
                positions = [(float(x), float(y), float(z),) for x, y, z in regex.findall(json)]
                self.append_points(positions)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def toggle_mode(self, caller):
        # Toggle between the two modes
        if (
            caller is self.poi_mode_button and self.poi_mode_button.isChecked()
        ) or (
            caller is self.handle_mode_button
            and not self.handle_mode_button.isChecked()
        ):
            self.handle_mode_button.setChecked(False)
            self.poi_mode_button.setChecked(True)
            self.current_mode = PointMode.POI
        else:
            self.handle_mode_button.setChecked(True)
            self.poi_mode_button.setChecked(False)
            self.current_mode = PointMode.SCALE_HANDLE
        self.renderer.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
