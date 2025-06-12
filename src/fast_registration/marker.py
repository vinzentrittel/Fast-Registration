from __future__ import annotations

from csv import DictReader, DictWriter
from dataclasses import dataclass, field
from os.path import isfile
from pathlib import Path
from re import compile as Regex
from threading import Timer
from time import sleep
from typing import List, Tuple

from numpy import argmin, zeros
from numpy.linalg import norm
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)
from vtk import (
    vtkActor,
    vtkCellArray,
    vtkCellPicker,
    vtkFloatArray,
    vtkInteractorStyleTrackballCamera,
    vtkPointPicker,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkPolyDataNormals,
    vtkRenderer,
    vtkRenderWindowInteractor,
    vtkTextActor,
    vtkTextRepresentation,
    vtkTextWidget,
)

from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .util import (
    CURVATURE_TYPE,
    load_stl,
    n_greatest_values,
    POINTS_HEADER,
    PointMode,
    smooth_normals,
)
from .alt_identity_clipper import calculate_curved_sections, WEIGHTED_CURVATURE_TYPE

LEFT_BUTTON_PRESS_EVENT = "LeftButtonPressEvent"
RIGHT_BUTTON_PRESS_EVENT = "RightButtonPressEvent"
END_INTERACTION_EVENT = "EndInteractionEvent"

@dataclass
class Landmark:
    point: Tuple[float, float, float] = field(default_factory=tuple)
    normal: Tuple[float, float, float] = field(default_factory=tuple)
    curvature: float = 0.0
    weighted_curvature: float = 0.0
    mode: PointMode = None

    @classmethod
    def at(cls, point_id: int, mesh: vtkPolyData, mode: PointMode) -> Landmark:
        return cls(
            point=mesh.GetPoint(point_id),
            normal=mesh.GetPointData().GetNormals().GetTuple(point_id),
            curvature=mesh.GetPointData().GetAbstractArray(CURVATURE_TYPE).GetTuple1(point_id),
            weighted_curvature=mesh.GetPointData().GetAbstractArray(
                WEIGHTED_CURVATURE_TYPE
            ).GetTuple1(point_id),
            mode=mode,
        )

def _path_to_csv(path: Path) -> Path:
    return Path(path.parent, f"{path.stem}.csv")

def save_landmarks(landmarks: List[Landmark], path: Path) -> None:
    with open(_path_to_csv(path), "w", encoding="utf-8") as csv_file:
        writer = DictWriter(csv_file, fieldnames=POINTS_HEADER)
        writer.writeheader()
        for landmark in landmarks:
            writer.writerow({
                **dict(zip(
                    POINTS_HEADER,
                    (
                        landmark.point
                        + landmark.normal
                        + (landmark.curvature,)
                        + (landmark.weighted_curvature,)
                        + (landmark.mode.name,)
                    )
                ))
            })

def load_landmarks(path: Path) -> List[Landmark]:
    if path.suffix == ".stl":
        path = _path_to_csv(path)
        if not path.is_file():
            return []
        with open(path, "r", encoding="utf-8") as csv_file:
            reader = DictReader(csv_file)
            result = []
            for row in reader:
                point = tuple(map(float, (row["x"], row["y"], row["z"],)))
                normal = tuple(map(float, (row["nx"], row["ny"], row["nz"],)))
                curvature = float(row["c"])
                weighted_curvature = float(row["wc"])
                mode = PointMode[row["kind"]]
                result.append(Landmark(point, normal, curvature, weighted_curvature, mode))
            return result
    elif "".join(path.suffixes) == ".mrk.json":
        regex = Regex(r"""\"position\":\s*\[([^,]+),\s*([^,]+),\s*([^,]+)],""")
        with open(path, "r", encoding="utf-8") as markups_file:
            json = "".join(markups_file.readlines())

        result = []
        for x, y, z in regex.findall(json):
            point = float(x), float(y), float(z)
            result.append(Landmark(
                point,
                normal=(0.0, 0.0, 0.0,),
                curvature=0.0,
                weighted_curvature=0.0,
                mode=PointMode.POI,
            ))
        return result

def load_geometry(path: Path) -> vtkPolyData:
    assert path.suffix == ".stl", f"Attempt to load {path.suffix} as geometry failed"

    mesh = load_stl(path)
    normals = vtkPolyDataNormals()
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.SetInputData(mesh)
    mesh = smooth_normals(normals.GetOutputPort())

    for name in (CURVATURE_TYPE, WEIGHTED_CURVATURE_TYPE,):
        float_array = numpy_to_vtk(zeros(mesh.GetNumberOfPoints()))
        float_array.SetName(name)
        mesh.GetPointData().AddArray(float_array)

    return mesh

def compute_curvature(mesh: vtkPolyData) -> List[Landmark]:
    mask, weighted_curvatures, curvatures = calculate_curved_sections(mesh)
    is_significant = n_greatest_values(curvatures, n=int(mesh.GetNumberOfPoints() / 2))

    landmarks = []
    for id_ in range(mesh.GetNumberOfPoints()):
        if mask.GetTuple1(id_) and is_significant.GetTuple1(id_):
            point = mesh.GetPoint(id_)
            normal = mesh.GetPointData().GetNormals().GetTuple(id_)
            landmarks.append(
                Landmark(
                    point,
                    normal,
                    curvatures.GetTuple1(id_),
                    weighted_curvatures.GetTuple1(id_),
                    PointMode.SCALE_HANDLE,
                )
            )
    return landmarks

class CurvatureWorker(QThread):
    finished = pyqtSignal(list)

    def __init__( self, mesh: vtkPolyData):
        super().__init__()
        self.mesh = mesh

    def run(self):
        landmarks = compute_curvature(self.mesh)
        self.finished.emit(landmarks)

class LandmarksManager:
    def __init__(self):
        self.landmarks: List[Landmark] = []
        self.landmarks_representation = vtkPolyData()
        self.set_all([])

    def add(self, landmark: Landmark) -> None:
        self.landmarks.append(landmark)

        point_id = self.landmarks_representation.GetPoints().InsertNextPoint(landmark.point)
        self.landmarks_representation.GetPointData().GetNormals().InsertTuple(point_id, landmark.normal)
        self.landmarks_representation.GetPointData().GetAbstractArray(CURVATURE_TYPE).InsertTuple1(
            point_id, landmark.curvature
        )
        self.landmarks_representation.GetPointData().GetAbstractArray(WEIGHTED_CURVATURE_TYPE).InsertTuple1(
            point_id, landmark.weighted_curvature
        )
        self.landmarks_representation.GetVerts().InsertNextCell(1)
        self.landmarks_representation.GetVerts().InsertCellPoint(point_id)
        self.landmarks_representation.GetVerts().Modified()
        self.landmarks_representation.GetPoints().Modified()
        self.landmarks_representation.Modified()
        self.landmarks_representation.BuildCells()
        self.landmarks_representation.BuildLinks()

    def remove(self, index: int) -> None:
        self.landmarks.pop(index)
        self.set_all(self.landmarks)

    def remove_by_location(self, location: Tuple[float, float, float]) -> None:
        if len(self.landmarks) == 0:
            return
        self.landmarks_representation.BuildPointLocator()
        point_id = self.landmarks_representation.GetPointLocator().FindClosestPoint(location)
        self.remove(point_id)

    def set_all(self, landmarks: List[Landmark]) -> None:
        self.landmarks = landmarks.copy()
        points = vtkPoints()
        cells = vtkCellArray()
        normals = self.make_array("Normals", number_of_components=3)
        curvatures = self.make_array(CURVATURE_TYPE)
        weighted_curvatures = self.make_array(WEIGHTED_CURVATURE_TYPE)

        for landmark in landmarks:
            point_id = points.InsertNextPoint(landmark.point)
            normals.InsertTuple(point_id, landmark.normal)
            curvatures.InsertTuple1(point_id, landmark.curvature)
            weighted_curvatures.InsertTuple1(point_id, landmark.weighted_curvature)
            cells.InsertNextCell(1)
            cells.InsertCellPoint(point_id)

        self.landmarks_representation.SetPoints(points)
        self.landmarks_representation.SetVerts(cells)
        self.landmarks_representation.Modified()
        self.landmarks_representation.GetPointData().SetNormals(normals)
        self.landmarks_representation.GetPointData().AddArray(curvatures)
        self.landmarks_representation.GetPointData().AddArray(weighted_curvatures)
        self.landmarks_representation.Modified()
        self.landmarks_representation.BuildCells()
        self.landmarks_representation.BuildLinks()

    def update(self, mesh: vtkPolyData) -> None:
        mesh.BuildPointLocator()
        locator = mesh.GetPointLocator()
        poi_info = [
            Landmark.at(
                point_id=locator.FindClosestPoint(l.point),
                mesh=mesh,
                mode=l.mode
            )
            for l in self.landmarks
        ]
        detailed_poi = [
            Landmark(l.point, i.normal, i.curvature, i.weighted_curvature, l.mode)
            for l, i in zip(self.landmarks, poi_info)
        ]
        self.set_all(detailed_poi)

    @staticmethod
    def make_array(name: str, number_of_components: int=1) -> vtkFloatArray:
        new_array = vtkFloatArray()
        new_array.SetName(name)
        new_array.SetNumberOfComponents(number_of_components)
        return new_array

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh: vtkPolyData = None
        self.path = Path()
        self.landmarks_manager = LandmarksManager()
        self.poi_manager = LandmarksManager()
        self.mouse_position = -1, -1
        self.latest_event = None

        self._setup_ui()

        self.render()
        self.vtk_widget.Start()

    def render(self) -> None:
        self.vtk_widget.GetRenderWindow().Render()

    def _setup_ui(self):
        self.setWindowTitle("Landmark Marker")
        self.resize(800, 600)
        self.setAcceptDrops(True)
        central = QWidget()

        # TODO: eventually adding interaction elements
        self.setCentralWidget(central)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(central)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk_widget.keyReleaseEvent = self.keyReleaseEvent

        self.text_widget = vtkTextWidget()
        self.text_widget.SetRepresentation(vtkTextRepresentation())
        self.text_widget.GetRepresentation().GetPositionCoordinate().SetValue(0.01, 0.955)
        self.text_widget.GetRepresentation().GetPosition2Coordinate().SetValue(0.99, 0.025)
        self.text_widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.text_widget.SetTextActor(vtkTextActor())
        self.text_widget.GetTextActor().GetTextProperty().SetColor(0.9, 0.9, 0.9)
        self.text_widget.GetTextActor().GetTextProperty().SetJustificationToLeft()
        self.text_widget.ResizableOff()
        self.text_widget.SelectableOff()
        self.text_widget.ProcessEventsOff()
        self.text_widget.GetBorderRepresentation().SetShowBorderToOff()
        self.text_widget.On()
        self.text = "Drop STL file"

        self.mesh_mapper, _ = self.make_actor_mapper_pair()
        self.mesh_mapper.SetScalarVisibility(False)

        self.landmarks_mapper, landmarks_actor = self.make_actor_mapper_pair()
        landmarks_property = landmarks_actor.GetProperty()
        landmarks_property.SetPointSize(10)
        landmarks_property.SetColor(0.5, 0.5, 0.5)
        landmarks_property.RenderPointsAsSpheresOn()
        self.landmarks_mapper.SetInputData(self.landmarks_manager.landmarks_representation)

        self.poi_mapper, poi_actor = self.make_actor_mapper_pair()
        poi_property = poi_actor.GetProperty()
        poi_property.SetPointSize(10)
        poi_property.SetColor(1.0, 0.5, 0.5)
        poi_property.RenderPointsAsSpheresOn()
        self.poi_mapper.SetInputData(self.poi_manager.landmarks_representation)

        picker = vtkCellPicker()
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.SetPicker(picker)
        interactor.AddObserver(LEFT_BUTTON_PRESS_EVENT, self.on_event)
        interactor.AddObserver(RIGHT_BUTTON_PRESS_EVENT, self.on_event)
        interactor.AddObserver(END_INTERACTION_EVENT, self.on_event)
        interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())

    @property
    def text(self) -> str:
        return self.text_widget.GetTextActor().GetInput()

    @text.setter
    def text(self, text: str) -> None:
        def animate(that_text):
            head = ""
            pause = 0.001 / len(text)
            for l in that_text:
                head = head + l
                self.text_widget.GetTextActor().SetInput(head)
                sleep(pause)
                self.vtk_widget.Render()

        Timer(0.0, lambda: animate(text)).start()

    def on_event(self, interactor: vtkRenderWindowInteractor, event: str) -> None:
        if (
            event == END_INTERACTION_EVENT
            and self.mouse_position == interactor.GetEventPosition()
        ):
            if self.latest_event == LEFT_BUTTON_PRESS_EVENT:
                self.pick(interactor)
            elif self.latest_event == RIGHT_BUTTON_PRESS_EVENT:
                self.unpick(interactor)

        self.mouse_position = interactor.GetEventPosition()
        self.latest_event = event

    def pick(self, interactor: vtkRenderWindowInteractor) -> None:
        picker = interactor.GetPicker()
        position = interactor.GetEventPosition()

        picker.Pick(*position, 0, self.renderer)
        if picker.GetMapper() is not self.mesh_mapper:
            return
        landmark = Landmark.at(picker.GetPointId(), mesh=self.mesh, mode=PointMode.POI)
        self.poi_manager.add(landmark)
        self.update_landmarks(write=True)

    def unpick(self, interactor: vtkRenderWindowInteractor) -> None:
        picker = interactor.GetPicker()
        position = interactor.GetEventPosition()

        picker.Pick(*position, 0, self.renderer)
        self.poi_manager.remove_by_location(picker.GetPickPosition())
        self.update_landmarks(write=True)

    def update_landmarks(self, write: bool=False) -> None:
        self.poi_mapper.Update()
        self.landmarks_mapper.Update()
        self.render()
        if not write:
            return
        save_landmarks(
            self.landmarks_manager.landmarks + self.poi_manager.landmarks,
            self.path,
        )
        self.text = f"New landmarks at '{self.path.stem}.csv'"

    def make_actor_mapper_pair(self) -> vtkPolyDataMapper:
        new_mapper = vtkPolyDataMapper()
        new_actor = vtkActor()
        new_actor.SetMapper(new_mapper)
        self.renderer.AddActor(new_actor)
        return new_mapper, new_actor

    def load_geometry(self, path: Path) -> None:
        # cleanup
        self.landmarks_manager.set_all([])
        self.poi_manager.set_all([])

        # load geometry
        self.path = path
        self.mesh = load_geometry(self.path)
        self.mesh_mapper.SetInputData(self.mesh)

        # load landmarks
        existing_landmarks = load_landmarks(self.path)
        if existing_landmarks:
            self.landmarks_manager.set_all(
                [l for l in existing_landmarks if l.mode is PointMode.SCALE_HANDLE]
            )
            self.poi_manager.set_all([l for l in existing_landmarks if l.mode is PointMode.POI])
            self.text = f"Now inspecting '{self.path.name}'"
        else:
            self.text = "Generating new landmarks..."
            def update(landmarks):
                self.landmarks_manager.set_all(landmarks)
                self.poi_manager.update(self.mesh)
                self.update_landmarks(write=True)
            self.worker = CurvatureWorker(self.mesh)
            self.worker.finished.connect(update)
            self.worker.start()

        self.update_landmarks()
        self.renderer.ResetCamera()
        self.render()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return

        path = Path(urls[0].toLocalFile())
        if path.suffix == ".stl":
            self.load_geometry(path)
        elif "".join(path.suffixes) == ".mrk.json":
            def update(_):
                self.poi_manager.update(self.mesh)
                self.update_landmarks(write=True)

            self.poi_manager.set_all(load_landmarks(path))
            self.update_landmarks()
            self.worker = CurvatureWorker(self.mesh)
            self.worker.finished.connect(update)
            self.worker.start()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Q:
            self.close()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
