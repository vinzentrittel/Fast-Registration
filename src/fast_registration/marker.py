"""
This module provides an interface for a user to provide landmarks for a given STL geometry.
"""
from csv import DictWriter
from os.path import basename
from re import compile as Regex
from time import sleep
from threading import Timer
from typing import List, Tuple

from PyQt5.QtCore import Qt # pylint: disable=no-name-in-module
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent # pylint: disable=no-name-in-module
from PyQt5.QtWidgets import ( # pylint: disable=no-name-in-module
    QApplication,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)
from vtk import ( # pylint: disable=no-name-in-module
    vtkActor,
    vtkCellArray,
    vtkCellPicker,
    vtkDataArray,
    vtkFloatArray,
    vtkInteractorStyleTrackballCamera,
    vtkLookupTable,
    vtkPoints,
    vtkPolyData,
    vtkPolyDataMapper,
    vtkPolyDataNormals,
    vtkRenderWindowInteractor,
    vtkRenderer,
    vtkTextActor,
    vtkTextRepresentation,
    vtkTextWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .util import (
    calculate_curvature,
    load_markers,
    load_stl,
    n_greatest_values,
    PointMode,
    POINTS_HEADER,
    smooth_normals,
    remesh,
)

class MainWindow(QMainWindow):
    """
    GUI for marking landmarks on an STL mesh.
    """
    # pylint: disable=attribute-defined-outside-init,too-many-instance-attributes
    def __init__(self) -> None:
        super().__init__()

        self._setup_window()
        self._setup_vertex_picking()
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
        """
        Return currently loaded STL mesh as vtkPolyData.
        """
        return self.geometry_mapper.GetInput()

    @geometry.setter
    def geometry(self, new_geometry: vtkPolyData) -> None:
        """
        Updates the STL mesh displayed to a new one.
        """
        normals = vtkPolyDataNormals()
        normals.ComputePointNormalsOn()
        normals.SplittingOff()
        normals.SetInputData(new_geometry)
        new_geometry = smooth_normals(normals.GetOutputPort())

        self.geometry_mapper.SetInputData(new_geometry)
        self.geometry_mapper.SetScalarVisibility(False)

        self.text = f"Loaded {new_geometry.GetNumberOfPoints()} vertices."
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    @property
    def points(self) -> None:
        """
        Return the landmarks of the currently loaded STL mesh as vtkPolyData.
        The set of landmarks returned depends on the currently selected PointMode.
        """
        return self._points[self.current_mode.value]

    @property
    def point_mapper(self) -> None:
        """
        Return the data mapper currently connected to the landmarks of the currently
        loaded STL mesh. The mapper returned depends on the currently selected PointMode.
        """
        return self._point_mappers[self.current_mode.value]

    @property
    def normals(self) -> vtkDataArray:
        """
        Return the normals to the landmakrs of the currently loaded STL mesh as vtkDataArray.
        The set of normals returned depends on the currently selected PointMode.
        """
        return self.points.GetPointData().GetNormals()

    @property
    def current_mode(self) -> PointMode:
        """
        Return the mode currently active in the UI.
        """
        return self._current_mode

    @current_mode.setter
    def current_mode(self, new_mode: PointMode) -> None:
        """
        Set new PointMode to 'new_mode'. The change results in a re-coloring of the previously
        active point set and the point set active after the mode switch.
        """
        self.point_actor.GetProperty().SetColor(0.5, 0.5, 0.5)
        self.point_actor.GetProperty().SetPointSize(9.9)
        self._current_mode = new_mode
        self.point_actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        self.point_actor.GetProperty().SetPointSize(10)

    @property
    def point_actor(self) -> vtkActor:
        """
        Return the visual actor connected to to the landmarks of the currently loaded STL mesh.
        The vtkActor returned depends on the currently selected PointMode.
        """
        return self._point_actors[self.current_mode.value]

    @property
    def text(self) -> str:
        """
        Return the text displayed in the upper left corner.
        """
        return self.text_widget.GetTextActor().GetInput()

    @text.setter
    def text(self, new_text: str) -> None:
        def animate(that_text):
            head = ""
            pause = 0.001 / len(new_text)
            for l in that_text:
                head = head + l
                self.text_widget.GetTextActor().SetInput(head)
                sleep(pause)
                self.vtk_widget.Render()

        Timer(0.0, lambda: animate(new_text)).start()

    def _setup_window(self) -> None:
        """
        Setup UI and connections for this program.
        """
        self.setWindowTitle("Landmark Marker")
        self.resize(800, 600)
        self.setAcceptDrops(True)
        self.left_button_pressed = False
        self.right_button_pressed = False
        self.mouse_position = (-1, -1)
        self._current_mode = PointMode.POI

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.keyReleaseEvent = self.keyReleaseEvent
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.vtk_widget)

        self.poi_mode_button = QPushButton("POI Mode")
        self.poi_mode_button.setCheckable(True)
        self.poi_mode_button.setChecked(True)
        self.poi_mode_button.clicked.connect(lambda: self.toggle_mode(self.poi_mode_button))
        self.handle_mode_button = QPushButton("Scale Handle Mode")
        self.handle_mode_button.setCheckable(True)
        self.handle_mode_button.clicked.connect(lambda: self.toggle_mode(self.handle_mode_button))
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.poi_mode_button)
        button_layout.addWidget(self.handle_mode_button)
        layout.addLayout(button_layout)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.2, 0.4)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

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
        self.text_widget.GetTextActor().SetInput("Drop STL file")
        self.text_widget.On()

    def _setup_vertex_picking(self):
        """
        Initializing everything that has to do with adding new landmarks to an STL mesh.
        """
        picker = vtkCellPicker()
        interactor: vtkRenderWindowInteractor = self.vtk_widget.GetRenderWindow().GetInteractor()
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
            lambda obj, _: self.on_release(obj, picker),
        )

        self._points = []
        self._point_mappers = []
        self._point_actors = []
        for _ in PointMode:
            self._points.append(vtkPolyData())
            self._points[-1].SetVerts(vtkCellArray())
            self._points[-1].SetPoints(vtkPoints())
            normals = vtkFloatArray()
            normals.SetName("Normals")
            normals.SetNumberOfComponents(3)
            self._points[-1].GetPointData().SetNormals(normals)

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

    def on_click(self, event: str, interactor: vtkRenderWindowInteractor) -> None:
        """
        Callback function to register left and right mouse button presses.

        Keyword arguments:
        event - name of the captured event.
        interactor - interaction object, that captured this event.
        """
        if "Left" in event:
            self.left_button_pressed = True
        elif "Right" in event:
            self.right_button_pressed = True
        self.mouse_position = interactor.GetEventPosition()

    def on_release(self, interactor: vtkRenderWindowInteractor, picker: vtkCellPicker) -> None:
        """
        Callback function to handle button clicks.

        Keyword arguments:
        interactor - interaction object, that captured the click.
        picker - picker instance, to assign a click to an vtk object.
        """
        if self.geometry and self.mouse_position == interactor.GetEventPosition():
            picker.Pick(*self.mouse_position, 0, self.renderer)

            if self.left_button_pressed and picker.GetMapper() == self.geometry_mapper:
                point_id = picker.GetPointId()
                self.add_point(
                    new_point=self.geometry.GetPoint(point_id),
                    new_normal=self.geometry.GetPointData().GetNormals().GetTuple(point_id),
                )
            elif self.right_button_pressed                  \
                and self.points.GetNumberOfPoints() > 0     \
            :
                position = picker.GetPickPosition()
                self.points.BuildPointLocator()
                id_to_delete = self.points.GetPointLocator().FindClosestPoint(position)
                self.remove_point(id_to_delete)

        self.left_button_pressed = False
        self.right_button_pressed = False

    def add_point(
        self, new_point: Tuple[float, float, float], new_normal: Tuple[float, float, float]
    ) -> None:
        """
        Insert a new point to the current set of points. The expanded point set depends on the
        currently selected PointMode. New points are stored immediately in a CSV file.

        Keyword arguments:
        new_point - 3D coordinates of a landmark point.
        new_normal - 3D normal vector associated with new landmark point.
        """
        new_point_id = self.points.GetPoints().InsertNextPoint(new_point)
        self.points.GetPointData().GetNormals().InsertTuple3(new_point_id, *new_normal)
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
        """
        Delete point with ID 'point_id' from the current set of points. The referenced point set
        depends on the currently selected PointMode. The updated points are stored immediately in a
        CSV file.

        Keyword arguments:
        point_id - ID of the point to be deleted as stored in the vtkPolyData currently selected.
                   The ID must reference an existing point.
        """
        self.set_points(
            new_points=[
                self.points.GetPoint(id_)
                for id_ in range(self.points.GetNumberOfPoints())
                if id_ != point_id
            ],
            new_normals=[
                self.normals.GetTuple(id_)
                for id_ in range(self.normals.GetNumberOfTuples())
                if id_ != point_id
            ],
        )
        self.write()

    def set_points(
        self,
        new_points: List[Tuple[float, float, float]],
        new_normals: List[Tuple[float, float, float]],
    ) -> None:
        """
        Assign a fresh list of 3D coordinates to the currently selected point set.
        The currently active point set depends on the selected PointMode.

        Keyword arguments:
        new_points - list of 3-tuples filled with floating point numbers, representing 3D
                     coordinates.
        new_normals - list of 3-tuples filled with floating point numbers, representing 3D
                      normal vectors for 'new_points' in order.
        """
        points = vtkPoints()
        verts = vtkCellArray()
        normals = vtkFloatArray()
        normals.SetName("Normals")
        normals.SetNumberOfComponents(3)
        for point, normal in zip(new_points, new_normals):
            point_id = points.InsertNextPoint(point)
            normals.InsertTuple(point_id, normal)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
        self.points.SetPoints(points)
        self.points.SetVerts(verts)
        self.points.GetPoints().Modified()
        self.points.GetVerts().Modified()
        self.points.Modified()
        self.points.GetPointData().SetNormals(normals)
        self.points.Modified()
        self.points.BuildCells()
        self.points.BuildLinks()
        self.point_mapper.Update()
        self.renderer.GetRenderWindow().Render()

    def write(self) -> None:
        """
        Write point sets for all PointModes in a CSV file.
        The CSV file will have the same file name as the previously loaded STL mesh file, but
        with a '.csv' extension as postfix.

        'L1.stl' -> 'L1.stl.csv'
        """
        previous_mode = self.current_mode
        with open(self.filename + ".csv", "w", encoding="utf-8") as point_file:
            csv = DictWriter(point_file, POINTS_HEADER)
            csv.writeheader()
            for mode in PointMode:
                self.current_mode = mode
                for id_ in range(self.points.GetNumberOfPoints()):
                    row = dict(zip(
                        POINTS_HEADER,
                        self.points.GetPoint(id_) + self.normals.GetTuple(id_) + (mode.name,),
                    ))
                    csv.writerow(row)
        self.text = f"Written to '{basename(self.filename) + '.csv'}'"
        self.current_mode = previous_mode

    def read(self) -> None:
        """
        Read all point sets for all PointModes from a CSV file.
        The CSV file should have the same file name as the previously loaded STL mesh file, but
        with a '.csv' extension as postfix.

        'L1.stl' -> 'L1.stl.csv'
        """
        previous_mode = self.current_mode
        for mode in PointMode:
            self.current_mode = mode
            points, normals = load_markers(self.filename + ".csv", mode)
            self.set_points(points, normals)
        self.current_mode = previous_mode
        self.renderer.GetRenderWindow().Render()

    def append_points(
        self,
        new_points: List[Tuple[float, float, float]],
        new_normals: List[Tuple[float, float, float]],
    ) -> None:
        """
        Add a list of 3D coordinates to the currently selected point set.
        The currently active point set depends on the selected PointMode.

        Keyword arguments:
        new_points - list of 3-tuples filled with floating point numbers, representing 3D
        coordinates.
        new_normals - list of 3-tuples filled with floating point numbers, representing 3D
                      normal vectors for 'new_points' in order.
        """
        old_points = [self.points.GetPoint(id_) for id_ in range(self.points.GetNumberOfPoints())]
        old_normals = [self.normals.GetTuple(id_) for id_ in range(self.points.GetNumberOfPoints())]
        self.set_points(old_points + new_points, old_normals + new_normals)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None: # pylint: disable=invalid-name
        """
        Just reject drag'n'drop actions, not involving files.

        Keyword arguments:
        event - object containing more information about the event instantiation.
        """
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent): # pylint: disable=invalid-name
        """
        Load mesh from file or points from .mrk.json file. If you are importing from json, make sure
        the file conforms to the structure in place in 3D slices Markup point fiducial export files.

        Keyword arguments:
        event - object containing more information about the event instantiation.
        """
        if event.mimeData().hasUrls():
            filename, *_ = event.mimeData().urls()
            filename = filename.path()
            if filename.endswith(".stl"):
                self.filename = filename
                self.geometry = load_stl(filename)
                self.read()
                event.accept()
            elif filename.endswith(".mrk.json"):
                regex = Regex(r"""\"position\":\s*\[([^,]+),\s*([^,]+),\s*([^,]+)],""")
                with open(filename, "r", encoding="utf-8") as markups_file:
                    json = "".join(markups_file.readlines())
                positions = [(float(x), float(y), float(z),) for x, y, z in regex.findall(json)]
                normals = [
                    self.geometry.GetPointData().GetNormals().GetTuple(
                        self.geometry.GetPointLocator().FindClosestPoint(position)
                    )
                    for position in positions
                ]
                self.append_points(new_points=positions, new_normals=normals)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent): # pylint: disable=invalid-name
        """
        Hard coded keybindings and there actions.
        """
        if event.key() == Qt.Key_S and not self.geometry is None:
            if self.geometry_mapper.GetScalarVisibility():
                self.geometry_mapper.SetScalarVisibility(False)
                self.renderer.GetRenderWindow().Render()
                return

            curvatures = calculate_curvature(self.geometry)
            scalars = n_greatest_values(curvatures, n=int(self.geometry.GetNumberOfPoints() / 2))
            scalars.SetName("ColorGroups")
            self.geometry.GetPointData().SetScalars(scalars)

            lookup_table = vtkLookupTable()
            lookup_table.SetNumberOfTableValues(2)
            lookup_table.Build()
            lookup_table.SetTableValue(0,  0.9, 0.9, 0.9)
            lookup_table.SetTableValue(1,  1.0, 0.0, 0.0)

            self.geometry_mapper.SetScalarRange(0, 1)
            self.geometry_mapper.SetLookupTable(lookup_table)
            self.geometry_mapper.SetScalarVisibility(True)
            self.renderer.GetRenderWindow().Render()
        elif event.key() == Qt.Key_D and not self.geometry is None:
            # Here is just some random stuff for debugging and displaying
            # WIP data.
            self.geometry = remesh(self.geometry, cluster_count=2000)
        elif event.key() == Qt.Key_Q:
            self.close()

    def toggle_mode(self, caller: QPushButton):
        """
        Switch between the PointModes on a button click.

        Keyword argument:
        caller - the Qt button, that invoked this callback.
        """
        if (
            caller is self.poi_mode_button and self.poi_mode_button.isChecked()
        ) or (
            caller is self.handle_mode_button
            and not self.handle_mode_button.isChecked()
        ):
            self.handle_mode_button.setChecked(False)
            self.poi_mode_button.setChecked(True)
            self.current_mode = PointMode.POI
            self.text = "POI Mode"
        else:
            self.handle_mode_button.setChecked(True)
            self.poi_mode_button.setChecked(False)
            self.current_mode = PointMode.SCALE_HANDLE
            self.text = "Scale Handle Mode"
        self.renderer.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
