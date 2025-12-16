# Installing

First install the packages, as dictated by _requirements.txt_. Then perform the setup via `pip install -e .`.
```
$ python3 -m pip install -r requriements.txt
$ python3 -m pip install -e .
```

# Running

Run as a Python module.

## Landmark Your Atlas
```
$ python3 -m fast_registration.marker
```

When launched, just drag-n-drop your atlas geometries STL file here. You can also drop a 3D Slic3r .mrk.json file for landmarks (or just point and click, to create new ones). If you are finished, wait for the program to write all necessary data to a save file.

## Running The Registration Tool

```
$ python3 -m fast_registration.__init__ path_to_target.stl path_to_source_atlas.stl
```
