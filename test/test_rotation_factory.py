import pytest
from rotation_factory import RotationFactory

from numpy import array, all, newaxis


def test_correspondence_from_rotations():
    test_vertex = array([-1, -1, -1])
    test_vertices = array([[-1, -1, -1]] + 26 * [[0, 0, 0]])
    if not all(test_vertices.shape == (27, 3,)):
        raise ValueError("Test code failed")
    
    correspondences_after_rotation = RotationFactory.correspondence_from_rotations(
        test_vertices
    )

    assert correspondences_after_rotation != pytest.approx(
        RotationFactory._rotate(test_vertices)
    )

    print("No rotation")
    assert all(correspondences_after_rotation[0][0] == test_vertex)

    print("Rotate 90° around Z")
    # 18th subcube (axis.py to_index(array([1, -1, -1])))
    assert correspondences_after_rotation[1][18] == pytest.approx(array([1, -1, -1]))

    print("Rotate 90° around Y")
    # 2nd subcube (axis.py: to_index(array([-1, -1, 1])))
    assert correspondences_after_rotation[4][2] == pytest.approx(array([-1, -1, 1]))

    print("Rotate 90° around X")
    # 6th subcube (axis.py: to_index(array([-1, 1, -1])))
    assert correspondences_after_rotation[16][6] == pytest.approx(array([-1, 1, -1]))

def test__rotate():
    some_points = array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ])
    possible_rotations = RotationFactory._rotate(some_points)
    assert possible_rotations.shape == (24,) + some_points.shape
    assert all(possible_rotations[0] == some_points)

    easy_points = array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    possible_rotations = RotationFactory._rotate(easy_points)

    print("Rotate 90° around Z")
    assert possible_rotations[1] == pytest.approx(array([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1],
    ]))

    print("Rotate 90° around Y")
    assert possible_rotations[4] == pytest.approx(array([
        [0, 0, -1],
        [0, 1,  0],
        [1, 0,  0],
    ]))

    print("Rotate 90° around X")
    assert possible_rotations[16] == pytest.approx(array([
        [1,  0, 0],
        [0,  0, 1],
        [0, -1, 0],
    ]))
    
def test__make_correspondence_order():
    test_vertices = array([[
        [-1, -1, -1],
        [-1, -1,  0],
        [-1, -1,  1],
        [ 0,  0,  0],
        [ 1,  1, -1],
        [ 1,  1,  0],
        [ 1,  1,  1],
    ]])

    expected_order = array([[0, 1, 2, 3, 4, 5, 6]])
    correspondence_order = RotationFactory._make_correspondence_order(test_vertices)

    assert all(
        RotationFactory._make_correspondence_order(test_vertices) \
        == expected_order
    )
    assert all(
        RotationFactory._make_correspondence_order(test_vertices[:, ::-1]) \
        == expected_order[:, ::-1]
    )

def test__make_rotations():
    pass

def test__make_location_correspondences():
    pass

def test__calc_section():
    _calc_section = RotationFactory._calc_section
    expected_section = array([-1, -1, -1])
    assert all(_calc_section(array([-1.0,  -1.0,  -1.0 ])) == expected_section)
    assert all(_calc_section(array([-0.5,  -0.5,  -0.5 ])) == expected_section)
    assert any(_calc_section(array([-0.29, -0.5,  -0.5 ])) != expected_section)
    assert any(_calc_section(array([-0.5,  -0.29, -0.5 ])) != expected_section)
    assert any(_calc_section(array([-0.5,  -0.5,  -0.29])) != expected_section)

    expected_section = array([1, 1, 1])
    assert all(_calc_section(array([1.0,  1.0,  1.0 ])) == expected_section)
    assert all(_calc_section(array([0.5,  0.5,  0.5 ])) == expected_section)
    assert any(_calc_section(array([0.29, 0.5,  0.5 ])) != expected_section)
    assert any(_calc_section(array([0.5,  0.29, 0.5 ])) != expected_section)
    assert any(_calc_section(array([0.5,  0.5,  0.29])) != expected_section)
