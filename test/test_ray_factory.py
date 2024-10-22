import pytest
import ray_factory

from numpy import array, isclose
from math import sqrt

def test_setup():
    assert ray_factory.DefaultRays is not None
    assert ray_factory.DefaultPoints is not None
    assert len(ray_factory.DefaultRays) == 27
    assert len(ray_factory.DefaultPoints) == 27
    assert all(len(ray) == 3 for ray in ray_factory.DefaultRays)
    assert all(len(point) == 3 for point in ray_factory.DefaultPoints)

def test_symmetry():
    assert ray_factory.DefaultRays.sum() == pytest.approx(array([0, 0, 0]))
    assert ray_factory.DefaultPoints.sum() == pytest.approx(array([0, 0, 0]))
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 0], -0.33, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 1], -0.33, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 2], -0.33, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 0], 0.0, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 1], 0.0, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 2], 0.0, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 0], 0.33, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 1], 0.33, atol=0.004)]) == 9
    assert len(ray_factory.DefaultPoints[isclose(ray_factory.DefaultPoints[:, 2], 0.33, atol=0.004)]) == 9

    sqrt2_over_2 = sqrt(2.0) / 2.0
    sqrt3_over_3 = sqrt(3.0) / 3.0
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], -sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], -sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], -sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], sqrt3_over_3)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], -sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], -sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], -sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], sqrt2_over_2)]) == 4
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], -1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], -1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], -1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], 1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], 1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], 1.0)]) == 1
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 0], 0.0)]) == 9
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 1], 0.0)]) == 9
    assert len(ray_factory.DefaultRays[isclose(ray_factory.DefaultRays[:, 2], 0.0)]) == 9

def test_values():
    expected_rays = array([
        [-0.57735027, -0.57735027, -0.57735027 ],
        [-0.70710678, -0.70710678,  0.         ],
        [-0.57735027, -0.57735027,  0.57735027 ],
        [-0.70710678,  0.        , -0.70710678 ],
        [-1.        ,  0.        ,  0.         ],
        [-0.70710678,  0.        ,  0.70710678 ],
        [-0.57735027,  0.57735027, -0.57735027 ],
        [-0.70710678,  0.70710678,  0.         ],
        [-0.57735027,  0.57735027,  0.57735027 ],
        [ 0.        , -0.70710678, -0.70710678 ],
        [ 0.        , -1.        ,  0.         ],
        [ 0.        , -0.70710678,  0.70710678 ],
        [ 0.        ,  0.        , -1.         ],
        [ 0.        ,  0.        ,  0.         ],
        [ 0.        ,  0.        ,  1.         ],
        [ 0.        ,  0.70710678, -0.70710678 ],
        [ 0.        ,  1.        ,  0.         ],
        [ 0.        ,  0.70710678,  0.70710678 ],
        [ 0.57735027, -0.57735027, -0.57735027 ],
        [ 0.70710678, -0.70710678,  0.         ],
        [ 0.57735027, -0.57735027,  0.57735027 ],
        [ 0.70710678,  0.        , -0.70710678 ],
        [ 1.        ,  0.        ,  0.         ],
        [ 0.70710678,  0.        ,  0.70710678 ],
        [ 0.57735027,  0.57735027, -0.57735027 ],
        [ 0.70710678,  0.70710678,  0.         ],
        [ 0.57735027,  0.57735027,  0.57735027 ],
    ])
    assert ray_factory.DefaultRays == pytest.approx(expected_rays)

    expected_points = array([
        [-0.33333333, -0.33333333, -0.33333333 ],
        [-0.33333333, -0.33333333,  0.         ],
        [-0.33333333, -0.33333333,  0.33333333 ],
        [-0.33333333,  0.        , -0.33333333 ],
        [-0.33333333,  0.        ,  0.         ],
        [-0.33333333,  0.        ,  0.33333333 ],
        [-0.33333333,  0.33333333, -0.33333333 ],
        [-0.33333333,  0.33333333,  0.         ],
        [-0.33333333,  0.33333333,  0.33333333 ],
        [ 0.        , -0.33333333, -0.33333333 ],
        [ 0.        , -0.33333333,  0.         ],
        [ 0.        , -0.33333333,  0.33333333 ],
        [ 0.        ,  0.        , -0.33333333 ],
        [ 0.        ,  0.        ,  0.         ],
        [ 0.        ,  0.        ,  0.33333333 ],
        [ 0.        ,  0.33333333, -0.33333333 ],
        [ 0.        ,  0.33333333,  0.         ],
        [ 0.        ,  0.33333333,  0.33333333 ],
        [ 0.33333333, -0.33333333, -0.33333333 ],
        [ 0.33333333, -0.33333333,  0.         ],
        [ 0.33333333, -0.33333333,  0.33333333 ],
        [ 0.33333333,  0.        , -0.33333333 ],
        [ 0.33333333,  0.        ,  0.         ],
        [ 0.33333333,  0.        ,  0.33333333 ],
        [ 0.33333333,  0.33333333, -0.33333333 ],
        [ 0.33333333,  0.33333333,  0.         ],
        [ 0.33333333,  0.33333333,  0.33333333 ],
    ])
    assert ray_factory.DefaultPoints == pytest.approx(expected_points)
