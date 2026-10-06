# SPDX-FileCopyrightText: 2026 Jinhang Dong
# SPDX-License-Identifier: MIT
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from kiss_icp.tools.point_cloud2 import read_point_cloud


@pytest.fixture(params=[("t", "I", 6), ("time", "f", 7), ("timestamp", "d", 8)])
def timestamp_field(request):
    return request.param


def make_cloud(points, timestamp_field=None):
    fields = [
        SimpleNamespace(name=name, offset=4 * idx, datatype=7, count=1, FLOAT32=7)
        for idx, name in enumerate(("x", "y", "z"))
    ]
    point_format = "<fff"
    if timestamp_field is not None:
        name, fmt, datatype = timestamp_field
        fields.append(
            SimpleNamespace(
                name=name,
                offset=12,
                datatype=datatype,
                count=1,
                UINT32=6,
                FLOAT32=7,
                FLOAT64=8,
            )
        )
        point_format += fmt
    data = b"".join(
        struct.pack(point_format, *point, 10 * (idx + 1))
        if timestamp_field is not None
        else struct.pack(point_format, *point)
        for idx, point in enumerate(points)
    )
    return SimpleNamespace(
        fields=fields,
        width=len(points),
        height=1,
        is_bigendian=False,
        point_step=struct.calcsize(point_format),
        row_step=len(points) * struct.calcsize(point_format),
        data=data,
    )


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
def test_nan_points_and_timestamps_are_filtered_together(timestamp_field, invalid_index):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    points[invalid_index, invalid_index] = np.nan
    cloud = make_cloud(points, timestamp_field)

    actual_points, actual_timestamps = read_point_cloud(cloud)

    np.testing.assert_array_equal(actual_points, np.delete(points, invalid_index, axis=0))
    np.testing.assert_array_equal(actual_timestamps, np.delete([10.0, 20.0, 30.0], invalid_index))
    assert actual_points.dtype == actual_timestamps.dtype == np.float64


def test_all_nan_points_return_empty_timestamps(timestamp_field):
    points, timestamps = read_point_cloud(make_cloud([[np.nan, 2.0, 3.0]], timestamp_field))
    assert points.shape == (0, 3)
    assert timestamps.shape == (0,)


def test_valid_points_keep_their_timestamps(timestamp_field):
    expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    points, timestamps = read_point_cloud(make_cloud(expected, timestamp_field))
    np.testing.assert_array_equal(points, expected)
    np.testing.assert_array_equal(timestamps, [10.0, 20.0])


def test_cloud_without_timestamp_field():
    points, timestamps = read_point_cloud(make_cloud([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]]))
    np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0]])
    assert timestamps.shape == (0,)


def test_empty_cloud(timestamp_field):
    points, timestamps = read_point_cloud(make_cloud([], timestamp_field))
    assert points.shape == (0, 3)
    assert timestamps.shape == (0,)
