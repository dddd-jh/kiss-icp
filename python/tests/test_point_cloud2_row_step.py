# SPDX-FileCopyrightText: 2026 Jinhang Dong
# SPDX-License-Identifier: MIT
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from kiss_icp.tools.point_cloud2 import read_point_cloud, read_points


def make_cloud(height, width, padding, timestamp_field=None, bigendian=False):
    fields = [
        SimpleNamespace(name=name, offset=4 * idx, datatype=7, count=1, FLOAT32=7)
        for idx, name in enumerate(("x", "y", "z"))
    ]
    if timestamp_field is not None:
        fields.append(
            SimpleNamespace(name=timestamp_field, offset=16, datatype=8, count=1, FLOAT64=8)
        )
    # Include unused bytes inside each point as well as at the end of each row.
    point_step = 24
    row_step = width * point_step + padding
    data = bytearray(b"\xfe" * (height * row_step))
    points = np.arange(height * width * 3, dtype=np.float64).reshape(-1, 3) + 1.0
    timestamps = np.arange(height * width, dtype=np.float64) * 0.125
    byteorder = ">" if bigendian else "<"
    for idx, point in enumerate(points):
        row, column = divmod(idx, width)
        offset = row * row_step + column * point_step
        struct.pack_into(byteorder + "fff", data, offset, *point)
        if timestamp_field is not None:
            struct.pack_into(byteorder + "d", data, offset + 16, timestamps[idx])
    return (
        SimpleNamespace(
            fields=fields,
            height=height,
            width=width,
            is_bigendian=bigendian,
            point_step=point_step,
            row_step=row_step,
            data=data,
        ),
        points,
        timestamps,
    )


@pytest.mark.parametrize("height,width", [(1, 3), (2, 3), (3, 2)])
@pytest.mark.parametrize("padding", [0, 4, 24])
def test_read_points_respects_row_step(height, width, padding):
    cloud, expected, timestamps = make_cloud(height, width, padding, "time")

    actual = read_points(cloud)

    assert actual.shape == (height * width,)
    np.testing.assert_array_equal(actual["x"], expected[:, 0])
    np.testing.assert_array_equal(actual["y"], expected[:, 1])
    np.testing.assert_array_equal(actual["z"], expected[:, 2])
    np.testing.assert_array_equal(actual["time"], timestamps)
    if padding == 0:
        assert np.shares_memory(actual, np.frombuffer(cloud.data, dtype=np.uint8))


@pytest.mark.parametrize("timestamp_field", [None, "t", "time", "timestamp"])
def test_read_point_cloud_skips_row_padding(timestamp_field):
    cloud, expected, timestamps = make_cloud(3, 2, 4, timestamp_field)

    actual_points, actual_timestamps = read_point_cloud(cloud)

    np.testing.assert_array_equal(actual_points, expected)
    np.testing.assert_array_equal(actual_timestamps, [] if timestamp_field is None else timestamps)


def test_field_selection_and_flat_indices_skip_row_padding():
    cloud, expected, timestamps = make_cloud(2, 3, 4, "time")
    indices = [5, 0, 3, 3]

    actual = read_points(cloud, field_names=["z", "time"], uvs=iter(indices))

    assert actual.dtype.names == ("z", "time")
    np.testing.assert_array_equal(actual["z"], expected[indices, 2])
    np.testing.assert_array_equal(actual["time"], timestamps[indices])


def test_organized_output_keeps_point_order_without_padding():
    cloud, expected, _ = make_cloud(2, 3, 4)

    actual = read_points(cloud, reshape_organized_cloud=True)

    np.testing.assert_array_equal(actual["x"].ravel(), expected[:, 0])


def test_row_padding_with_non_native_byte_order():
    cloud, expected, timestamps = make_cloud(2, 3, 4, "time", sys.byteorder == "little")

    actual_points, actual_timestamps = read_point_cloud(cloud)

    np.testing.assert_array_equal(actual_points, expected)
    np.testing.assert_array_equal(actual_timestamps, timestamps)


@pytest.mark.parametrize("height,width", [(0, 0), (1, 0), (0, 3)])
def test_empty_cloud(height, width):
    cloud, _, _ = make_cloud(height, width, 0, "time")

    points, timestamps = read_point_cloud(cloud)

    assert points.shape == (0, 3)
    assert timestamps.shape == (0,)
