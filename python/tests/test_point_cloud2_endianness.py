# SPDX-FileCopyrightText: 2026 Jinhang Dong
# SPDX-License-Identifier: MIT
"""PointCloud2 decoding must preserve the serialized message buffer."""

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from kiss_icp.tools.point_cloud2 import read_point_cloud, read_points


def make_cloud(bigendian, buffer_type, empty=False):
    values = [(1.25, -2.5, 3.75, 0.0, 7), (4.5, 5.25, -6.75, 0.1, 256)]
    if empty:
        values = []
    order = ">" if bigendian else "<"
    raw = b"".join(struct.pack(order + "fff d H 2x", *point) for point in values)
    if buffer_type == "bytes":
        data = raw
    elif buffer_type == "bytearray":
        data = bytearray(raw)
    else:
        data = np.frombuffer(raw, dtype=np.uint8).copy()
    fields = [
        SimpleNamespace(
            name=name, offset=offset, datatype=code, count=1, FLOAT32=7, FLOAT64=8, UINT16=4
        )
        for name, offset, code in [
            ("x", 0, 7),
            ("y", 4, 7),
            ("z", 8, 7),
            ("time", 12, 8),
            ("ring", 20, 4),
        ]
    ]
    cloud = SimpleNamespace(
        width=len(values),
        height=1,
        point_step=24,
        row_step=24 * len(values),
        fields=fields,
        data=data,
        is_bigendian=bigendian,
    )
    return cloud, values


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("buffer_type", ["bytes", "bytearray", "numpy"])
@pytest.mark.parametrize("field_names", [None, ["x", "z"], ["time", "ring"]])
def test_read_points_preserves_buffer(bigendian, buffer_type, field_names):
    cloud, values = make_cloud(bigendian, buffer_type)
    original = bytes(cloud.data)
    names = field_names or ["x", "y", "z", "time", "ring"]
    for _ in range(2):
        points = read_points(cloud, field_names=field_names)
        for name in names:
            index = ["x", "y", "z", "time", "ring"].index(name)
            np.testing.assert_array_equal(points[name], [point[index] for point in values])
        assert bytes(cloud.data) == original
    if bigendian == (not np.little_endian):
        assert np.shares_memory(points, np.frombuffer(cloud.data, dtype=np.uint8))


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("buffer_type", ["bytes", "bytearray", "numpy"])
def test_read_point_cloud_preserves_buffer(bigendian, buffer_type):
    cloud, values = make_cloud(bigendian, buffer_type)
    original = bytes(cloud.data)
    for _ in range(2):
        points, timestamps = read_point_cloud(cloud)
        np.testing.assert_array_equal(points, [point[:3] for point in values])
        np.testing.assert_array_equal(timestamps, [point[3] for point in values])
        assert bytes(cloud.data) == original


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("buffer_type", ["bytes", "bytearray", "numpy"])
def test_read_points_uvs_preserves_buffer(bigendian, buffer_type):
    cloud, values = make_cloud(bigendian, buffer_type)
    original = bytes(cloud.data)
    points = read_points(cloud, field_names=["z", "ring"], uvs=[1, 0])
    np.testing.assert_array_equal(points["z"], [point[2] for point in values[::-1]])
    np.testing.assert_array_equal(points["ring"], [point[4] for point in values[::-1]])
    assert bytes(cloud.data) == original


@pytest.mark.parametrize("bigendian", [False, True])
@pytest.mark.parametrize("buffer_type", ["bytes", "bytearray", "numpy"])
def test_read_points_empty_buffer(bigendian, buffer_type):
    cloud, _ = make_cloud(bigendian, buffer_type, empty=True)
    result = read_points(cloud)
    assert result.shape == (0,)
    assert bytes(cloud.data) == b""
