# SPDX-FileCopyrightText: 2026 Jinhang Dong
# SPDX-License-Identifier: MIT
import struct

import numpy as np
import pytest

from kiss_icp.datasets.helipr import HeLiPRDataset

# Aeva added intensity after this recording timestamp.
AEVA_INTENSITY_TIMESTAMP = 1691936557946849179
LAYOUTS = [
    ("Avia", "fffBBBL", False),
    ("Aeva", "ffffflB", True),
    ("Aeva", "ffffflBf", False),
    ("Ouster", "ffffIHHH", False),
    ("Velodyne", "ffffHf", False),
]


def make_dataset(tmp_path, sequence, layout, legacy, count, trailing=b""):
    records = []
    for i in range(count):
        xyz = (10.0 + i, 20.0 + i, 30.0 + i)
        time = 100 * (i + 1)
        if sequence == "Avia":
            record = (*xyz, 10, 20, 3, time)
        elif sequence == "Aeva":
            record = (*xyz, 1.5, -0.25, time, 3)
            if not legacy:
                record += (2.5,)
        elif sequence == "Ouster":
            record = (*xyz, 1.5, time, 10, 3, 20)
        else:
            record = (*xyz, 1.5, 3, float(time))
        records.append(record)

    payload = b"".join(struct.pack("=" + layout, *record) for record in records) + trailing
    scan_dir = tmp_path / "LiDAR" / sequence
    scan_dir.mkdir(parents=True)
    first_time = AEVA_INTENSITY_TIMESTAMP - 1 if legacy else AEVA_INTENSITY_TIMESTAMP + 1
    frame_times = [first_time, first_time + 1]
    for time in frame_times:
        (scan_dir / f"{time}.bin").write_bytes(payload)
    gt_dir = tmp_path / "LiDAR_GT"
    gt_dir.mkdir()
    (gt_dir / f"global_{sequence}_gt.txt").write_text(
        "".join(f"{time} 0 0 0 0 0 0 1\n" for time in frame_times)
    )
    return HeLiPRDataset(tmp_path, sequence), np.array(records)


@pytest.mark.parametrize("sequence,layout,legacy", LAYOUTS)
@pytest.mark.parametrize("count", [1, 3])
def test_get_data_keeps_last_complete_record(tmp_path, sequence, layout, legacy, count):
    dataset, expected = make_dataset(tmp_path, sequence, layout, legacy, count)

    actual = dataset.get_data(0)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sequence,layout,legacy", LAYOUTS)
def test_get_data_ignores_incomplete_tail(tmp_path, sequence, layout, legacy):
    # Preserve the incomplete-record handling introduced in #338.
    dataset, expected = make_dataset(tmp_path, sequence, layout, legacy, 3, b"\x01\x02\x03")

    np.testing.assert_array_equal(dataset.get_data(0), expected)


@pytest.mark.parametrize("sequence,layout,legacy", LAYOUTS)
def test_getitem_preserves_points_and_timestamp_range(tmp_path, sequence, layout, legacy):
    dataset, expected = make_dataset(tmp_path, sequence, layout, legacy, 3)

    for idx in range(len(dataset)):
        points, timestamps = dataset[idx]
        np.testing.assert_array_equal(points, expected[:, :3])
        np.testing.assert_array_equal(timestamps, [0.0, 0.5, 1.0])
