"""读取本目录的 STL 文件，计算包围盒、表面积、体积与网格封闭性。

用法:
    python stl_analysis.py

仅依赖 Python 标准库与 numpy。
"""

from __future__ import annotations

import collections
import struct
from pathlib import Path

import numpy as np


def load_stl(path: Path):
    """读取二进制 STL，返回三角面数组，形状为 (n, 3, 3)。"""
    data = path.read_bytes()
    face_count = struct.unpack_from("<I", data, 80)[0]
    records = np.frombuffer(data, dtype=np.uint8, count=face_count * 50, offset=84).reshape(
        face_count, 50
    )
    return np.frombuffer(
        records[:, 12:48].tobytes(), dtype="<f4"
    ).reshape(face_count, 3, 3).astype(float)


def analyze(path: Path) -> dict:
    tris = load_stl(path)
    n = len(tris)
    pts = tris.reshape(-1, 3)

    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area = float(np.linalg.norm(cross, axis=1).sum() / 2)
    signed_volume = float(
        np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    )

    # 封闭性：把每条边按端点坐标统计出现次数。
    # 完整封闭且方向一致的网格中每条边恰好出现 2 次。
    counts: collections.Counter = collections.Counter()
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for pa, pb in zip(tris[:, a], tris[:, b]):
            key = tuple(sorted((tuple(pa), tuple(pb))))
            counts[key] += 1
    edge_categories = collections.Counter(counts.values())
    watertight = set(edge_categories) == {2}

    return {
        "faces": n,
        "bbox_mm": (pts.max(axis=0) - pts.min(axis=0)) * 1000,
        "area_cm2": area * 1e4,
        "volume_cm3": abs(signed_volume) * 1e6,
        "open_edges": edge_categories.get(1, 0),
        "watertight": watertight,
    }


def main() -> None:
    for path in sorted(Path(__file__).parent.glob("*.stl")):
        r = analyze(path)
        bbox = ", ".join(f"{v:.1f}" for v in r["bbox_mm"])
        print(
            f"{path.name}: {r['faces']} faces | bbox {bbox} mm | "
            f"area {r['area_cm2']:.1f} cm^2 | volume {r['volume_cm3']:.1f} cm^3 | "
            f"open edges {r['open_edges']} | watertight {r['watertight']}"
        )


if __name__ == "__main__":
    main()
