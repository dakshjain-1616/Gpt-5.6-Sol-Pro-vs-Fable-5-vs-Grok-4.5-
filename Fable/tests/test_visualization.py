"""Tests: final-image dimensions, visual normalization/clipping bounds."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from visualize import clip01


class TestClipping:
    """Visual normalization must clip to fixed scales without altering raw data."""

    def test_congestion_scale_bounds(self):
        assert clip01(-5.0, 0.0, 100.0) == 0.0
        assert clip01(0.0, 0.0, 100.0) == 0.0
        assert clip01(50.0, 0.0, 100.0) == 0.5
        assert clip01(100.0, 0.0, 100.0) == 1.0
        assert clip01(250.0, 0.0, 100.0) == 1.0   # clipped, never >1

    def test_queue_scale_bounds(self):
        assert clip01(20.0, 0.0, 40.0) == 0.5
        assert clip01(400.0, 0.0, 40.0) == 1.0

    def test_wait_scale_bounds(self):
        assert clip01(90.0, 0.0, 180.0) == 0.5
        assert clip01(1e6, 0.0, 180.0) == 1.0
        assert clip01(-1.0, 0.0, 180.0) == 0.0

    def test_bad_scale_raises(self):
        with pytest.raises(ValueError):
            clip01(1.0, 10.0, 10.0)


@pytest.fixture(scope="module")
def rendered_png(tmp_path_factory):
    """Render a PNG from synthetic spatial/aggregate inputs via visualize.py."""
    tmp = tmp_path_factory.mktemp("viz")
    import sumolib

    net = sumolib.net.readNet(str(ROOT / "network" / "grid4x4.net.xml"))
    edges = [e.getID() for e in net.getEdges()]
    n = len(edges)
    spatial = {
        "edges": edges,
        "edge_avg_occupancy": [((i * 7) % 130) for i in range(n)],  # some >100 -> clipped
        "edge_avg_queue": [((i * 3) % 55) for i in range(n)],       # some >40 -> clipped
        "edge_avg_vehicles": [float(i % 9) for i in range(n)],
        "tls_avg_wait": {f"{c}{r}": float((ord(c) + r) * 17 % 220)
                         for c in "ABCD" for r in range(4)},
        "episodes_averaged": 1,
    }
    agg = {
        "algorithm": "Parameter-shared Double DQN",
        "overall": {
            "avg_waiting_time": {"mean": 123.4},
            "avg_queue_length": {"mean": 3.21},
            "throughput": {"mean": 987.0},
            "gridlock_duration": {"mean": 90.0},
        },
    }
    sp, ag = tmp / "spatial.json", tmp / "aggregate.json"
    sp.write_text(json.dumps(spatial))
    ag.write_text(json.dumps(agg))
    out = tmp / "out.png"
    r = subprocess.run(
        [sys.executable, str(ROOT / "visualize.py"), "--spatial", str(sp),
         "--aggregate", str(ag), "--output", str(out)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    return out


class TestFinalImage:
    def test_dimensions_exactly_1600x1600(self, rendered_png):
        from PIL import Image
        assert Image.open(rendered_png).size == (1600, 1600)

    def test_render_deterministic(self, rendered_png):
        """Re-render with same inputs -> byte-identical output."""
        import hashlib
        out2 = rendered_png.parent / "out2.png"
        r = subprocess.run(
            [sys.executable, str(ROOT / "visualize.py"),
             "--spatial", str(rendered_png.parent / "spatial.json"),
             "--aggregate", str(rendered_png.parent / "aggregate.json"),
             "--output", str(out2)],
            capture_output=True, text=True, cwd=str(ROOT))
        assert r.returncode == 0, r.stderr
        h1 = hashlib.sha256(rendered_png.read_bytes()).hexdigest()
        h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
        assert h1 == h2

    def test_out_of_range_inputs_do_not_crash(self, rendered_png):
        """Synthetic inputs deliberately exceed all scales; render must succeed."""
        assert rendered_png.exists() and rendered_png.stat().st_size > 10_000
