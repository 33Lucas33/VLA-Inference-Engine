import subprocess
from pathlib import Path


def test_gpu_wait_propagates_gate_failure(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    root = Path(__file__).resolve().parents[2]
    (scripts / "wait_for_gpu.sh").write_text((root / "scripts/wait_for_gpu.sh").read_text())
    (scripts / "gpu_gate.sh").write_text("#!/bin/bash\nexit 2\n")
    result = subprocess.run(["bash", str(scripts / "wait_for_gpu.sh")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "next check" not in result.stderr
