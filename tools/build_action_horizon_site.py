"""Copy the selected experiment recordings into the static GitHub Pages site."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/experiments/action_horizon_20261120/evidence.json"
SITE = ROOT / "docs/showcase"


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    (SITE / "videos").mkdir(parents=True, exist_ok=True)
    (SITE / "posters").mkdir(parents=True, exist_ok=True)
    rows = []
    for row in evidence["rows"]:
        source = ROOT / row["video_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["video_sha256"]:
            raise ValueError(f"Video differs from recorded evidence: {source}")
        name = f"{row['suite'].removeprefix('libero_')}-k{row['n_action_steps']}"
        shutil.copyfile(source, SITE / "videos" / f"{name}.mp4")
        reader = imageio.get_reader(source)
        try:
            frame = reader.get_data(min(row["video_frames"] // 3, row["video_frames"] - 1))
        finally:
            reader.close()
        Image.fromarray(frame).save(SITE / "posters" / f"{name}.jpg", quality=85)
        rows.append(
            {
                "suite": row["suite"],
                "n_action_steps": row["n_action_steps"],
                "success": row["success"],
                "environment_steps": row["environment_steps"],
                "model_calls": row["model_calls"],
                "asset_name": name,
            }
        )
    (SITE / "data.json").write_text(
        json.dumps({"model_chunk_size": evidence["model_chunk_size"], "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} videos and posters in {SITE}")


if __name__ == "__main__":
    main()
