from vla_inference_engine.benchmark.smoke import validate_smoke_result


def test_smoke_requires_requested_task_and_nonempty_video(tmp_path):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    payload = {
        "per_task": [{"task_group": "libero_spatial", "task_id": 0,
                      "metrics": {"successes": [True], "sum_rewards": [1.0],
                                  "max_rewards": [1.0], "video_paths": ["episode.mp4"]}}],
        "per_group": {"libero_spatial": {"n_episodes": 1}},
        "overall": {"n_episodes": 1, "video_paths": ["episode.mp4"]},
    }
    good = validate_smoke_result(payload, tmp_path, suite="libero_spatial",
                                 task_id=0, episodes=1)
    assert good["valid"] and good["formal_result"] is False
    video.unlink()
    bad = validate_smoke_result(payload, tmp_path, suite="libero_spatial",
                                task_id=0, episodes=1)
    assert not bad["valid"]
    assert any("video is missing" in error for error in bad["errors"])
