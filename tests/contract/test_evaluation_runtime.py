"""Entry points must share protocol and execution environment."""

import json
import os
import subprocess
import sys

from vla_inference_engine.benchmark.libero import build_eval_command, build_formal_command
from vla_inference_engine.benchmark.runtime import evaluation_environment, project_root


def test_formal_and_smoke_share_runtime_options(tmp_path):
    args = dict(model_path=tmp_path / "model with spaces", seed=1000,
                output_dir=tmp_path / "seed_1000")
    smoke = build_eval_command(**args, lerobot_eval="lerobot-eval", suite="libero_spatial",
                               task_id=0, episodes=1)
    formal = build_formal_command(**args)
    differing = {token.split("=", 1)[0] for token in set(smoke) ^ set(formal)}
    assert differing == {"--env.task", "--env.task_ids", "--eval.n_episodes"}
    assert "--eval.n_episodes=10" in formal
    assert "--env.task=libero_spatial,libero_object,libero_goal,libero_10" in formal


def test_environment_and_shell_command_use_selected_python(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime with spaces"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin/python").symlink_to(sys.executable)
    monkeypatch.setenv("CONDA_ENV_PATH", str(runtime))
    root = project_root()
    env = evaluation_environment(root, model=tmp_path / "model", label="selected")
    assert env["PYTHON_BIN"] == str(runtime / "bin/python")
    assert env["PATH"].split(os.pathsep)[0] == str(runtime / "bin")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["EVAL_LABEL"] == "selected"
    # Exercise the real Bash bridge; special path characters remain single argv entries.
    model = tmp_path / "model space '$ ;\nnext"
    out = tmp_path / "run space/seed_1000"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; build_formal_eval_command "$2" 1000 "$3"; '
         'printf "%s\\0" "${EVAL_COMMAND[@]}"',
         "test", str(root / "scripts/common.sh"), str(model), str(out)],
        env=env, capture_output=True, check=True,
    )
    actual = [item.decode() for item in result.stdout.split(b"\0") if item]
    assert actual == build_formal_command(model_path=model, seed=1000, output_dir=out)


def test_cpu_imports_do_not_load_lerobot_or_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys,json; import vla_inference_engine.inference; "
         "import vla_inference_engine.cli; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True, text=True, check=True,
    )
    modules = json.loads(result.stdout)
    assert "torch" not in modules and "lerobot" not in modules
