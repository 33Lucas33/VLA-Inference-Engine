from pathlib import Path

from tools.validate_eval_command import EXPECTED_SUITE_HORIZONS
from tools.validate_pairing_protocol import PAIRING_FIX_COMMITS, validate_pairing_protocol


def test_fixed_formal_protocol_proves_outcome_independent_episode_pairing():
    root = Path(__file__).resolve().parents[1]
    result = validate_pairing_protocol(
        root, root / "outputs/preflight/formal_eval_smolvla_libero_paper_seed1000.json"
    )
    assert result["valid"] is True, result["errors"]
    assert result["pairing_fix_ancestry"] == {name: True for name in PAIRING_FIX_COMMITS}
    assert result["terminal_step_reset_calls"] == 0
    preflight_assertions = {row["name"]: row for row in result["assertions"]}
    assert preflight_assertions["formal_preflight.episode_length_override"]["observed"] is None
    assert (
        preflight_assertions["formal_preflight.resolved_suite_horizons"]["observed"]
        == EXPECTED_SUITE_HORIZONS
    )
    assert all(
        count == 50
        for counts in result["init_state_counts"].values()
        for count in counts
    )
    assert [
        trace["selected_init_state_ids"] for trace in result["formal_reset_traces"]
    ] == [list(range(10))] * 3
    assert result["seed_design"] == {
        "formal_process_seeds": [1000, 2000, 3000],
        "episode_seeds_per_task": [
            list(range(1000, 1010)),
            list(range(2000, 2010)),
            list(range(3000, 3010)),
        ],
        "fixed_init_state_ids_per_process": list(range(10)),
        "unique_fixed_init_state_ids_per_task_across_all_seeds": 10,
        "stochastic_replicates_per_task_init_state": 3,
        "interpretation": (
            "Three stochastic replications of fixed init-state IDs 0..9; "
            "not 30 distinct initial states per task."
        ),
    }
