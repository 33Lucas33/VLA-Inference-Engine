"""Shared, CPU-only contract for the current SmolVLA/LIBERO benchmark."""

SUITE_HORIZONS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
SUITE_ORDER = tuple(SUITE_HORIZONS)
TASK_IDS = tuple(range(10))
FORMAL_SEEDS = (1000, 2000, 3000)
EPISODES_PER_TASK = 10
INPUT_SHAPES = {
    "observation.images.image": (3, 256, 256),
    "observation.images.image2": (3, 256, 256),
    "observation.state": (8,),
}
ACTION_SHAPE = (7,)

# Shared by smoke and formal runs. Task selection, episode count and output
# location are supplied by the command builder, never by a second shell list.
INFERENCE_FLAGS = (
    "--policy.device=cuda",
    "--policy.num_steps=10",
    "--policy.n_action_steps=1",
    "--policy.compile_model=false",
    "--env.type=libero",
    "--env.control_mode=relative",
    "--env.observation_height=256",
    "--env.observation_width=256",
    "--env.init_states=true",
    "--env.hard_reset=true",
    "--env.max_parallel_tasks=1",
    "--eval.batch_size=1",
    "--eval.use_async_envs=false",
)
