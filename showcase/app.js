const taskInfo = {
  libero_spatial: { title: "黑碗放盘", instruction: "把盘子与小碗之间的黑碗拿起，放到盘子上。" },
  libero_object: { title: "字母汤罐入篮", instruction: "拿起字母汤罐，放到篮子里。" },
  libero_goal: { title: "打开中间抽屉", instruction: "打开柜子中间的抽屉。" },
};
const horizons = [1, 5, 10, 20];
const state = { task: "libero_spatial", horizon: 1, rows: [] };
const $ = (selector) => document.querySelector(selector);

function renderGrid() {
  const rows = horizons.map((k) => state.rows.find((row) => row.suite === state.task && row.n_action_steps === k));
  $("#case-grid").replaceChildren(...rows.map((row) => {
    const button = document.createElement("button");
    const selected = row.n_action_steps === state.horizon;
    button.type = "button";
    button.className = `case-card${selected ? " is-active" : ""}`;
    button.setAttribute("aria-pressed", String(selected));
    button.setAttribute("aria-label", `执行 ${row.n_action_steps} 步，${row.success ? "成功" : "失败"}，模型调用 ${row.model_calls} 次，环境步数 ${row.environment_steps}，播放视频`);
    button.innerHTML = `<img src="showcase/posters/${row.asset_name}.jpg" alt="" loading="lazy"><div class="card-body"><div class="card-top"><span class="card-k">K = ${row.n_action_steps}</span><span class="card-status${row.success ? "" : " fail"}">${row.success ? "成功" : "失败"}</span></div><div class="card-details">模型调用 <b>${row.model_calls}</b> 次 <span aria-hidden="true">·</span> ${row.environment_steps} 步</div><div class="card-meter" aria-hidden="true"><i style="width:${Math.max(2, 100 * row.model_calls / rows[0].model_calls)}%"></i></div></div>`;
    button.addEventListener("click", () => selectCase(state.task, row.n_action_steps));
    return button;
  }));
}

function selectCase(task, horizon) {
  state.task = task;
  state.horizon = horizon;
  const row = state.rows.find((item) => item.suite === task && item.n_action_steps === horizon);
  const video = $("#episode-video");
  video.pause();
  video.src = `showcase/videos/${row.asset_name}.mp4`;
  video.poster = `showcase/posters/${row.asset_name}.jpg`;
  video.load();
  $("#viewer-task-name").textContent = taskInfo[task].title;
  $("#viewer-case").textContent = `K = ${horizon}`;
  $("#task-instruction").textContent = taskInfo[task].instruction;
  $("#stat-result").textContent = row.success ? "成功" : "失败";
  $("#stat-result").classList.toggle("fail", !row.success);
  $("#stat-calls").textContent = row.model_calls;
  $("#stat-steps").textContent = row.environment_steps;
  $("#stat-horizon").textContent = `K = ${horizon}`;
  document.querySelectorAll(".task-tab").forEach((button) => {
    const active = button.dataset.task === task;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderGrid();
}

document.querySelectorAll(".task-tab").forEach((button) => {
  button.addEventListener("click", () => selectCase(button.dataset.task, state.horizon));
});
const grid = $(".action-grid");
for (let index = 0; index < 50; index += 1) {
  const cell = document.createElement("i");
  if (index < 10) cell.classList.add("active");
  grid.append(cell);
}
fetch("showcase/data.json")
  .then((response) => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then((data) => {
    state.rows = data.rows;
    selectCase(state.task, state.horizon);
    $("#video-loading").hidden = true;
  })
  .catch(() => { $("#video-loading").textContent = "实验数据加载失败，请刷新页面。"; });
