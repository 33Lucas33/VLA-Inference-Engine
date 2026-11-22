const taskInfo = {
  libero_spatial: { title: "黑碗放盘", instruction: "把盘子与小碗之间的黑碗拿起，放到盘子上。" },
  libero_object: { title: "字母汤罐入篮", instruction: "拿起字母汤罐，放到篮子里。" },
  libero_goal: { title: "打开中间抽屉", instruction: "打开柜子中间的抽屉。" },
};
const horizons = [1, 5, 10, 20];
const state = { task: "libero_spatial", horizon: 1, rows: [], successRows: [] };
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

function renderSuccessStats() {
  const rows = state.successRows
    .filter((row) => row.suite === state.task)
    .sort((a, b) => a.k - b.k);
  $("#stats-task-title").textContent = `${taskInfo[state.task].title} · task 0`;
  const body = $("#success-table-body");
  body.replaceChildren(...rows.map((row) => {
    const tr = document.createElement("tr");
    tr.classList.toggle("is-current", row.k === state.horizon);
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = `K = ${row.k}`;
    tr.append(heading);

    const success = document.createElement("td");
    const rateCell = document.createElement("div");
    rateCell.className = "rate-cell";
    const rateValue = document.createElement("strong");
    rateValue.textContent = `${Math.round(row.success_rate * 100)}%`;
    const rateCount = document.createElement("span");
    rateCount.className = "rate-count";
    rateCount.textContent = `${row.successes} / ${row.episodes}`;
    const track = document.createElement("span");
    track.className = "rate-track";
    track.setAttribute("aria-label", `${row.successes} 次成功，共 ${row.episodes} 次`);
    const fill = document.createElement("i");
    fill.style.width = `${Math.round(row.success_rate * 100)}%`;
    track.append(fill);
    rateCell.append(rateValue, rateCount, track);
    success.append(rateCell);
    tr.append(success);

    const calls = document.createElement("td");
    calls.textContent = row.mean_model_calls.toFixed(2);
    tr.append(calls);
    const inference = document.createElement("td");
    inference.textContent = `${row.mean_inference_total_s.toFixed(2)} s`;
    tr.append(inference);
    const latency = document.createElement("td");
    latency.textContent = `${Math.round(row.call_latency_p50_s * 1000)} / ${Math.round(row.call_latency_p95_s * 1000)} ms`;
    tr.append(latency);
    const rollout = document.createElement("td");
    rollout.textContent = `${row.mean_rollout_wall_s.toFixed(2)} s`;
    tr.append(rollout);
    return tr;
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
  video.play().catch(() => {});
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
  document.querySelectorAll(".horizon-tab").forEach((button) => {
    const active = Number(button.dataset.horizon) === horizon;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  renderGrid();
  renderSuccessStats();
}

document.querySelectorAll(".task-tab").forEach((button) => {
  button.addEventListener("click", () => selectCase(button.dataset.task, state.horizon));
});
document.querySelectorAll(".horizon-tab").forEach((button) => {
  button.addEventListener("click", () => selectCase(state.task, Number(button.dataset.horizon)));
});
Promise.all([fetch("showcase/data.json?v=20261122-stats1"), fetch("showcase/success-rate.json?v=20261122-stats1")])
  .then(async (responses) => {
    for (const response of responses) {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }
    const [videos, stats] = await Promise.all(responses.map((response) => response.json()));
    state.rows = videos.rows;
    state.successRows = stats.results;
    selectCase(state.task, state.horizon);
    $("#video-loading").hidden = true;
  })
  .catch(() => {
    $("#video-loading").hidden = false;
    $("#video-loading").textContent = "实验数据加载失败，请刷新页面。";
  });
