const $ = id => document.getElementById(id);
let catalog = null, session = null, socket = null, state = 'IDLE', generation = -1, loading = false;
let actionHistory = [], lastChartStep = -1;
const controls = ['start', 'pause', 'resume', 'step', 'reset', 'stop'];

function message(text) { $('message').textContent = text; }
function buttons() {
  $('load').disabled = loading;
  $('start').disabled = state !== 'READY';
  $('pause').disabled = state !== 'RUNNING';
  $('resume').disabled = state !== 'PAUSED';
  $('step').disabled = !['READY','PAUSED'].includes(state);
  $('reset').disabled = !session || state === 'LOADING';
  $('stop').disabled = !session || ['CLOSED','ERROR','FINISHED'].includes(state);
}
function taskList() {
  const suite = $('suite').value, tasks = catalog.tasks[suite] || [];
  $('task').replaceChildren(...tasks.map(t => new Option(`${t.id} · ${t.name}`, t.id)));
  taskLanguage();
}
function taskLanguage() {
  const task = catalog.tasks[$('suite').value]?.find(t => t.id === Number($('task').value));
  $('language').textContent = task?.language || '任务描述不可用';
}
function showAction(id, values) {
  $(id).textContent = Array.isArray(values) ? values.map((v,i) => `${['x','y','z','rx','ry','rz','grip'][i]}: ${Number(v).toFixed(4)}`).join('\n') : '—';
}
function drawChart() {
  const canvas = $('actionChart'), ctx = canvas.getContext('2d');
  const width = canvas.width, height = canvas.height, left = 38, right = width-12;
  ctx.clearRect(0, 0, width, height);
  const scale = Math.max(0.1, ...actionHistory.flatMap(row => row.values.map(Math.abs)));
  const y = value => height/2 - value*(height-26)/(2*scale);
  ctx.strokeStyle = '#d5dfda'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(left,height/2); ctx.lineTo(right,height/2); ctx.stroke();
  ctx.fillStyle = '#64736f'; ctx.font = '11px sans-serif';
  ctx.fillText(`+${scale.toFixed(2)}`, 4, 15);
  ctx.fillText('0', 21, height/2-4);
  ctx.fillText(`-${scale.toFixed(2)}`, 4, height-7);
  if (!actionHistory.length) return;
  const first = actionHistory[0].step, last = actionHistory.at(-1).step;
  for (const [axis,color] of ['#c65a50','#387dba','#328e67'].entries()) {
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
    actionHistory.forEach((row,index) => {
      const x=left+(row.step-first)* (right-left)/Math.max(1,last-first);
      if (index===0) ctx.moveTo(x,y(row.values[axis]));
      else ctx.lineTo(x,y(row.values[axis]));
    });
    ctx.stroke();
  }
}
function display(snapshot) {
  if (!snapshot || !session || snapshot.session_id !== session) return;
  if (snapshot.generation < generation) return;
  if (snapshot.generation > generation) {
    generation = snapshot.generation;
    $('mainCamera').removeAttribute('src'); $('wristCamera').removeAttribute('src');
    $('progress').textContent = '0 / —';
    $('outcome').textContent = '交互运行独立记录，不计入正式测评。';
    for (const id of ['normalized','processed','environment']) showAction(id);
    actionHistory=[]; lastChartStep=-1; drawChart();
  }
  state = snapshot.state;
  $('state').textContent = state;
  const f = snapshot.frame;
  if (f && f.generation === generation) {
    $('progress').textContent = `${f.step} / ${f.horizon}`;
    $('inference').textContent = f.inference_ms ? `${f.inference_ms.toFixed(1)} ms` : '—';
    $('elapsed').textContent = `${f.elapsed_s.toFixed(1)} s`;
    if (f.cameras?.image) $('mainCamera').src = `data:image/jpeg;base64,${f.cameras.image}`;
    if (f.cameras?.image2) $('wristCamera').src = `data:image/jpeg;base64,${f.cameras.image2}`;
    showAction('normalized', f.actions?.normalized);
    showAction('processed', f.actions?.processed);
    showAction('environment', f.actions?.environment);
    if (f.step > lastChartStep && Array.isArray(f.actions?.environment)) {
      actionHistory.push({step:f.step, values:f.actions.environment.slice(0,3)});
      lastChartStep=f.step; drawChart();
    }
  }
  const last = snapshot.latest;
  if (last?.type === 'task') $('language').textContent = last.language;
  if (last?.type === 'state' && last.state === 'FINISHED') $('outcome').textContent = last.success ? '任务成功' : `任务结束：${last.reason}`;
  if (last?.type === 'state' && last.state === 'ERROR') message(last.error || '工作进程出错，请检查 worker.log');
  buttons();
}
async function api(path, method='GET', body=null) {
  const response = await fetch(path, {method, headers: {'Content-Type':'application/json'}, body: body ? JSON.stringify(body) : null});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}
function connect() {
  socket?.close();
  if (!session) return;
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const current = new WebSocket(`${protocol}//${location.host}/api/session/${session}/stream`);
  socket = current;
  current.onopen = () => $('connection').textContent = '已连接';
  current.onmessage = event => display(JSON.parse(event.data));
  current.onclose = () => {
    if (socket === current && session) { $('connection').textContent = '连接中断'; setTimeout(connect, 1500); }
  };
}
async function load() {
  if (loading) return;
  const previousSession = session;
  try {
    loading=true; socket?.close(); socket=null; session=null; generation=-1;
    sessionStorage.removeItem('liveSession');
    $('mainCamera').removeAttribute('src'); $('wristCamera').removeAttribute('src');
    actionHistory=[]; lastChartStep=-1; drawChart();
    message('正在加载场景与模型…'); state='LOADING'; buttons();
    const result = await api('/api/session','POST',{suite:$('suite').value,task_id:Number($('task').value),model:$('model').value,seed:Number($('seed').value),init_state:Number($('initState').value)});
    session=result.session_id; generation=result.generation;
    sessionStorage.setItem('liveSession',session);
    $('outcome').textContent='交互运行独立记录，不计入正式测评。';
    display(result); connect();
    message('加载中；出现初始画面后可接管。');
  } catch(e) {
    message(e.message); state='IDLE';
    if (previousSession) {
      try {
        const previous = await api(`/api/session/${previousSession}`);
        session=previousSession; generation=previous.generation;
        sessionStorage.setItem('liveSession',session);
        display(previous); connect();
      } catch (_error) { sessionStorage.removeItem('liveSession'); }
    }
  }
  finally { loading=false; buttons(); }
}
async function command(name) {
  if (!session) return;
  try {
    const result=await api(`/api/session/${session}/command`,'POST',{command:name,command_id:crypto.randomUUID()});
    if (name==='reset') display(result);
    message(name==='pause' ? '正在暂停，等待当前步完成…' : `命令已发送：${name}`);
  } catch(e) { message(e.message); }
}
async function init() {
  try {
    catalog=await api('/api/catalog');
    $('suite').replaceChildren(...Object.keys(catalog.tasks).map(s => new Option(s,s)));
    $('model').replaceChildren(...catalog.models.map(m => new Option(`${m.label}${m.available?'':'（未安装）'}`,m.label)));
    taskList();
    $('connection').textContent='本地就绪';
    message('选择任务后加载。');
    const saved=sessionStorage.getItem('liveSession');
    if (saved) {
      try { const current=await api(`/api/session/${saved}`);session=saved;generation=current.generation;display(current);connect(); }
      catch (_error) { sessionStorage.removeItem('liveSession'); }
    }
  } catch(e) { message(e.message); }
  buttons();
}
$('suite').addEventListener('change',taskList);
$('task').addEventListener('change',taskLanguage);
$('load').addEventListener('click',load);
for (const name of controls) $(name).addEventListener('click',() => command(name));
init();
