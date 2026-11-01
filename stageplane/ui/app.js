const $ = id => document.getElementById(id);
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[character]));
const number = (value, fallback = 0) => Number.isFinite(Number(value)) ? Number(value) : fallback;
const clamp = value => Math.max(0, Math.min(100, number(value)));
const list = value => Array.isArray(value) ? value : [];
const activeRequest = request => ['running', 'recovering'].includes(request.status);
const formatCount = value => number(value).toLocaleString();
const formatBytes = value => {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  const bytes = Math.max(0, Number(value));
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  const exponent = Math.max(0, Math.min(units.length - 1, Math.floor(Math.log2(bytes || 1) / 10)));
  return `${(bytes / 1024 ** exponent).toLocaleString(undefined, { maximumFractionDigits: exponent > 2 ? 2 : 1 })} ${units[Math.max(0, exponent)]}`;
};
const duration = ms => ms == null ? '—' : number(ms) < 10 ? `${number(ms).toFixed(2)} ms` : number(ms) < 1000 ? `${Math.round(number(ms))} ms` : number(ms) < 60000 ? `${(number(ms) / 1000).toFixed(1)} s` : `${Math.floor(number(ms) / 60000)}m ${Math.floor(number(ms) / 1000) % 60}s`;
const timeLabel = timestamp => timestamp ? new Date(number(timestamp) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
const modelLabel = model => String(model?.source || model?.key || '').split('/').filter(Boolean).pop() || 'Unknown model';
const statusLabels = { ready: 'Ready', loading: 'Loading', idle: 'Idle', draining: 'Draining', error: 'Load error', offline: 'Offline', running: 'Running', recovering: 'Recovering', completed: 'Completed', failed: 'Failed', cancelled: 'Cancelled' };
const statusChip = status => `<span class="status-chip ${Object.hasOwn(statusLabels, status) ? status : 'offline'}"><i></i>${escapeHTML(statusLabels[status] || status)}</span>`;
const empty = (title, message, columns) => columns ? `<tr><td colspan="${columns}" class="empty-cell"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(message)}</span></td></tr>` : `<div class="empty-cell"><strong>${escapeHTML(title)}</strong><span>${escapeHTML(message)}</span></div>`;

let state = { nodes: [], models: [], deployments: {}, events: [], requests: [], jobs: [], summary: {} };
let token = '';
try { token = sessionStorage.getItem('stageplane-token') || ''; } catch { /* Storage can be disabled by the browser. */ }
let socket, reconnectTimer, pollTimer, renderTimer, refreshPromise;
let reconnectAttempt = 0, liveEvents = false, authBlocked = false, disposed = false;
let credentialsVersion = 0;
let lastSnapshotAt = 0, selectedModel = '', modelOptionsSignature = '', noticeKind = '';
let deploymentFormModel = '', deploymentFormDirty = false;
let running = false, runController, importPending = false, detailSelection = null;
let pages = { nodes: 0, requests: 0, stages: 0 };
const PAGE_SIZE = { nodes: 25, requests: 10, stages: 30 };
let lastTrace = [], traceTime = 0;
let graph, graphDirty = true, graphSignature = '', graphUnavailable = false;
$('executor-host').textContent = location.host;

function showNotice(message, kind = 'action', error = false) {
  noticeKind = kind;
  $('notice').textContent = message;
  $('notice').className = `notice${error ? ' error' : ''}`;
  $('notice').hidden = !message;
}
function openDialog(id) { if (!$(id).open) $(id).showModal(); }
function showAccess() { $('token').value = token; openDialog('auth-dialog'); }
function setToken(value) {
  token = value.trim(); credentialsVersion++; authBlocked = false; reconnectAttempt = 0;
  state = { nodes: [], models: [], deployments: {}, events: [], requests: [], jobs: [], summary: {} };
  lastSnapshotAt = 0; selectedModel = ''; modelOptionsSignature = ''; runController?.abort(); scheduleRender();
  try { token ? sessionStorage.setItem('stageplane-token', token) : sessionStorage.removeItem('stageplane-token'); } catch { /* Session-only in memory fallback. */ }
  stopSocket(); showNotice(''); refresh(); connect();
}
function authFailure(status) {
  authBlocked = true; stopSocket();
  state = { nodes: [], models: [], deployments: {}, events: [], requests: [], jobs: [], summary: {} };
  lastSnapshotAt = 0; selectedModel = ''; modelOptionsSignature = ''; scheduleRender();
  showNotice(status === 403 ? 'This key cannot access cluster operations. Use an operations or administrator key in Access settings.' : 'Executor authentication is required. Enter a valid access token to connect.', 'connection', true);
  if (status === 401) showAccess();
  renderConnection();
}
async function responseError(response) {
  let message = `Request failed (HTTP ${response.status})`;
  try {
    const body = await response.json();
    const detail = body.detail || body.error?.message || body.message;
    if (typeof detail === 'string') message = detail;
    else if (detail) message = JSON.stringify(detail);
  } catch { /* An upstream HTML error still has an HTTP status. */ }
  return new Error(message.slice(0, 600));
}
async function api(path, method = 'GET', body) {
  const version = credentialsVersion;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(path, { method, cache: 'no-store', signal: controller.signal,
      headers: { ...(body === undefined ? {} : { 'Content-Type': 'application/json' }), ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    if (version === credentialsVersion && (response.status === 401 || (response.status === 403 && path === '/api/state'))) authFailure(response.status);
    if (!response.ok) throw await responseError(response);
    return await response.json();
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('The executor did not respond within 15 seconds. Refresh to check the current state.');
    throw error;
  } finally { clearTimeout(timeout); }
}
function applySnapshot(snapshot) {
  if (!snapshot || !Array.isArray(snapshot.nodes) || !Array.isArray(snapshot.models)) return;
  if (snapshot.time && state.time && snapshot.time < state.time) return;
  const newerEvents = state.events.filter(event => number(event.time) > number(snapshot.time));
  const events = new Map([...list(snapshot.events), ...newerEvents].map(event => [event.id || `${event.kind}:${event.time}`, event]));
  state = { ...snapshot, nodes: snapshot.nodes, models: snapshot.models, deployments: snapshot.deployments || {},
    events: [...events.values()].slice(-400), requests: list(snapshot.requests), jobs: list(snapshot.jobs), summary: snapshot.summary || {} };
  lastSnapshotAt = Date.now(); authBlocked = false;
  if (noticeKind === 'connection') showNotice('');
  scheduleRender();
}
function scheduleRender() {
  if (!renderTimer) renderTimer = setTimeout(() => { renderTimer = null; render(); }, 60);
}
async function refresh() {
  if (refreshPromise) return refreshPromise;
  if (disposed || authBlocked) return;
  const version = credentialsVersion;
  $('refresh-button').disabled = true;
  refreshPromise = api('/api/state').then(snapshot => { if (version === credentialsVersion) applySnapshot(snapshot); }).catch(error => {
    if (version === credentialsVersion && !authBlocked) showNotice(`Connection interrupted. Displaying the last snapshot. ${error.message}`, 'connection', true);
  }).finally(() => { refreshPromise = null; $('refresh-button').disabled = false; renderConnection(); });
  return refreshPromise;
}
async function poll() {
  await refresh();
  if (!disposed) pollTimer = setTimeout(poll, document.hidden ? 10000 : 2000);
}
function renderConnection() {
  const age = lastSnapshotAt ? Date.now() - lastSnapshotAt : Infinity;
  const healthy = age < 8000 && !authBlocked;
  const color = healthy ? 'good' : lastSnapshotAt && !authBlocked ? 'warning' : 'bad';
  const label = authBlocked ? 'Access required' : healthy ? liveEvents ? 'Live connection' : 'Polling executor' : lastSnapshotAt ? 'Snapshot is stale' : 'Connecting';
  const detail = lastSnapshotAt ? `Updated ${Math.floor(age / 1000)}s ago` : 'Waiting for executor';
  $('connection').innerHTML = `${escapeHTML(label)}<small>${escapeHTML(detail)}</small>`;
  $('connection-dot').className = `status-dot ${color}`;
  $('sync-status').innerHTML = `<i class="status-dot ${color}"></i>${escapeHTML(healthy ? liveEvents ? 'Live' : 'Polling' : authBlocked ? 'Access required' : 'Reconnecting')}`;
  $('snapshot-time').textContent = state.time ? `Snapshot ${timeLabel(state.time)}` : 'Waiting for a snapshot';
}
function stopSocket() {
  clearTimeout(reconnectTimer); reconnectTimer = null; liveEvents = false;
  if (socket) { socket.onopen = socket.onmessage = socket.onclose = socket.onerror = null; socket.close(); socket = null; }
}
function connect() {
  if (disposed || authBlocked) return;
  stopSocket();
  const current = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/events`);
  socket = current;
  current.onopen = () => current.send(JSON.stringify({ token }));
  current.onmessage = message => {
    if (socket !== current) return;
    let event;
    try { event = JSON.parse(message.data); } catch { return; }
    if (event.kind === 'snapshot') {
      liveEvents = true; reconnectAttempt = 0; applySnapshot(event.state); renderConnection(); return;
    }
    if (!event.kind || !liveEvents) return;
    if (!state.events.some(item => item.id && item.id === event.id)) state.events.push(event);
    state.events = state.events.slice(-400);
    if (event.kind === 'token_committed') {
      const record = state.requests.find(request => request.id === event.request);
      if (record) { record.output_tokens = number(event.offset) + 1; record.last_trace = list(event.trace); record.updated = event.time; }
      if (!record || record.model === selectedModel) {
        lastTrace = list(event.trace); traceTime = performance.now(); graphDirty = true;
        $('graph-caption').textContent = `Token ${number(event.offset) + 1} · ${duration(event.ms)}`;
      }
    }
    if (['request_started', 'request_completed', 'request_failed', 'request_cancelled', 'placement', 'node_joined', 'node_lost', 'model_imported'].includes(event.kind)) refresh();
    scheduleRender();
  };
  current.onclose = event => {
    if (socket !== current) return;
    socket = null; liveEvents = false; renderConnection();
    if (event.code === 1008) { authFailure(401); return; }
    if (!disposed && !authBlocked) {
      const delay = Math.min(30000, 1000 * 2 ** Math.min(reconnectAttempt++, 5)) + Math.random() * 300;
      reconnectTimer = setTimeout(connect, delay);
    }
  };
  current.onerror = () => { /* onclose controls the single reconnect timer. */ };
}
function currentModel() { return state.models.find(model => model.key === selectedModel); }
function workerStatus(node) {
  if (!node.active) return 'offline';
  if (node.draining) return 'draining';
  if (Object.keys(node.metrics?.errors || {}).length) return 'error';
  if (node.metrics?.loading) return 'loading';
  return list(node.ready).length ? 'ready' : 'idle';
}
function elapsed(request) {
  if (!activeRequest(request)) return request.elapsed_ms ?? (request.finished && request.started ? (request.finished - request.started) * 1000 : null);
  if (request.elapsed_ms != null) return number(request.elapsed_ms) + Math.max(0, Date.now() - lastSnapshotAt);
  return Math.max(0, ((state.time || Date.now() / 1000) - number(request.started)) * 1000 + Math.max(0, Date.now() - lastSnapshotAt));
}
function page(items, key) {
  const size = PAGE_SIZE[key]; pages[key] = Math.min(pages[key], Math.max(0, Math.ceil(items.length / size) - 1));
  const start = pages[key] * size;
  $(`${key === 'nodes' ? 'node' : key === 'stages' ? 'stage' : 'request'}-prev`).disabled = start === 0;
  $(`${key === 'nodes' ? 'node' : key === 'stages' ? 'stage' : 'request'}-next`).disabled = start + size >= items.length;
  return { items: items.slice(start, start + size), start, end: Math.min(start + size, items.length), total: items.length };
}
function rangeText(result, noun) { return result.total ? `${result.start + 1}–${result.end} of ${formatCount(result.total)} ${noun}` : `No ${noun} to display`; }
function routeReady(model) {
  if (!model || !list(model.stages).length) return false;
  const deployment = state.deployments[model.key];
  if (!deployment) return false;
  if (typeof deployment.capacity?.serving === 'boolean') return deployment.capacity.serving;
  if (deployment.capacity?.status === 'insufficient_capacity') return false;
  const nodes = new Map(state.nodes.map(node => [node.id, node]));
  return model.stages.every(stage => list(deployment.placements?.[stage.id]).some(id => {
    const node = nodes.get(id);
    return node?.active && !node.draining && list(node.ready).includes(`${model.key}:${stage.id}`);
  }));
}
function render() {
  const signature = JSON.stringify(state.models.map(model => [model.key, model.source, model.revision]));
  if (signature !== modelOptionsSignature) {
    modelOptionsSignature = signature;
    if (!state.models.some(model => model.key === selectedModel)) selectedModel = state.models[0]?.key || '';
    $('model-select').replaceChildren(...(state.models.length ? state.models.map(model => new Option(`${modelLabel(model)} · ${String(model.revision || '').slice(0, 8)}`, model.key)) : [new Option('No models available', '')]));
    $('model-select').value = selectedModel;
  }
  const model = currentModel();
  const active = state.nodes.filter(node => node.active);
  const serving = active.filter(node => !node.draining && list(node.ready).length);
  const requests = state.requests.filter(activeRequest);
  const completed = state.requests.filter(request => request.status === 'completed');
  const failed = state.requests.filter(request => request.status === 'failed');
  const latest = [...state.requests].reverse().find(request => request.ttft_ms != null);
  const latestCompleted = completed.at(-1);
  $('active-count').innerHTML = lastSnapshotAt ? `${serving.length}<span> / ${formatCount(state.nodes.length)}</span>` : '—';
  $('workers-detail').textContent = `${active.length} online · ${active.filter(node => node.metrics?.loading).length} loading · ${active.filter(node => node.draining).length} draining`;
  $('worker-bar').style.width = `${clamp(serving.length / (state.nodes.length || 1) * 100)}%`;
  $('nav-count').textContent = formatCount(state.nodes.length);
  $('node-badge').textContent = formatCount(state.nodes.length);
  $('nav-requests').textContent = requests.length;
  $('request-count').textContent = lastSnapshotAt ? requests.length : '—';
  $('requests-detail').textContent = `${requests.filter(request => request.status === 'recovering').length} recovering · ${state.requests.length} recent requests`;
  $('request-outcomes').textContent = `${completed.length} completed · ${failed.length} failed in retained history`;
  $('ttft').textContent = latest ? duration(latest.ttft_ms) : '—';
  $('ttft-detail').textContent = latest ? `Request ${String(latest.id).slice(-8)} · ${timeLabel(latest.started)}` : 'Most recent measured request';
  $('throughput').textContent = latestCompleted?.tokens_per_second != null ? `${number(latestCompleted.tokens_per_second).toFixed(2)} output tokens/s · last completion` : 'Output rate unavailable';
  const rss = active.reduce((sum, node) => sum + number(node.metrics?.rss_bytes), 0);
  const memoryBudget = active.reduce((sum, node) => sum + number(node.metrics?.memory_limit_bytes ?? node.memory_bytes), 0);
  $('memory').textContent = active.length ? formatBytes(rss) : '—';
  $('memory-detail').textContent = active.length ? `${formatBytes(memoryBudget)} published serving budget` : 'Aggregate resident process memory';
  $('memory-bar').style.width = `${clamp(rss / (memoryBudget || 1) * 100)}%`;
  const ready = routeReady(model), deployment = model && state.deployments[model.key];
  const capacity = deployment?.capacity;
  if (deploymentFormModel !== selectedModel) { deploymentFormModel = selectedModel; deploymentFormDirty = false; }
  if (!deploymentFormDirty && ![$('replicas'), $('groups')].includes(document.activeElement)) {
    $('replicas').value = deployment?.replicas || 1;
    $('groups').value = deployment?.groups || 1;
  }
  $('run').disabled = running || !ready || authBlocked || Date.now() - lastSnapshotAt > 8000;
  $('deploy').disabled = !model || authBlocked;
  const readinessLabels = { rebalancing: 'Rebalancing', loading: 'Loading stages', degraded: 'Reduced replicas', blocked: 'Capacity blocked' };
  const warning = readinessLabels[capacity?.readiness];
  $('route-status').textContent = warning || (ready ? 'Route ready' : model ? deployment ? 'Not ready' : 'Undeployed' : 'No model');
  $('route-status').className = `badge${warning ? ' warning' : ready ? ' good' : model ? ' warning' : ''}`;
  $('model-summary').textContent = model ? `${model.architecture} · ${formatCount(list(model.stages).length)} stages · ${formatBytes(model.artifact_bytes)} artifacts` : 'Import a model to get started.';
  let capacityMessage = capacity?.status === 'insufficient_capacity' ? `${capacity.message || 'Desired placement exceeds available capacity.'}${ready ? ' Existing ready replicas continue serving.' : ''}` : '';
  if (!capacityMessage && capacity?.readiness === 'rebalancing') capacityMessage = `Rebalancing placement. ${formatCount(capacity.target_ready_stages)} / ${formatCount(capacity.total_stages)} target stages ready. Existing ready replicas continue serving while replacements load.`;
  if (!capacityMessage && capacity?.readiness === 'loading') capacityMessage = `Loading model stages. ${formatCount(capacity.ready_stages)} / ${formatCount(capacity.total_stages)} stages ready. Inference becomes available when a complete route is ready.`;
  if (!capacityMessage && capacity?.readiness === 'degraded') capacityMessage = `${formatCount(list(capacity.under_replicated_stages).length)} stages have fewer than ${formatCount(capacity.replica_target)} target replicas. ${ready ? 'Inference is available on the remaining ready route.' : 'Waiting for a complete ready route.'}`;
  $('capacity-banner').hidden = !capacityMessage;
  $('capacity-banner').textContent = capacityMessage;
  renderRequests(); renderNodes(); renderStages(model); renderEvents(); renderImports(); renderConnection();
  if ($('detail-dialog').open && detailSelection) renderDetail();
  buildGraph();
}
function renderRequests() {
  let requests = [...state.requests].sort((a, b) => Number(activeRequest(b)) - Number(activeRequest(a)) || number(b.started) - number(a.started));
  const filter = $('request-filter').value;
  if (filter !== 'all') requests = requests.filter(request => filter === 'active' ? activeRequest(request) : filter === 'failed' ? ['failed', 'cancelled'].includes(request.status) : request.status === filter);
  const result = page(requests, 'requests');
  $('request-badge').textContent = state.requests.length;
  $('request-table').innerHTML = result.items.map(request => {
    const model = state.models.find(model => model.key === request.model);
    const elapsedMs = elapsed(request);
    const rate = number(request.output_tokens) && elapsedMs ? `${(number(request.output_tokens) / (elapsedMs / 1000)).toFixed(2)} tok/s` : 'Awaiting first token';
    return `<tr><td><strong class="truncate" title="${escapeHTML(request.id)}">${escapeHTML(String(request.id).slice(0, 9))}…${escapeHTML(String(request.id).slice(-8))}</strong><small class="truncate" title="${escapeHTML(request.model)}">${escapeHTML(modelLabel(model || { key: request.model }))}</small></td><td>${statusChip(request.status)}</td><td><strong>${formatCount(request.output_tokens)}</strong><small>${escapeHTML(number(request.output_tokens) ? rate : `${formatCount(request.prompt_tokens)} prompt tokens`)}</small></td><td>${duration(elapsedMs)}<small>Started ${timeLabel(request.started)}</small></td><td>${duration(request.ttft_ms)}</td><td>${number(request.recoveries) ? `${formatCount(request.recoveries)} replans` : '—'}</td><td><button class="row-button" data-request="${escapeHTML(request.id)}" aria-label="Inspect request ${escapeHTML(request.id)}">Inspect ↗</button></td></tr>`;
  }).join('') || empty('No requests in this view', filter === 'all' ? 'Send a request through the API or use the inference console below.' : 'Try another filter to view recent requests.', 7);
  $('request-page-info').textContent = `${rangeText(result, 'requests')} · prompts are not displayed`;
}
function resourceBar(percent) { return `<div class="resource-bar"><i class="${percent > 90 ? 'high' : ''}" style="width:${clamp(percent)}%"></i></div>`; }
function workerAction(node) { return `<button class="row-button" data-${node.draining ? 'resume' : 'drain'}="${escapeHTML(node.id)}" ${!node.active ? 'disabled' : ''}>${node.draining ? 'Resume' : 'Drain'}</button>`; }
function renderNodes() {
  const search = $('node-search').value.trim().toLowerCase();
  const filter = $('node-filter').value;
  const nodes = state.nodes.filter(node => (!search || `${node.id} ${node.url} ${node.group}`.toLowerCase().includes(search)) && (filter === 'all' || workerStatus(node) === filter));
  const result = page(nodes, 'nodes');
  $('node-table').innerHTML = result.items.map(node => {
    const metrics = node.metrics || {}, status = workerStatus(node);
    const cores = number(node.cpu_cores ?? metrics.cpu_limit_cores);
    const cpuUsed = number(metrics.cpu_usage_percent) / 100;
    const cpuPercent = cores && metrics.cpu_usage_percent != null ? cpuUsed / cores * 100 : null;
    const memoryBudget = number(metrics.memory_limit_bytes ?? node.memory_bytes);
    const memoryPercent = memoryBudget ? number(metrics.rss_bytes) / memoryBudget * 100 : 0;
    return `<tr><td><strong class="truncate" title="${escapeHTML(node.id)}">${escapeHTML(node.id)}</strong><small class="truncate" title="${escapeHTML(node.url)}">${escapeHTML(node.url)}</small></td><td>${statusChip(status)}</td><td class="resource-cell"><div><span>${cpuPercent == null ? '—' : `${cpuPercent.toFixed(1)}%`}</span><b>${cores ? `${cores.toFixed(2)} cores` : 'Unreported'}</b></div>${resourceBar(cpuPercent)}<small>${metrics.cpu_usage_percent == null ? 'Usage unreported' : `${cpuUsed.toFixed(2)} cores consumed`}</small></td><td class="resource-cell"><div><span>${formatBytes(metrics.rss_bytes)}</span></div>${resourceBar(memoryPercent)}<small>${formatBytes(memoryBudget)} budget</small></td><td><strong>${list(node.ready).length} stages</strong><small>${formatCount(metrics.sessions)} sessions · ${escapeHTML(metrics.dtype || 'dtype unreported')}</small></td><td>${escapeHTML(String(node.transport || '').toUpperCase())}<small>Group ${escapeHTML(node.group)} · ${escapeHTML(metrics.device || 'device unreported')}</small></td><td><div class="row-actions"><button class="row-button" data-worker="${escapeHTML(node.id)}" aria-label="Inspect worker ${escapeHTML(node.id)}">Inspect ↗</button>${workerAction(node)}</div></td></tr>`;
  }).join('') || empty(search || filter !== 'all' ? 'No matching workers' : 'No workers registered', search || filter !== 'all' ? 'Change the search or status filter.' : 'Start a node and connect it to this executor.', 7);
  $('node-page-info').textContent = `${rangeText(result, 'workers')} · CPU is normalized to published cores`;
}
function renderStages(model) {
  const stages = list(model?.stages), deployment = model ? state.deployments[model.key] : null;
  const nodes = new Map(state.nodes.map(node => [node.id, node]));
  const ready = (id, stage) => { const node = nodes.get(id); return node?.active && !node.draining && list(node.ready).includes(`${model.key}:${stage.id}`); };
  const result = page(stages, 'stages');
  const readyStages = stages.filter(stage => list(deployment?.placements?.[stage.id]).some(id => ready(id, stage))).length;
  $('stage-count').textContent = formatCount(stages.length);
  $('stage-coverage').textContent = model ? `${readyStages} / ${stages.length} target stages ready` : 'No deployment';
  $('stage-list').innerHTML = result.items.map(stage => {
    const owners = list(deployment?.placements?.[stage.id]);
    const retiring = list(deployment?.retiring_placements?.[stage.id]);
    const layerLabel = stage.kind === 'blocks' ? `Layers ${number(stage.start)}–${number(stage.end) - 1}` : stage.kind;
    return `<div class="stage-row"><div><strong>${escapeHTML(stage.id)}</strong><small>${escapeHTML(layerLabel)} · ${formatBytes(stage.weight_bytes)}</small></div><div>${owners.slice(0, 4).map(id => `<span class="stage-chip ${ready(id, stage) ? 'ready' : ''}" title="${escapeHTML(id)}">${escapeHTML(id)}</span>`).join('')}${owners.length > 4 ? `<span class="stage-chip" title="${escapeHTML(owners.slice(4).join(', '))}">+${owners.length - 4}</span>` : ''}${owners.length ? '' : '<span class="muted">Unassigned</span>'}${retiring.length ? `<small title="${escapeHTML(retiring.join(', '))}">${retiring.length} previous replicas retained</small>` : ''}</div><span class="muted">${owners.filter(id => ready(id, stage)).length}/${owners.length} target ready</span></div>`;
  }).join('') || empty('No model stages yet', 'Select a model to inspect chunk placement and replica readiness.');
  $('stage-page-info').textContent = `${rangeText(result, 'stages')} · ${deployment ? `${number(deployment.replicas)} target replicas · ${number(deployment.groups)} groups` : 'Not deployed'}`;
}
function renderEvents() {
  const operationsOnly = $('activity-filter').value === 'operations';
  const events = [...state.events].reverse().filter(event => !operationsOnly || event.kind !== 'token_committed').slice(0, 45);
  const titles = { node_joined: 'Worker joined', node_lost: 'Worker disconnected', node_draining: 'Worker draining', placement: 'Placement updated', request_started: 'Inference admitted', recovery: 'Restoring committed prefix', chunk_retry: 'Replacement route selected', insufficient_capacity: 'Insufficient capacity', request_completed: 'Inference completed', request_failed: 'Inference failed', request_cancelled: 'Request cancelled', model_imported: 'Model imported', model_import_failed: 'Model import failed' };
  $('activity-list').innerHTML = events.map(event => {
    let detail = event.node || (event.chunks ? `${event.chunks.length} stage assignments changed` : String(event.request || event.model || '').slice(-32));
    if (event.kind === 'recovery') detail = `Epoch ${number(event.epoch)} · ${number(event.committed)} tokens retained`;
    if (event.kind === 'insufficient_capacity') detail = event.message || detail;
    if (event.trace) detail = `${list(event.trace).length} stages · ${duration(event.ms)} · request ${String(event.request || '').slice(-8)}`;
    const severity = /failed|lost/.test(event.kind) ? 'error' : /recovery|retry|insufficient/.test(event.kind) ? 'warning' : '';
    return `<div class="activity-item"><i class="activity-dot ${severity}"></i><div>${escapeHTML(event.kind === 'token_committed' ? `Token ${number(event.offset) + 1} committed` : titles[event.kind] || event.kind)}<small>${escapeHTML(detail)}</small></div><time>${timeLabel(event.time)}</time></div>`;
  }).join('') || empty(operationsOnly ? 'No operations events yet' : 'No events yet', 'Worker changes, placement updates, and inference activity appear here.');
}
function renderImports() {
  $('imports-panel').hidden = !state.jobs.length;
  $('imports-list').innerHTML = [...state.jobs].reverse().slice(0, 10).map(job => `<div class="import-row"><div><strong class="truncate" title="${escapeHTML(job.source)}">${escapeHTML(job.source)}</strong><small>${escapeHTML(job.error || (job.status === 'running' ? 'Downloading and verifying model artifacts' : job.model || ''))}</small></div>${statusChip(job.status)}<span class="muted">${timeLabel(job.started)}</span></div>`).join('');
}
function detailCards(items) { return `<div class="detail-grid">${items.map(([label, value]) => `<div><span>${escapeHTML(label)}</span><strong>${escapeHTML(value)}</strong></div>`).join('')}</div>`; }
function showDetail(kind, id) { detailSelection = { kind, id }; renderDetail(); openDialog('detail-dialog'); }
function renderDetail() {
  if (!detailSelection) return;
  if (detailSelection.kind === 'worker') {
    const node = state.nodes.find(node => node.id === detailSelection.id);
    if (!node) { $('detail-body').innerHTML = empty('Worker no longer in the snapshot', 'Close this panel and refresh the inventory.'); return; }
    const metrics = node.metrics || {};
    $('detail-title').textContent = node.id;
    const stageTimings = Object.entries(metrics.stage_ms || {});
    $('detail-body').innerHTML = `<div class="detail-subtitle">${escapeHTML(node.url)} · ${escapeHTML(node.transport)} · ${statusChip(workerStatus(node))}</div><div class="row-actions">${workerAction(node)}</div>${detailCards([
      ['Published memory', formatBytes(node.memory_bytes)], ['Available memory', formatBytes(metrics.available_memory_bytes)], ['Resident process memory', formatBytes(metrics.rss_bytes)],
      ['Published CPU', `${number(node.cpu_cores ?? metrics.cpu_limit_cores).toFixed(3)} cores`], ['Available CPU', metrics.available_cpu_cores == null ? 'Unreported' : `${number(metrics.available_cpu_cores).toFixed(3)} cores`], ['Free storage', formatBytes(metrics.available_storage_bytes)],
      ['Loaded weight memory', formatBytes(metrics.loaded_weight_bytes)], ['Cached artifacts', formatBytes(metrics.cache_bytes)], ['Heartbeat', timeLabel(node.last_seen)],
      ['Stage calls', formatCount(metrics.calls)], ['Active sessions', formatCount(metrics.sessions)], ['Failure domain', node.failure_domain || 'Unspecified'],
    ])}<div class="detail-section-title">Loaded stages (${list(node.ready).length})</div><div>${list(node.ready).slice(0, 150).map(key => `<span class="stage-chip" title="${escapeHTML(key)}">${escapeHTML(key)}</span>`).join('') || '<span class="muted">No stages loaded.</span>'}${list(node.ready).length > 150 ? '<p class="muted">Showing the first 150 loaded stages.</p>' : ''}</div>${stageTimings.length ? `<div class="detail-section-title">Latest stage execution time</div><div class="table-scroll"><table><thead><tr><th>STAGE</th><th>TIME</th></tr></thead><tbody>${stageTimings.slice(0, 100).map(([stage, ms]) => `<tr><td>${escapeHTML(stage)}</td><td>${duration(ms)}</td></tr>`).join('')}</tbody></table></div>` : ''}${Object.keys(metrics.errors || {}).length ? `<div class="detail-section-title">Load errors</div><div class="error-note">${escapeHTML(Object.entries(metrics.errors).map(([stage, error]) => `${stage}: ${error}`).join('\n'))}</div>` : ''}`;
  } else {
    const request = state.requests.find(request => request.id === detailSelection.id);
    if (!request) { $('detail-body').innerHTML = empty('Request no longer in retained history', 'The executor retains a bounded recent request history.'); return; }
    $('detail-title').textContent = 'Inference request';
    const trace = list(request.last_trace), route = list(request.route);
    $('detail-body').innerHTML = `<div class="detail-subtitle">${escapeHTML(request.id)}<br>${escapeHTML(request.model)}</div>${detailCards([
      ['Status', statusLabels[request.status] || request.status], ['Elapsed', duration(elapsed(request))], ['First token', duration(request.ttft_ms)],
      ['Prompt tokens', formatCount(request.prompt_tokens)], ['Output tokens', formatCount(request.output_tokens)], ['Recoveries', formatCount(request.recoveries)],
      ['Started', timeLabel(request.started)], ['Finished', timeLabel(request.finished)], ['Boundary traffic estimate', formatBytes(request.boundary_bytes)], ['Tenant', request.tenant || 'default'],
    ])}${request.error ? `<p class="error-note">${escapeHTML(request.error)}</p>` : ''}<div class="detail-section-title">Current route (${route.length} stages)</div><div class="detail-route">${route.slice(0, 200).map(hop => `<span class="stage-chip" title="${escapeHTML(hop.node)}">${escapeHTML(hop.chunk)} · ${escapeHTML(hop.node)}</span>`).join('<span>→</span>') || '<span class="muted">Route telemetry is not available yet.</span>'}</div>${route.length > 200 ? '<p class="muted">Showing the first 200 route stages.</p>' : ''}<div class="detail-section-title">Last completed token trace</div>${trace.length ? `<div class="table-scroll"><table><thead><tr><th>STAGE</th><th>WORKER</th><th>TIME</th><th>OUTPUT SHAPE</th></tr></thead><tbody>${trace.slice(0, 200).map(hop => `<tr><td>${escapeHTML(hop.chunk)}</td><td>${escapeHTML(hop.node)}</td><td>${duration(hop.ms)}</td><td>${escapeHTML(list(hop.output_shape).join(' × '))}</td></tr>`).join('')}</tbody></table></div>` : '<p class="muted">Stage timings arrive after a complete token route succeeds.</p>'}`;
  }
}

// Event delegation keeps controls stable as new snapshots replace table rows.
document.addEventListener('click', async event => {
  const button = event.target.closest('button');
  if (!button) return;
  if (button.dataset.close) $(button.dataset.close).close();
  if (button.dataset.worker) showDetail('worker', button.dataset.worker);
  if (button.dataset.request) showDetail('request', button.dataset.request);
  if (button.dataset.drain || button.dataset.resume) {
    const resume = Boolean(button.dataset.resume), nodeId = button.dataset.resume || button.dataset.drain;
    button.disabled = true;
    try { await api(`/api/nodes/${encodeURIComponent(nodeId)}/${resume ? 'resume' : 'drain'}`, 'POST'); showNotice(resume ? `Worker ${nodeId} resumed. It is available for model placement again.` : `Worker ${nodeId} is draining. Existing sessions can finish while new placement uses other workers.`); await refresh(); }
    catch (error) { showNotice(error.message, 'action', true); button.disabled = false; }
  }
});
$('auth-button').onclick = $('header-auth').onclick = showAccess;
$('auth-form').onsubmit = event => { event.preventDefault(); setToken($('token').value); $('auth-dialog').close(); };
$('clear-token').onclick = () => { $('token').value = ''; setToken(''); $('auth-dialog').close(); };
$('refresh-button').onclick = () => { if (authBlocked) showAccess(); else refresh(); };
$('open-import').onclick = () => openDialog('import-dialog');
$('model-select').onchange = () => { selectedModel = $('model-select').value; pages.stages = 0; lastTrace = []; traceTime = 0; graphSignature = ''; render(); };
for (const id of ['replicas', 'groups']) $(id).addEventListener('input', () => { deploymentFormDirty = true; });
for (const [control, key] of [['node-search', 'nodes'], ['node-filter', 'nodes'], ['request-filter', 'requests']]) $(control).addEventListener(control === 'node-search' ? 'input' : 'change', () => { pages[key] = 0; scheduleRender(); });
$('activity-filter').onchange = renderEvents;
for (const [name, key] of [['node', 'nodes'], ['request', 'requests'], ['stage', 'stages']]) {
  $(`${name}-prev`).onclick = () => { pages[key] = Math.max(0, pages[key] - 1); render(); };
  $(`${name}-next`).onclick = () => { pages[key]++; render(); };
}
document.querySelectorAll('nav a').forEach(link => link.addEventListener('click', () => {
  document.querySelectorAll('nav a').forEach(item => item.classList.toggle('selected', item === link));
}));
$('import-kind').onchange = () => {
  const bundle = $('import-kind').value === 'bundle';
  $('layers-field').hidden = bundle; $('sha-field').hidden = !bundle;
  $('import-layers').disabled = bundle;
  $('import-source').placeholder = bundle ? 'https://models.example/model.stageplane' : 'Qwen/Qwen3-4B-Instruct-2507';
  $('start-import').textContent = bundle ? 'Import verified chunks' : 'Download & partition';
};
$('import-form').onsubmit = async event => {
  event.preventDefault(); if (importPending) return;
  importPending = true; $('start-import').disabled = true;
  try {
    const bundle = $('import-kind').value === 'bundle';
    const job = await api('/api/models/import', 'POST', { kind: bundle ? 'bundle' : 'checkpoint', source: $('import-source').value.trim(), revision: $('import-revision').value.trim(), layers_per_chunk: number($('import-layers').value, 1), ...(bundle && $('import-sha').value.trim() ? { manifest_sha256: $('import-sha').value.trim() } : {}) });
    $('import-status').textContent = `Import ${String(job.id).slice(0, 8)} started. You can close this dialog; progress is shown in Model imports.`;
    await refresh();
  } catch (error) { $('import-status').textContent = error.message; }
  finally { importPending = false; $('start-import').disabled = false; }
};
$('deploy').onclick = async () => {
  if (!selectedModel) return;
  const replicas = number($('replicas').value), groups = number($('groups').value);
  if (!Number.isInteger(replicas) || replicas < 1 || replicas > 32 || !Number.isInteger(groups) || groups < 1 || groups > 100) { showNotice('Choose 1–32 replicas and 1–100 placement groups.', 'action', true); return; }
  $('deploy').disabled = true;
  try { await api('/api/deployments', 'POST', { model: selectedModel, replicas, groups }); deploymentFormDirty = false; showNotice('Placement accepted. Workers are verifying and loading their assigned stages.'); }
  catch (error) { showNotice(error.message, 'action', true); }
  finally { await refresh(); }
};
$('stop').onclick = () => runController?.abort();
$('run').onclick = async () => {
  if (running || !routeReady(currentModel())) return;
  const prompt = $('prompt').value, maxTokens = number($('max-tokens').value);
  if (!prompt.trim() || !Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 2048) { showNotice('Enter a prompt and choose 1–2,048 output tokens.', 'action', true); return; }
  running = true; runController = new AbortController();
  $('run').disabled = true; $('stop').hidden = false; $('response').textContent = ''; $('response-state').textContent = 'Generating';
  $('request-stats').textContent = 'Waiting for the first committed token…';
  let reader, finished = false;
  try {
    const response = await fetch('/v1/chat/completions', { method: 'POST', signal: runController.signal,
      headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({ model: selectedModel, messages: [{ role: 'user', content: prompt }], max_tokens: maxTokens, temperature: 0, stream: true }),
    });
    if (response.status === 401) authFailure(401);
    if (!response.ok) throw await responseError(response);
    if (!response.body) throw new Error('The executor did not return a response stream.');
    reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
    const processBlock = block => {
      const data = block.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trimStart()).join('\n');
      if (!data) return;
      if (data === '[DONE]') { finished = true; return; }
      const payload = JSON.parse(data);
      if (payload.error) throw new Error(payload.error.message || 'Inference failed');
      const content = payload.choices?.[0]?.delta?.content || '';
      $('response').textContent += content;
      if (content) $('request-stats').textContent = 'Streaming response…';
      if (payload.choices?.[0]?.finish_reason) finished = true;
      if (payload.usage) $('request-stats').textContent = `${formatCount(payload.usage.prompt_tokens)} prompt tokens · ${formatCount(payload.usage.completion_tokens)} output tokens`;
    };
    while (true) {
      const { done, value } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const blocks = pending.split(/\r?\n\r?\n/); pending = blocks.pop();
      for (const block of blocks) processBlock(block);
      if (pending.length > 2 ** 20) throw new Error('The response stream contains an oversized event.');
      if (done) { if (pending.trim()) processBlock(pending); break; }
    }
    if (!finished) throw new Error('The stream ended before completion. Inspect the request status before retrying.');
    $('response-state').textContent = 'Complete';
  } catch (error) {
    const cancelled = error.name === 'AbortError';
    $('response-state').textContent = cancelled ? 'Cancelled' : 'Failed';
    $('request-stats').textContent = cancelled ? 'Request cancelled from this console.' : error.message;
    if (!$('response').textContent) $('response').textContent = cancelled ? 'No output committed before cancellation.' : error.message;
  } finally {
    if (reader) await reader.cancel().catch(() => {});
    running = false; runController = null; $('stop').hidden = true; await refresh(); render();
  }
};

// The graph is a bounded view of inventory; exact counts remain in the tables.
const GRAPH_NODE_LIMIT = 64, GRAPH_STAGE_LIMIT = 6, GRAPH_EDGE_LIMIT = 160;
async function initGraph() {
  try {
    const THREE = await import('/ui/vendor/three.module.min.js');
    const box = $('graph'), scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, .1, 200);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5)); box.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xffffff, 0xa0b5d8, 3));
    const light = new THREE.DirectionalLight(0xffffff, 3); light.position.set(4, 8, 5); scene.add(light);
    const group = new THREE.Group(); scene.add(group);
    const packet = new THREE.Mesh(new THREE.SphereGeometry(.075, 10, 10), new THREE.MeshBasicMaterial({ color: 0xd9a24b })); packet.visible = false; scene.add(packet);
    const grid = new THREE.GridHelper(32, 40, 0xdce5f4, 0xeaf0fa); grid.position.y = -.75; scene.add(grid);
    graph = { THREE, scene, camera, renderer, group, packet, positions: new Map(), nodeMeshes: [], stageMeshes: [], rotation: -.25, zoom: 14, targetZoom: 14 };
    let dragging = false, lastX = 0, moved = false;
    renderer.domElement.addEventListener('pointerdown', event => { dragging = true; moved = false; lastX = event.clientX; renderer.domElement.setPointerCapture(event.pointerId); });
    renderer.domElement.addEventListener('pointermove', event => { if (!dragging) return; if (Math.abs(event.clientX - lastX) > 1) moved = true; graph.rotation += (event.clientX - lastX) * .008; lastX = event.clientX; graphDirty = true; });
    renderer.domElement.addEventListener('pointerup', event => { dragging = false; if (!moved) selectGraphNode(event); });
    renderer.domElement.addEventListener('pointercancel', () => { dragging = false; });
    renderer.domElement.addEventListener('wheel', event => { event.preventDefault(); graph.zoom = Math.max(7, Math.min(65, graph.zoom + event.deltaY * .015)); graphDirty = true; }, { passive: false });
    renderer.domElement.addEventListener('webglcontextlost', event => { event.preventDefault(); graphUnavailable = true; $('graph-empty').hidden = false; $('graph-empty').textContent = 'The 3D context was lost. Worker and stage tables remain available. Refresh the page to restore the graph.'; });
    $('reset-camera').onclick = () => { graph.rotation = -.25; graph.zoom = graph.targetZoom; graphDirty = true; };
    new ResizeObserver(() => { if (!box.clientWidth || !box.clientHeight) return; camera.aspect = box.clientWidth / box.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(box.clientWidth, box.clientHeight); graphDirty = true; }).observe(box);
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    function animate() {
      if (disposed) return;
      requestAnimationFrame(animate);
      if (document.hidden || graphUnavailable) return;
      const elapsed = performance.now() - traceTime;
      const live = elapsed < 1800 && lastTrace.length > 0 && !reducedMotion.matches;
      if (!graphDirty && !live) return;
      graphDirty = live;
      camera.position.set(Math.sin(graph.rotation) * graph.zoom, graph.zoom * .65, Math.cos(graph.rotation) * graph.zoom); camera.lookAt(0, 0, 0);
      const progress = Math.min(elapsed / 1800, 1) * Math.max(lastTrace.length - 1, 0), index = Math.max(0, Math.floor(progress));
      const hop = lastTrace[index], next = lastTrace[Math.min(index + 1, lastTrace.length - 1)];
      for (const mesh of graph.nodeMeshes) { const active = live && hop?.node === mesh.userData.id; mesh.material.emissive.setHex(active ? 0x4d80b6 : 0); mesh.material.emissiveIntensity = active ? .35 : 0; }
      for (const mesh of graph.stageMeshes) mesh.material.color.setHex(live && hop?.node === mesh.userData.id && hop?.chunk === mesh.userData.chunk ? 0xd9a24b : 0xa8cfc3);
      packet.visible = Boolean(live && graph.positions.has(hop?.node) && graph.positions.has(next?.node));
      if (packet.visible) { packet.position.copy(graph.positions.get(hop.node)).lerp(graph.positions.get(next.node), progress - index); packet.position.y = .4; }
      renderer.render(scene, camera);
    }
    animate(); buildGraph();
  } catch (error) {
    graphUnavailable = true; $('graph-empty').hidden = false;
    $('graph-empty').textContent = 'The 3D view is unavailable in this browser. Worker inventory and stage placement are available in the tables.';
    $('reset-camera').disabled = true;
    console.warn('Stageplane topology view unavailable:', error.message);
  }
}
function graphLabel(text, color = '#667fa7') {
  const { THREE } = graph;
  const canvas = document.createElement('canvas'); canvas.width = 384; canvas.height = 96;
  const context = canvas.getContext('2d'); context.font = '500 28px -apple-system, sans-serif'; context.textAlign = 'center'; context.fillStyle = color; context.fillText(text, 192, 42);
  const texture = new THREE.CanvasTexture(canvas), sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false })); sprite.scale.set(2.2, .55, 1); return sprite;
}
function buildGraph() {
  if (!graph || graphUnavailable) return;
  const model = currentModel(), deployment = model ? state.deployments[model.key] : null;
  const nodes = [...state.nodes].sort((a, b) => Number(b.active) - Number(a.active) || String(a.id).localeCompare(String(b.id))).slice(0, GRAPH_NODE_LIMIT);
  const signature = JSON.stringify([nodes.map(node => [node.id, node.active, node.draining, list(node.ready)]), model?.key, deployment?.placements]);
  if (signature === graphSignature) return;
  graphSignature = signature; graphDirty = true;
  const { THREE, group } = graph;
  group.traverse(object => { object.geometry?.dispose(); if (object.material) { object.material.map?.dispose(); object.material.dispose(); } }); group.clear();
  graph.nodeMeshes = []; graph.stageMeshes = []; graph.positions.clear();
  $('graph-empty').hidden = nodes.length > 0;
  const limited = state.nodes.length > GRAPH_NODE_LIMIT || nodes.some(node => list(node.ready).filter(key => !model || key.startsWith(`${model.key}:`)).length > GRAPH_STAGE_LIMIT);
  $('graph-subset').hidden = !limited;
  $('graph-subset').textContent = `Showing ${nodes.length} of ${formatCount(state.nodes.length)} workers, up to ${GRAPH_STAGE_LIMIT} stage slabs per worker. Exact inventory and replica counts are available in the tables. Links show a representative placement path.`;
  const large = nodes.length > 10;
  const columns = Math.ceil(Math.sqrt(nodes.length));
  nodes.forEach((node, index) => {
    const angle = index / Math.max(nodes.length, 1) * Math.PI * 2;
    const position = large ? new THREE.Vector3((index % columns - (columns - 1) / 2) * 2.25, 0, (Math.floor(index / columns) - (Math.ceil(nodes.length / columns) - 1) / 2) * 2.2) : new THREE.Vector3(Math.cos(angle) * 3.35, 0, Math.sin(angle) * 2.8);
    graph.positions.set(node.id, position);
    const material = new THREE.MeshStandardMaterial({ color: node.active ? node.draining ? 0xc5b38a : 0x8aa8d7 : 0xc9d1df, metalness: .12, roughness: .6 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1.45, .2, 1.05), material); mesh.position.copy(position); mesh.userData = { id: node.id }; group.add(mesh); graph.nodeMeshes.push(mesh);
    const stages = list(node.ready).filter(key => !model || key.startsWith(`${model.key}:`));
    stages.slice(0, GRAPH_STAGE_LIMIT).forEach((key, offset) => { const slab = new THREE.Mesh(new THREE.BoxGeometry(1.13, .10, .8), new THREE.MeshStandardMaterial({ color: 0xa8cfc3, roughness: .7 })); slab.position.set(position.x, .2 + offset * .15, position.z); slab.userData = { id: node.id, chunk: key.split(':').pop() }; graph.stageMeshes.push(slab); group.add(slab); });
    const title = graphLabel(String(node.id).length > 20 ? `${String(node.id).slice(0, 18)}…` : node.id); title.position.set(position.x, -.35, position.z); group.add(title);
    const count = graphLabel(`${stages.length} stages`, '#9bacc5'); count.scale.set(1.7, .43, 1); count.position.set(position.x, -.62, position.z); group.add(count);
  });
  const edges = new Set();
  if (model && deployment) {
    let previous;
    for (const stage of list(model.stages)) {
      const next = list(deployment.placements?.[stage.id]).find(id => graph.positions.has(id));
      if (previous && next && previous !== next && edges.size < GRAPH_EDGE_LIMIT) {
        const key = JSON.stringify([previous, next].sort());
        if (!edges.has(key)) { edges.add(key); const points = [graph.positions.get(previous).clone().add(new THREE.Vector3(0, .2, 0)), graph.positions.get(next).clone().add(new THREE.Vector3(0, .2, 0))]; group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: 0xb6c8e5, transparent: true, opacity: .7 }))); }
      }
      previous = next;
    }
  }
  const targetZoom = large ? Math.max(17, columns * 4.1) : 14;
  if (graph.targetZoom !== targetZoom) graph.zoom = targetZoom;
  graph.targetZoom = targetZoom;
}
function selectGraphNode(event) {
  const { THREE, renderer, camera, group } = graph;
  const bounds = renderer.domElement.getBoundingClientRect(), pointer = new THREE.Vector2((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(pointer, camera);
  const hit = ray.intersectObjects(group.children).find(item => item.object.userData.id);
  if (!hit) { $('graph-tooltip').hidden = true; return; }
  const node = state.nodes.find(node => node.id === hit.object.userData.id);
  if (!node) return;
  $('graph-tooltip').innerHTML = `<strong>${escapeHTML(node.id)}</strong> · ${escapeHTML(node.transport)}<br>${formatBytes(node.metrics?.rss_bytes)} RSS · ${list(node.ready).length} loaded stages`;
  $('graph-tooltip').hidden = false; showDetail('worker', node.id);
}
window.addEventListener('pageshow', event => { if (event.persisted && disposed) location.reload(); });
window.addEventListener('online', () => { reconnectAttempt = 0; refresh(); connect(); });
document.addEventListener('visibilitychange', () => { if (!document.hidden) { graphDirty = true; refresh(); renderConnection(); } });
window.addEventListener('pagehide', () => { disposed = true; clearTimeout(pollTimer); clearTimeout(renderTimer); stopSocket(); runController?.abort(); graph?.renderer.dispose(); });
poll(); connect(); initGraph();
