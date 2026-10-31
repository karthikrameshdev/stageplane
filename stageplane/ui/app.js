const $ = id => document.getElementById(id);
const esc = text => String(text ?? '').replace(/[&<>"']/g, x => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
let state = {nodes:[],models:[],deployments:{},events:[],requests:[]}, socket, running = false;
let token = sessionStorage.getItem('stageplane-token') || '';
let lastCapacityMessage = '';
const capacityBanner = document.createElement('div');capacityBanner.className='capacity-banner';capacityBanner.hidden=true;document.querySelector('.metrics').before(capacityBanner);
let activeNodes = new Set(), selectedNode, lastTrace = [], traceTime = 0;
const bytes = n => n >= 1024**3 ? `${(n / 1024**3).toFixed(2)} GiB` : `${(n / 1024**2).toFixed(1)} MiB`;
async function api(path, method='GET', body) {
  const r = await fetch(path, {method, headers:{'Content-Type':'application/json',...(token ? {Authorization:`Bearer ${token}`} : {})},...(body ? {body:JSON.stringify(body)} : {})});
  if (r.status===401) $('auth-dialog').showModal();
  if (!r.ok) throw new Error((await r.json()).detail || `HTTP ${r.status}`);
  return r.json();
}
function currentModel() { return state.models.find(m=>m.key===$('model-select').value) || state.models[0]; }
function render() {
  const active = state.nodes.filter(n=>n.active), m = currentModel();
  $('active-count').innerHTML = `${active.length}<span> / ${state.nodes.length}</span>`;
  $('nav-count').textContent = state.nodes.length;
  $('node-badge').textContent = state.nodes.length;
  $('stage-count').textContent = m?.stages.length || 0;
  $('model-name').textContent = m ? `${m.source.split('/').pop()} · ${m.architecture}` : 'No model registered';
  $('memory').textContent = active.length ? bytes(active.reduce((v,n)=>v+(n.metrics?.rss_bytes||0),0)) : '—';
  const recent = [...state.requests].reverse().find(r=>r.ttft_ms);
  $('ttft').innerHTML = recent ? `${Math.round(recent.ttft_ms)}<span> ms</span>` : '—';
  let ready = false;
  if(m) ready = m.stages.every(s=>(state.deployments[m.key]?.placements[s.id]||[]).some(id=>state.nodes.find(n=>n.id===id)?.ready.includes(`${m.key}:${s.id}`)));
  const caution=m ? state.deployments[m.key]?.capacity?.status==='insufficient_capacity' : false;
  const message=caution ? state.deployments[m.key].capacity.message : '';
  capacityBanner.hidden=!message;capacityBanner.textContent=message;
  if(message && message!==lastCapacityMessage) console.warn(message);lastCapacityMessage=message;
  if(!running) $('run').disabled=!ready;
  $('cluster-status').textContent = ready ? 'Complete route ready' : active.length ? 'Stages loading or undeployed' : 'Waiting for workers';
  const old = $('model-select').value;
  if ($('model-select').options.length!==state.models.length || !state.models.some(m=>m.key===$('model-select').options[0]?.value)) {
    $('model-select').innerHTML = state.models.map(m=>`<option value="${esc(m.key)}">${esc(m.source.split('/').pop())} · ${esc(m.revision.slice(0,8))}</option>`).join('') || '<option value="">No models available</option>';
    if(state.models.some(m=>m.key===old)) $('model-select').value=old;
  }
  $('node-table').innerHTML = state.nodes.map(n=>{
    const chunks = n.ready.map(k=>k.split(':').pop());
    const status = !n.active ? 'Offline' : n.draining ? 'Draining' : n.metrics?.errors && Object.keys(n.metrics.errors).length ? 'Load error' : 'Active';
    return `<tr><td><strong>${esc(n.id)}</strong><small>${esc(n.url)}</small></td><td><span class="node-status ${!n.active?'offline':n.draining?'draining':''}">${status}</span></td><td title="${esc(chunks.join(', '))}">${chunks.slice(0,4).map(c=>`<span class="stage-chip">${esc(c)}</span>`).join('')}${chunks.length>4?`<span class="stage-chip">+${chunks.length-4}</span>`:''}${chunks.length?'':'<span class="muted">Awaiting assignment</span>'}</td><td>${bytes(n.metrics?.rss_bytes||0)}<small>${n.metrics?.dtype||''} · ${n.metrics?.sessions||0} sessions</small><small>Limit ${bytes(n.memory_bytes)} · ${Number(n.cpu_cores ?? n.metrics?.cpu_limit_cores ?? 0).toFixed(2)} CPU cores</small><small>CPU usage ${(n.metrics?.cpu_usage_percent||0).toFixed(1)}%</small></td><td>${esc(n.transport)}</td><td>${n.group}</td><td><button class="drain" data-node="${esc(n.id)}" ${n.draining||!n.active?'disabled':''}>Drain ↗</button></td></tr>`;
  }).join('') || '<tr><td colspan="7" class="empty">No workers registered. Start a node to join this executor.</td></tr>';
  document.querySelectorAll('.drain').forEach(b=>b.onclick=async()=>{try{await api(`/api/nodes/${b.dataset.node}/drain`,'POST');await refresh();}catch(e){showError(e);}});
  $('stage-list').innerHTML = m ? m.stages.map(s=>{
    const owners = state.deployments[m.key]?.placements[s.id]||[];
    return `<div class="stage-row"><div><b>${esc(s.id)}</b><small>${s.kind==='blocks'?`Layers ${s.start}–${s.end-1}`:esc(s.kind)} · ${bytes(s.weight_bytes)}</small></div><div>${owners.map(id=>`<span class="stage-chip">${esc(id)}</span>`).join('')||'<span class="muted">Unassigned</span>'}</div><span>${owners.length} replica${owners.length===1?'':'s'}</span></div>`;
  }).join('') : '<div class="empty">No model stages yet.</div>';
  renderEvents();
  if(scene) buildGraph();
}
function renderEvents() {
  const events = [...state.events].reverse().slice(0,35);
  $('activity-list').innerHTML=events.map(e=>{
    const title = {'node_joined':'Worker joined','node_lost':'Worker disconnected','placement':'Placement updated','request_started':'Inference admitted','token_committed':`Token ${(e.offset||0)+1} committed`,'recovery':'Restoring committed prefix','chunk_retry':'Retrying on a replacement route','insufficient_capacity':'Insufficient node capacity','request_completed':'Inference completed','request_failed':'Inference failed','request_cancelled':'Request cancelled'}[e.kind]||e.kind;
    let detail = e.node || (e.chunks ? `${e.chunks.length} stages reassigned` : e.request?.slice(-12) || '');
    if(e.trace) detail = e.trace.map(t=>`${t.chunk} → ${t.node} ${t.ms.toFixed(1)}ms`).join(' · ');
    if(e.kind==='recovery') detail=`Epoch ${e.epoch} · ${e.committed} committed tokens`;
    return `<div class="activity-item"><i class="activity-dot"></i><div>${esc(title)}<small>${esc(detail)}</small></div><time>${new Date(e.time*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}</time></div>`;
  }).join('')||'<div class="empty">Waiting for cluster activity.</div>';
}
async function refresh() { try {state = await api('/api/state');render();} catch(e){$('connection').textContent='Disconnected';} }
function connect() {
  if(socket) socket.close();
  socket = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/events`);
  socket.onopen=()=>{socket.send(JSON.stringify({token}));$('connection').textContent='Executor connected';};
  socket.onmessage=event=>{
    const e=JSON.parse(event.data);
    if(e.kind==='snapshot') {state=e.state;render();return;}
    state.events.push(e);state.events=state.events.slice(-400);
    if(e.kind==='token_committed') {
      lastTrace=e.trace;traceTime=performance.now();activeNodes=new Set(e.trace.map(t=>t.node));
      $('graph-caption').textContent=`Token ${e.offset+1} · ${e.ms.toFixed(0)} ms`;
      $('request-stats').textContent=`Token ${e.offset+1} · ${e.trace.length} stages · ${e.ms.toFixed(0)} ms`;
    }
    renderEvents();
    if(['placement','node_joined','node_lost','request_completed'].includes(e.kind)) refresh();
  };
  socket.onclose=()=>{$('connection').textContent='Reconnecting';setTimeout(connect,3000);};
}
function showError(e){$('response').textContent=e.message;$('response-state').textContent='Error';}
$('auth-button').onclick=()=>$('auth-dialog').showModal();
$('save-token').onclick=()=>{token=$('token').value;sessionStorage.setItem('stageplane-token',token);setTimeout(()=>{refresh();connect();},0);};
$('model-select').onchange=()=>render();
const importButton = document.createElement('button');
importButton.className='import-model';importButton.textContent='+ Import a model';
$('model-select').after(importButton);
document.body.insertAdjacentHTML('beforeend', `<dialog id="import-dialog"><form id="import-form"><h2>Import open weights</h2><p>Import a checkpoint, or reuse a published bundle from HTTP, Git, Drive, or a local path.</p><label>Import type<select id="import-kind"><option value="checkpoint">Original checkpoint · download and chunk</option><option value="bundle">Published Stageplane bundle · reuse chunks</option></select></label><label>Model source<input id="import-source" required placeholder="Qwen/Qwen3.5-0.8B"></label><label>Revision<input id="import-revision" value="main" required></label><label>Layers per chunk<input id="import-layers" type="number" min="1" max="256" value="2" required></label><label>Manifest SHA-256 (optional for bundles)<input id="import-sha" pattern="[0-9a-f]{64}" placeholder="Publisher’s verified manifest digest"></label><div id="import-status" aria-live="polite"></div><div class="dialog-actions"><button type="button" id="close-import">Close</button><button type="submit" class="primary" id="start-import">Download & partition</button></div></form></dialog>`);
importButton.onclick=()=>$('import-dialog').showModal();
$('close-import').onclick=()=>$('import-dialog').close();
$('import-kind').onchange=()=>{
  const bundle=$('import-kind').value==='bundle';
  $('import-layers').disabled=bundle;
  $('import-layers').parentElement.style.display=bundle?'none':'';
  $('import-source').placeholder=bundle?'https://models.example/model.stageplane':'Qwen/Qwen3-4B-Instruct-2507';
  $('start-import').textContent=bundle?'Import verified chunks':'Download & partition';
};
$('import-form').onsubmit=async e=>{
  e.preventDefault();$('start-import').disabled=true;
  try {
    const job=await api('/api/models/import','POST',{kind:$('import-kind').value,...($('import-sha').value ? {manifest_sha256:$('import-sha').value} : {}),source:$('import-source').value,revision:$('import-revision').value,layers_per_chunk:Number($('import-layers').value)});
    $('import-status').textContent='Downloading and verifying artifacts. This can take several minutes.';
    const poll=async()=>{try{const result=await api(`/api/jobs/${job.id}`);if(result.status==='running'){setTimeout(poll,3000);return;}$('start-import').disabled=false;if(result.status==='failed'){$('import-status').textContent=result.error;return;}$('import-status').textContent='Model imported. Select it and deploy stages to your workers.';await refresh();$('model-select').value=result.model;render();}catch(error){$('import-status').textContent=error.message;$('start-import').disabled=false;}};
    poll();
  }catch(error){$('import-status').textContent=error.message;$('start-import').disabled=false;}
};
$('deploy').onclick=async()=>{
  try {await api('/api/deployments','POST',{model:$('model-select').value,replicas:Number($('replicas').value),groups:Number($('groups').value)});await refresh();}catch(e){showError(e);}
};
$('run').onclick=async()=>{
  if(running) return;
  running=true;$('run').disabled=true;$('response').textContent='';$('response-state').textContent='Generating';
  try {
    const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json',...(token?{Authorization:`Bearer ${token}`}:{})},body:JSON.stringify({model:$('model-select').value,messages:[{role:'user',content:$('prompt').value}],max_tokens:Number($('max-tokens').value),temperature:0,stream:true})});
    if(!r.ok)throw Error((await r.json()).detail||`HTTP ${r.status}`);
    const reader=r.body.getReader(),decoder=new TextDecoder();let pending='';
    while(true){const {done,value}=await reader.read();if(done)break;pending+=decoder.decode(value,{stream:true});const blocks=pending.split('\n\n');pending=blocks.pop();for(const block of blocks){if(!block.startsWith('data: ')||block==='data: [DONE]')continue;const e=JSON.parse(block.slice(6));if(e.error)throw Error(e.error.message);$('response').textContent+=e.choices?.[0]?.delta?.content||'';if(e.usage)$('request-stats').textContent=`${e.usage.prompt_tokens} prompt tokens · ${e.usage.completion_tokens} output tokens`;}}
    $('response-state').textContent='Complete';
  }catch(e){showError(e);}finally{running=false;$('run').disabled=false;await refresh();}
};

// The graph is driven entirely by the executor inventory and stage assignments.
let THREE,scene,camera,renderer,graphGroup,nodeMeshes=[],stageMeshes=[],graphPositions={},packet,signature='',rotation=-.25,zoom=13;
async function initGraph(){
  try {
    THREE=await import('/ui/vendor/three.module.min.js');
    const box=$('graph');scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(38,box.clientWidth/box.clientHeight,.1,100);
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(box.clientWidth,box.clientHeight);box.appendChild(renderer.domElement);
    scene.add(new THREE.HemisphereLight(0xffffff,0x809a92,3));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(4,8,5);scene.add(light);
    graphGroup=new THREE.Group();scene.add(graphGroup);
    packet=new THREE.Mesh(new THREE.SphereGeometry(.085,12,12),new THREE.MeshBasicMaterial({color:0xd9a24b}));packet.visible=false;scene.add(packet);
    const grid=new THREE.GridHelper(20,32,0xdce7e2,0xeaf0ed);grid.position.y=-1.1;scene.add(grid);
    let dragging=false,lastX=0,moved=false;
    renderer.domElement.onpointerdown=e=>{dragging=true;lastX=e.clientX;moved=false;renderer.domElement.setPointerCapture(e.pointerId);};
    renderer.domElement.onpointermove=e=>{if(dragging){if(Math.abs(e.clientX-lastX)>1)moved=true;rotation+=(e.clientX-lastX)*.008;lastX=e.clientX;}};
    renderer.domElement.onpointerup=e=>{dragging=false;if(!moved)select(e);};
    renderer.domElement.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(7,Math.min(30,zoom+e.deltaY*.008));},{passive:false});
    $('reset-camera').onclick=()=>{rotation=-.25;zoom=13;};
    new ResizeObserver(()=>{camera.aspect=box.clientWidth/box.clientHeight;camera.updateProjectionMatrix();renderer.setSize(box.clientWidth,box.clientHeight);}).observe(box);
    function animate(){
      requestAnimationFrame(animate);camera.position.set(Math.sin(rotation)*zoom,zoom*.62,Math.cos(rotation)*zoom);camera.lookAt(0,0,0);
      const elapsed=performance.now()-traceTime,progress=Math.min(elapsed/1800,1)*Math.max(lastTrace.length-1,0),index=Math.floor(progress),hop=lastTrace[index],next=lastTrace[Math.min(index+1,lastTrace.length-1)];
      const live=elapsed<1800&&hop;
      for(const mesh of nodeMeshes){const active=live&&hop.node===mesh.userData.id;mesh.material.emissive.setHex(active?0x3c9b78:0x000000);mesh.material.emissiveIntensity=active ? .25 : 0;}
      for(const mesh of stageMeshes){const active=live&&hop.node===mesh.userData.id&&hop.chunk===mesh.userData.chunk;mesh.material.color.setHex(active?0xd9a24b:0xa6cbbd);}
      packet.visible=Boolean(live&&graphPositions[hop.node]&&graphPositions[next.node]);
      if(packet.visible){packet.position.copy(graphPositions[hop.node]).lerp(graphPositions[next.node],progress-index);packet.position.y=.45;}
      renderer.render(scene,camera);
    }animate();buildGraph();
  }catch(e){$('graph-empty').textContent='3D view unavailable. Worker inventory and stage distribution remain available below.';console.warn(e);}
}
function label(text,color='#526c65'){
  const canvas=document.createElement('canvas');canvas.width=512;canvas.height=128;const ctx=canvas.getContext('2d');ctx.font='500 38px -apple-system, sans-serif';ctx.textAlign='center';ctx.fillStyle=color;ctx.fillText(text,256,58);const tex=new THREE.CanvasTexture(canvas);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,transparent:true,depthTest:false}));sprite.scale.set(2.4,.6,1);return sprite;
}
function buildGraph(){
  const m=currentModel(),d=m?state.deployments[m.key]:null;
  const next=JSON.stringify([state.nodes.map(n=>[n.id,n.active,n.ready]),m?.key,d?.placements]);
  if(next===signature)return;signature=next;
  graphGroup.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material){o.material.map?.dispose();o.material.dispose();}});graphGroup.clear();nodeMeshes=[];stageMeshes=[];
  $('graph-empty').style.display=state.nodes.length?'none':'block';
  const positions={};graphPositions=positions;
  state.nodes.forEach((n,i)=>{
    const angle=i/Math.max(state.nodes.length,1)*Math.PI*2;
    const pos=new THREE.Vector3(Math.cos(angle)*3.3,0,Math.sin(angle)*2.7);positions[n.id]=pos;
    const material=new THREE.MeshStandardMaterial({color:n.active?0x7cae9d:0xc5cbd0,metalness:.12,roughness:.5});
    const base=new THREE.Mesh(new THREE.BoxGeometry(1.55,.25,1.2),material);base.position.copy(pos);base.userData={id:n.id};graphGroup.add(base);nodeMeshes.push(base);
    const stages=n.ready.filter(k=>!m||k.startsWith(m.key+':'));
    stages.forEach((s,j)=>{const slab=new THREE.Mesh(new THREE.BoxGeometry(1.25,.12,.95),new THREE.MeshStandardMaterial({color:0xa6cbbd,roughness:.6}));slab.position.set(pos.x,pos.y+.25+j*.19,pos.z);slab.userData={id:n.id,chunk:s.split(':').pop()};stageMeshes.push(slab);graphGroup.add(slab);});
    const title=label(n.id);title.position.set(pos.x,-.5,pos.z);graphGroup.add(title);
    const count=label(`${stages.length} stages`, '#8fa69d');count.scale.set(1.8,.45,1);count.position.set(pos.x,-.83,pos.z);graphGroup.add(count);
  });
  const edges=new Set();
  if(m&&d){for(let i=0;i<m.stages.length-1;i++){const a=d.placements[m.stages[i].id]||[],b=d.placements[m.stages[i+1].id]||[];for(const x of a)for(const y of b){if(x===y||!positions[x]||!positions[y])continue;const key=[x,y].sort().join(':');if(edges.has(key))continue;edges.add(key);const points=[positions[x].clone().add(new THREE.Vector3(0,.3,0)),positions[y].clone().add(new THREE.Vector3(0,.3,0))];graphGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:0xc2d9ce,transparent:true,opacity:.8})));}}}
}
function select(e){
  const box=renderer.domElement.getBoundingClientRect(),pointer=new THREE.Vector2((e.clientX-box.left)/box.width*2-1,-(e.clientY-box.top)/box.height*2+1),ray=new THREE.Raycaster();ray.setFromCamera(pointer,camera);const hits=ray.intersectObjects(graphGroup.children).filter(x=>x.object.userData.id);const tooltip=$('graph-tooltip');if(!hits.length){tooltip.style.display='none';return;}const n=state.nodes.find(n=>n.id===hits[0].object.userData.id);tooltip.innerHTML=`<b>${esc(n.id)}</b> · ${esc(n.transport)} · ${bytes(n.metrics?.rss_bytes||0)}<br>${n.ready.map(k=>esc(k.split(':').pop())).join(' · ')||'No stages loaded'}`;tooltip.style.display='block';
}
await refresh();connect();initGraph();
