/* Standalone failure browser. No network or cross-case GPU geometry cache. */
(() => {
  "use strict";
  const M = window.MANIFEST;
  const $ = id => document.getElementById(id);
  const esc = x => String(x ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const finite = x => typeof x === "number" && Number.isFinite(x);
  const fmt = x => !finite(x) ? "未知" : x === 0 ? "0" : Math.abs(x) < 0.0001 ? x.toExponential(2) : x.toFixed(5).replace(/0+$/, "").replace(/\.$/, "");
  const num = x => Number.isInteger(x) ? x : null;
  const mean = xs => xs.length ? xs.reduce((a,b) => a+b, 0)/xs.length : null;
  const percentile = (xs, p) => { if (!xs.length) return null; const a=[...xs].sort((x,y)=>x-y), q=(a.length-1)*p, i=Math.floor(q); return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(q-i); };
  const stages = {trimming:"裁剪面失败",solid_construction:"实体构造失败",roundtrip_boundary:"STEP 检查失败"};
  const cohorts = {regressed:"32维成功 → 64维失败",both_failed:"两版均失败"};
  const sourceLabels = {fitted:"拟合后 · 末路径诊断重放",raw:"原始预测 · 64维",gt:"GT · 仅诊断参照",baseline:"来源模型 · 32维"};
  const reasons = {accepted:"通过",trim_rejected:"裁剪未通过",unexplained_wire_skip:"存在被丢弃的边界环",open_wire:"开放边界环",single_open_edge:"单条开放边",no_wire:"未组出边界环",no_wires:"未组出边界环",invalid_face:"裁剪面检查失败",boundary_incomplete:"边界不完整",exception:"异常",wire_construction_failed:"组环失败",wire_count_decreased_during_face_fix:"修复时丢环",no_input_edges:"无输入边",no_solid:"未构造出实体",native_rejected:"原生 STEP 合法性未通过",boundary_rejected:"STEP 边界检查未通过",no_accepted_solid:"没有通过验收的实体",native_accepted:"原生 STEP 通过"};
  const pathLabels = {legacy:"原始构造路径",roundtrip_optimized:"完整边界检查路径",fit_score_v2_roundtrip_optimized:"拟合候选评分路径（末路径）"};
  const colors = {face:"#a9b8c9",fail:"#c87572",selected:"#6091c7",edge:"#485a70",selectedEdge:"#c74c47",projection:"#da9039",endpoint:"#8464ad",valid:"#4a8e91",failed:"#cf685a"};
  const cache = new Map();
  let requestToken=0, renderToken=0, current=null, filtered=[], geometryBound=false;
  const state={id:null, source:"fitted", face:null, edge:null, path:0, distribution:"cdf", focus:"all", loading:false, camera:null, revision:0, error:null};
  const plotConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:["sendDataToCloud","toImage"],scrollZoom:true};
  const font={family:"Archivo, SourceHanSans, Microsoft YaHei, sans-serif",size:11,color:"#647082"};

  async function decodeCase(id){
    if(cache.has(id)){const value=cache.get(id);cache.delete(id);cache.set(id,value);return value;}
    const packed=window.CASE_DATA[id];
    if(!packed)throw new Error("该案例缺少内嵌几何数据");
    if(typeof DecompressionStream!=="function")throw new Error("此浏览器不支持本地 gzip 解压，请用现代 Chrome 或 Edge 打开。");
    const binary=Uint8Array.from(atob(packed),c=>c.charCodeAt(0));
    const text=await new Response(new Blob([binary]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
    const value=JSON.parse(text);
    if(value.id!==id)throw new Error("案例身份与内嵌几何不一致");
    cache.set(id,value);while(cache.size>3)cache.delete(cache.keys().next().value);return value;
  }
  function view(){return current?.views?.[state.source]||null;}
  function selectedMeta(){return M.cases.find(c=>c.id===state.id);}
  function isPoint(p){return Array.isArray(p)&&p.length>=3&&p.slice(0,3).every(finite);}
  function gridPoints(grid){return (grid||[]).flat().filter(isPoint);}
  function ownerEdges(v,face){
    if(!v)return[];if(face===null)return(v.edges||[]).map((_,i)=>i);
    return [...new Set((v.connectivity||[]).filter(c=>c[1]===face||c[2]===face).map(c=>c[0]))].filter(e=>Array.isArray(v.edges?.[e]));
  }
  function records(v=view()){
    return(v?.owner_records||[]).filter(r=>state.face===null||r.owner===state.face);
  }
  function selectedRecords(){return records().filter(r=>state.edge===null||r.edge===state.edge);}
  function bestRecord(){return [...selectedRecords()].filter(r=>finite(r.max)).sort((a,b)=>b.max-a.max)[0]||selectedRecords()[0]||null;}
  function failedFaces(){return selectedMeta()?.failed_faces||[];}
  function setMetric(id,value){$(id).textContent=fmt(value);$(id).classList.toggle("unknown",!finite(value));}
  function renderList(){
    const q=$("search").value.trim().toLowerCase(), stage=$("stage-filter").value,cohort=$("cohort-filter").value;
    filtered=M.cases.filter(c=>(stage==="all"||c.stage===stage)&&(cohort==="all"||c.cohort===cohort)&&(!q||[c.id,stages[c.stage],...(c.reasons||[]),...(c.reasons||[]).map(r=>reasons[r]||r)].join(" ").toLowerCase().includes(q)));
    const sort=$("sort").value;
    filtered.sort((a,b)=>sort==="id"?a.id.localeCompare(b.id):((finite(b[sort])?b[sort]:-Infinity)-(finite(a[sort])?a[sort]:-Infinity))||a.id.localeCompare(b.id));
    $("list-count").textContent=`${filtered.length} / ${M.cases.length} 件`;
    $("case-list").innerHTML=filtered.map(c=>`<button class="case-item" data-id="${esc(c.id)}" aria-current="${c.id===state.id}"><div class="case-top"><span class="case-id">${esc(c.id)}</span><span class="tag ${c.cohort==="regressed"?"warn":""}">${c.cohort==="regressed"?"退步":"持续失败"}</span></div><div class="case-bottom"><span>${esc(stages[c.stage]||c.stage)}</span><span title="拟合后最大边面距离">${fmt(c.replay_max)}</span></div></button>`).join("")||'<p class="empty">没有符合筛选条件的案例</p>';
    updateNav();
  }
  function updateNav(){const index=filtered.findIndex(c=>c.id===state.id);$("previous").disabled=index<=0;$("next").disabled=index<0||index>=filtered.length-1;$("case-position").textContent=index>=0?`${index+1} / ${filtered.length}`:"当前案例不在筛选结果中";}
  function renderHeading(){
    const c=selectedMeta();if(!c)return;
    $("case-title").textContent=c.id;$("case-stage").textContent=stages[c.stage]||c.stage;$("case-stage").className="tag fail";$("case-cohort").textContent=cohorts[c.cohort]||c.cohort;
    const replay=current.replay||{},hasFitted=!!current.views?.fitted,match=replay.matches_history;
    let text=hasFitted?(match===true?"日志一致的诊断重放：逐面裁剪记录已与原评估核对。三维显示末条恢复路径的拟合曲面和边。":"诊断重放尚未确认与历史一致；请结合原始预测和历史日志查看，不能把当前几何当作原评估快照。"):("拟合后几何未取得；当前可查看原始预测、GT和历史日志。"+(replay.error?` ${replay.error}`:""));
    if(match===false)text=`重放与历史不一致：${(replay.mismatch_fields||[]).join("、")||"详见诊断记录"}。原评估失败结论保持不变。`;
    $("replay-status").textContent=text;$("replay-status").className="status-banner"+(!hasFitted||match!==true?" warning":"");
    $("failed-faces").innerHTML=failedFaces().length?failedFaces().map(f=>`<button class="chip" data-face="${f}" aria-pressed="${state.face===f}">失败面 ${f}</button>`).join(""):'<span class="small muted">末路径各面裁剪均通过；失败发生在后续阶段。</span>';
  }
  function chooseWorstEdge(){
    const available=ownerEdges(view(),state.face),worst=records().filter(r=>available.includes(r.edge)&&finite(r.max)).sort((a,b)=>b.max-a.max)[0];
    state.edge=worst?worst.edge:(available[0]??null);
  }
  function renderSelectors(){
    $("source").innerHTML=Object.keys(sourceLabels).filter(k=>current.views?.[k]).map(k=>`<option value="${k}">${sourceLabels[k]}</option>`).join("");$("source").value=state.source;
    $("endpoint-mode-label").classList.toggle("hidden",state.source!=="fitted");
    const v=view(),fail=new Set(failedFaces());
    $("face").innerHTML='<option value="all">全部面</option>'+(v?.faces||[]).map((_,i)=>`<option value="${i}">面 ${i}${fail.has(i)?" · 裁剪失败":""}</option>`).join("");$("face").value=state.face===null?"all":String(state.face);
    const edges=ownerEdges(v,state.face);if(state.edge!==null&&!edges.includes(state.edge))chooseWorstEdge();
    $("edge").innerHTML='<option value="all">全部相关边</option>'+edges.map(e=>`<option value="${e}">${state.source==="fitted"?"拓扑边":"边"} ${e}</option>`).join("");$("edge").value=state.edge===null?"all":String(state.edge);
    const paths=current.diagnostics?.paths||[];
    $("history-path").innerHTML=paths.map((p,i)=>`<option value="${i}">${pathLabels[p.label]||p.label||`路径 ${i+1}`}</option>`).join("");$("history-path").value=String(state.path);
  }
  function renderMetrics(){
    const v=view(),rs=selectedRecords(),values=rs.map(r=>r.mean).filter(finite),p95=rs.map(r=>r.p95).filter(finite),max=rs.map(r=>r.max).filter(finite);
    setMetric("selected-mean",values.length===rs.length?mean(values):null);setMetric("selected-p95",p95.length&&p95.length===rs.length?Math.max(...p95):null);setMetric("selected-max",max.length&&max.length===rs.length?Math.max(...max):null);
    const faceText=state.face===null?"所有 owner 面":`面 ${state.face}`,edgeText=state.edge===null?"全部相关边":`${state.source==="fitted"?"拓扑边":"边"} ${state.edge}`;
    $("metric-context").textContent=`${edgeText} → ${faceText}${rs.length>1?` · ${rs.length} 组边面关系`:""}`;
    $("measurement-kind").textContent=state.source==="fitted"?"解析面投影":"三角网格距离";
    const known=rs.reduce((a,r)=>a+(finite(r.valid_points)?r.valid_points:0),0),unknown=rs.reduce((a,r)=>a+(finite(r.unknown_points)?r.unknown_points:0),0);
    const hasPointCounts=rs.length&&rs.every(r=>finite(r.valid_points)&&finite(r.unknown_points));
    $("metric-coverage").innerHTML=rs.length?(hasPointCounts?`距离点：<strong>${known}</strong> 已测 / <strong>${unknown}</strong> 未知。`:`边面关系：<strong>${values.length} / ${rs.length}</strong> 组均值已测；该来源未打包逐点计数。`)+(rs.length>1?"均值按边面关系等权；P95取各关系P95的最大值。":""):"该来源没有此边面关系的距离记录；未知不表示零间隙。";
    const ep=activeEndpoints(),ds=ep.map(r=>r.distance).filter(finite),inclusive=state.source==="fitted"&&$("endpoint-mode").value!=="other";
    let endpointText=ep.length?`${inclusive?"最近连接候选（含自身首尾）":state.source==="fitted"?"不同边近邻（旧代理）":"原始端点近邻代理"}：均值 <strong>${fmt(ds.length===ep.length?mean(ds):null)}</strong>，最大 <strong>${fmt(ds.length&&ds.length===ep.length?Math.max(...ds):null)}</strong>；${ep.length-ds.length} 项未知。<br>${inclusive?"这是局部候选距离下界，不证明裁剪环闭合。":state.source==="fitted"?"近闭合整圈可能被旧代理连到远处另一环，不能据此认定真实裂缝。":"仅同owner局部近邻，含历史GT辅助闭合标记。"}`:"此来源 / 选择未取得端点记录。";
    if(state.source==="fitted"&&state.edge!==null){const reference=(v?.endpoint_records||[]).filter(r=>r.edge===state.edge&&(state.face===null||r.owner===state.face)),seams=reference.map(r=>r.own_seam_distance).filter(finite),others=reference.map(r=>r.original_neighbor_distance).filter(finite);endpointText+=`<br>自身首尾：<strong>${fmt(seams.length?Math.max(...seams):null)}</strong>；原不同边邻居：均值 <strong>${fmt(others.length===reference.length?mean(others):null)}</strong>，最大 <strong>${fmt(others.length&&others.length===reference.length?Math.max(...others):null)}</strong>；${reference.length-others.length} 项未知。`;endpointText+=reference.slice(0,4).map(r=>`<br><span class="tiny">${r.endpoint===0?"首端":"尾端"} / 面 ${r.owner} → ${r.neighbor_edge===r.edge?"自身相反端点":Number.isInteger(r.neighbor_edge)?`边 ${r.neighbor_edge} ${r.neighbor_endpoint===0?"首端":"尾端"}`:"候选未知"}：${fmt(r.distance)}</span>`).join("");}
    $("endpoint-summary").innerHTML=endpointText;
    const all=records().slice().sort((a,b)=>(finite(b.max)?b.max:-Infinity)-(finite(a.max)?a.max:-Infinity));$("owner-count").textContent=`${all.length} 组`;
    $("owner-table").innerHTML=all.map(r=>`<tr class="clickable ${state.edge===r.edge&&(state.face===null||state.face===r.owner)?"selected":""}" data-edge="${r.edge}" data-owner="${r.owner}" tabindex="0"><td class="mono">${r.edge} → ${r.owner}</td><td class="num">${fmt(r.mean)}</td><td class="num">${fmt(r.p95)}</td><td class="num">${fmt(r.max)}</td></tr>`).join("")||'<tr><td colspan="4" class="muted">此来源未取得边面距离明细</td></tr>';
  }
  function renderHistory(){
    const path=current?.diagnostics?.paths?.[state.path],last=(current?.diagnostics?.paths?.length||0)-1;
    $("path-note").textContent=`当前三维：${sourceLabels[state.source]}。${state.source==="fitted"?"拟合几何仅对应末路径诊断重放；":""}切换下列历史日志不会切换三维几何${state.path!==last?"，此路径并无对应几何快照":""}。`;
    if(!path){$("face-log-summary").textContent="未取得该路径的历史日志。";$("attempt-table").innerHTML="";$("face-log-json").textContent="null";return;}
    const f=(path.faces||[]).find(f=>f.face_index===state.face);
    if(!f){
      const attempts=path.solid_attempts||[];$("attempt-last-heading").textContent="面 / 环 / 原生检查";
      if(attempts.length){$("face-log-summary").innerHTML=`<strong>实体构造与 STEP 验收</strong><br><span class="tiny muted">裁剪面 ${path.used_face_count??"?"} / ${path.input_face_count??"?"}；${path.complete_faces===true?"所有面裁剪通过":path.complete_faces===false?"裁剪未完整":"裁剪完整性未知"}。可在上方选择面查看裁剪尝试。</span>`;$("attempt-table").innerHTML=attempts.map(a=>`<tr><td class="mono">${fmt(a.tolerance)}</td><td class="error-text">${esc(reasons[a.status]||a.status||"未知")}${finite(a.maximum_stored_tolerance)?`<span class="attempt-status muted">存储容差最大 ${fmt(a.maximum_stored_tolerance)}</span>`:""}</td><td class="tiny">${a.num_faces??"?"} / ${a.num_wires??"?"}<br>${a.native_valid_solid===true?"合法实体":a.native_valid_solid===false?"合法性未通过":"未取得原生检查"}${a.complete_boundary_counts===false?"<br>边界计数不完整":""}</td></tr>`).join("");}
      else{$("face-log-summary").innerHTML=`${state.face===null?"选择一个面，查看每次容差尝试。":`面 ${state.face} 在此路径没有逐面记录。`}<br><span class="tiny muted">路径状态：${esc(reasons[path.status]||path.status||"未知")}；原生合法实体：${path.native_valid_solid===true?"是":path.native_valid_solid===false?"否":"未知"}；面 / 环：${path.num_faces??"?"} / ${path.num_wires??"?"}</span>`;$("attempt-table").innerHTML="";}
      $("face-log-json").textContent=JSON.stringify(path,null,2);return;
    }
    $("attempt-last-heading").textContent="输入→有效→最终环";
    const success=f.accepted_attempt!==null&&f.accepted_attempt!==undefined;
    $("face-log-summary").innerHTML=`<strong>面 ${state.face}</strong> · ${success?`第 ${f.accepted_attempt+1} 次尝试通过`:'<span class="error-text">全部尝试未通过</span>'}<br><span class="tiny muted">历史输入拓扑边：${Array.isArray(f.input_edge_ids)?f.input_edge_ids.length?f.input_edge_ids.join(", "):"无输入边":"未知"}。${state.path===last?"仅与末路径拟合后视图索引对应。":"本历史路径没有几何快照，不能据此索引当前视图。"}</span>`;
    $("attempt-table").innerHTML=(f.attempts||[]).map(a=>{const rejected=a.rejection?`<span class="attempt-status muted">${esc(reasons[a.rejection]||a.rejection)}</span>`:"";const skip=(a.skipped_wires||[]).length?`<span class="attempt-status error-text">${esc((a.skipped_wires||[]).map(x=>reasons[x]||x).join("、"))}</span>`:"";return `<tr><td class="mono">${fmt(a.tolerance)}</td><td><span class="${a.status==="accepted"?"":"error-text"}">${esc(reasons[a.status]||a.status||"未知")}</span>${rejected}${skip}</td><td class="mono">${a.input_wire_count??"未知"} → ${a.effective_wire_count??"未知"} → ${a.final_wire_count??"未知"}${a.natural_wire_replacement?'<span class="attempt-status muted">自然边界替换</span>':""}</td></tr>`;}).join("");$("face-log-json").textContent=JSON.stringify(f,null,2);
  }
  function surfaceTrace(grid,face){
    const nx=grid?.length||0,ny=grid?.[0]?.length||0;if(nx<2||ny<2)return null;
    const pts=grid.flat(),x=[],y=[],z=[],i=[],j=[],k=[];
    if(!pts.every(isPoint))return null;for(const p of pts){x.push(p[0]);y.push(p[1]);z.push(p[2]);}
    for(let a=0;a<nx-1;a++)for(let b=0;b<ny-1;b++){const q=a*ny+b;i.push(q,q);j.push(q+ny,q+ny+1);k.push(q+ny+1,q+1);}
    const selected=face===state.face,failed=failedFaces().includes(face),opacity=Number($("opacity").value);
    return{type:"mesh3d",x,y,z,i,j,k,color:selected?colors.selected:failed?colors.fail:colors.face,opacity:selected?Math.min(opacity+.15,.95):opacity,flatshading:false,lighting:{ambient:.65,diffuse:.6,specular:.15,roughness:.8},name:`面 ${face}`,hovertemplate:`面 ${face}${failed?" · 末路径裁剪失败":""}<extra></extra>`,meta:{kind:"face",face},showlegend:false};
  }
  function lineTrace(points,color,width,name,meta,mode="lines"){
    return{type:"scatter3d",mode,x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),line:{color,width},marker:{color,size:3},name,meta,hovertemplate:name+"<extra></extra>",showlegend:false};
  }
  function maximumGap(){
    let result=null;for(const r of selectedRecords()){const curve=view()?.edges?.[r.edge];if(!curve||!r.closest||!r.distances)continue;for(let i=0;i<r.distances.length;i++){const d=r.distances[i];if(finite(d)&&isPoint(curve[i])&&isPoint(r.closest[i])&&(!result||d>result.distance))result={a:curve[i],b:r.closest[i],distance:d,edge:r.edge,owner:r.owner,index:i};}}
    return result;
  }
  function activeEndpoints(){
    const v=view(),rs=state.source==="fitted"&&$("endpoint-mode").value==="other"?v?.endpoint_other_edge_records:v?.endpoint_records;
    return(rs||[]).filter(r=>(state.face===null||r.owner===state.face)&&(state.edge===null||r.edge===state.edge));
  }
  function endpointConnectors(){
    const v=view();return activeEndpoints().filter(r=>finite(r.distance)&&Number.isInteger(r.neighbor_edge)).map(r=>{const a=v.edges?.[r.edge]?.at(r.endpoint),b=v.edges?.[r.neighbor_edge]?.at(r.neighbor_endpoint);return isPoint(a)&&isPoint(b)?{...r,a,b}:null;}).filter(Boolean);
  }
  function allPoints(){const v=view();return(v?.faces||[]).flatMap(gridPoints).concat((v?.edges||[]).flat().filter(isPoint));}
  function focusPoints(){
    const v=view();if(state.focus==="face"&&state.face!==null)return gridPoints(v?.faces?.[state.face]);
    if(state.focus==="edge"&&state.edge!==null)return(v?.edges?.[state.edge]||[]).filter(isPoint);
    if(state.focus==="gap"){const gap=maximumGap();if(gap)return[gap.a,gap.b];const ep=endpointConnectors().sort((a,b)=>b.distance-a.distance)[0];if(ep)return[ep.a,ep.b];}
    return allPoints();
  }
  function ranges(){
    const pts=focusPoints();if(!pts.length)return[[-1,1],[-1,1],[-1,1]];
    const low=[0,1,2].map(a=>Math.min(...pts.map(p=>p[a]))),high=[0,1,2].map(a=>Math.max(...pts.map(p=>p[a])));let radius=Math.max(...high.map((v,a)=>v-low[a]))*.62;
    if(state.focus==="gap")radius=Math.max(radius,.00015);else radius=Math.max(radius,.001);
    return[0,1,2].map(a=>[(low[a]+high[a])/2-radius,(low[a]+high[a])/2+radius]);
  }
  async function renderGeometry(reset=false){
    if(!current)return;const token=++renderToken,v=view();if(!v)return;
    const traces=[],isolate=$("isolate").checked;
    for(let face=0;face<(v.faces||[]).length;face++){if(isolate&&state.face!==null&&face!==state.face)continue;const tr=surfaceTrace(v.faces[face],face);if(tr)traces.push(tr);}
    const owned=new Set(ownerEdges(v,state.face));for(let e=0;e<(v.edges||[]).length;e++){if(isolate&&!owned.has(e))continue;const points=(v.edges[e]||[]).filter(isPoint);if(!points.length)continue;const selected=e===state.edge;const tr=lineTrace(points,selected?colors.selectedEdge:colors.edge,selected?7:owned.has(e)?3:1.5,`${state.source==="fitted"?"拓扑边":"边"} ${e}`,{kind:"edge",edge:e});tr.opacity=selected?1:owned.has(e)?.9:.35;traces.push(tr);}
    const selected=selectedRecords();
    if($("show-projections").checked){for(const r of selected){const curve=v.edges?.[r.edge];if(!curve||!r.closest||!r.distances)continue;const points=[];const stride=state.edge===null?8:2;for(let i=0;i<r.closest.length;i+=stride)if(isPoint(curve[i])&&isPoint(r.closest[i]))points.push(curve[i],r.closest[i],null);if(points.length)traces.push(lineTrace(points,colors.projection,2,`边 ${r.edge} → 面 ${r.owner} 的投影`,{kind:"projection",edge:r.edge,owner:r.owner}));}}
    const gap=maximumGap();if(gap&&$("show-projections").checked){const tr=lineTrace([gap.a,gap.b],colors.projection,7,`最大边面距离 ${fmt(gap.distance)}`,{kind:"projection",edge:gap.edge,owner:gap.owner},"lines+markers");tr.marker.size=5;traces.push(tr);const midpoint=gap.a.map((a,i)=>(a+gap.b[i])/2);traces.push({type:"scatter3d",mode:"text",x:[midpoint[0]],y:[midpoint[1]],z:[midpoint[2]],text:[fmt(gap.distance)],textposition:"top center",textfont:{...font,size:12,color:"#9c6624"},hoverinfo:"skip",showlegend:false});}
    const ep=endpointConnectors();if($("show-endpoints").checked&&ep.length){const points=[];for(const r of ep)points.push(r.a,r.b,null);traces.push(lineTrace(points,colors.endpoint,4,state.source==="fitted"&&$("endpoint-mode").value!=="other"?"最近连接候选 · 含自身首尾 · 仅下界":"同 owner 的不同边端点近邻代理",{kind:"endpoint"},"lines+markers"));}
    if($("show-gt").checked&&state.source!=="gt"&&current.views?.gt){const gt=current.views.gt,indices=isolate?ownerEdges(gt,state.face):(gt.edges||[]).map((_,i)=>i),points=[];for(const e of indices)points.push(...(gt.edges[e]||[]).filter(isPoint),null);if(points.length){const tr=lineTrace(points,"#31847e",3,"GT 边界 · 同坐标诊断参照",{kind:"gt"});tr.line.dash="dash";tr.opacity=.8;traces.push(tr);}}
    const xyz=ranges(),axis=a=>({range:xyz[a],title:"",visible:false,showspikes:false});
    const layout={margin:{l:0,r:0,t:0,b:0},paper_bgcolor:"#fff",font,showlegend:false,uirevision:`${state.id}-${state.revision}`,scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:"cube",dragmode:"orbit",camera:state.camera||{eye:{x:1.4,y:-1.6,z:1.1},up:{x:0,y:0,z:1}},bgcolor:"#fafbfc"}};
    if(reset){Plotly.purge($("geometry"));geometryBound=false;}
    await Plotly.react($("geometry"),traces,layout,plotConfig);if(token!==renderToken)return;
    if(!geometryBound){$("geometry").on("plotly_click",ev=>{const m=ev.points?.[0]?.data?.meta;if(m?.kind==="face")setSelection(m.face,null,true);else if(m?.kind==="edge"){state.edge=m.edge;const conn=(view().connectivity||[]).find(c=>c[0]===m.edge);if(state.face!==null&&conn&&!conn.slice(1).includes(state.face))state.face=conn[1];refresh();}});$("geometry").on("plotly_relayout",ev=>{if(ev["scene.camera"])state.camera=ev["scene.camera"];});geometryBound=true;}
    $("focus-face").disabled=state.face===null;$("focus-edge").disabled=state.edge===null;$("focus-gap").disabled=!gap&&!ep.length;
    $("view-caption").textContent=`${sourceLabels[state.source]} · ${(v.faces||[]).length} 面 / ${(v.edges||[]).length} 边。${state.source==="fitted"?"曲面是采样显示网格；距离投影到拟合后的解析面。":"面为未裁剪的 UV 网格；当前边编号不对应历史 post 输入边编号。"}${!selected.some(r=>r.closest)?" 此来源没有投影连线数据。":""}${$("show-gt").checked?" GT按原坐标叠加，无配准；右侧距离仍是边→owner，不是GT误差。":""}`;
    await renderProfile();
  }
  async function renderProfile(){
    const rs=selectedRecords().filter(r=>Array.isArray(r.distances)),traces=rs.map(r=>({type:"scatter",mode:"lines+markers",x:r.distances.map((_,i)=>i),y:r.distances.map(x=>finite(x)?x:null),connectgaps:false,name:`边${r.edge}→面${r.owner}`,line:{width:1.6},marker:{size:3},hovertemplate:"采样点 %{x}<br>距离 %{y:.6g}<extra>%{fullData.name}</extra>"}));
    const title=rs.length?`逐点到面距离 · ${state.source==="fitted"?"64点解析面投影":"原始网格测量"}`:"此来源 / 选择没有逐点距离记录";
    await Plotly.react($("gap-profile"),traces,{height:185,margin:{l:58,r:20,t:37,b:35},font,paper_bgcolor:"#fff",plot_bgcolor:"#fff",title:{text:title,font:{...font,size:11},x:.04},showlegend:rs.length>1&&rs.length<7,legend:{orientation:"h",y:1.2},xaxis:{title:{text:"曲线采样点序号",font:{size:10}},gridcolor:"#f0f2f5",zeroline:false},yaxis:{rangemode:"tozero",gridcolor:"#eef1f5",zeroline:false,tickformat:".2~g"}}, {...plotConfig,displayModeBar:false,scrollZoom:false});
  }
  function distributionDefinition(key){
    const d=M.definitions?.[key];if(typeof d==="string")return d;
    if(key.startsWith("replay_endpoint_other"))return"仅316件失败进行了诊断重放；成功对照未重放。原同owner不同边端点近邻代理，对近闭合整圈可能错连到远处另一环，较大值不能直接视为真实裂缝。每CAD一个值，未知单列。";
    if(key.startsWith("replay_endpoint"))return"仅316件失败进行了诊断重放；成功对照未重放。最近连接候选（含自身首尾）取原同owner不同边近邻与曲线自身相反端点的较小值，是局部距离下界，不证明全局wire配对、定向或闭合。每CAD一个值，未知单列。";
    if(key.startsWith("replay"))return"仅316件失败进行了诊断重放；成功对照未重放。按CAD等权；显示统计只使用有限值，未知单列。拟合距离是64个采样点到有界解析owner曲面的投影，不是连续曲线的最大误差上界。";
    if(key==="raw_p95")return"每件CAD取所有边-owner关系P95的最大值，不是将全部点合并后的P95。原始预测每条曲线16点到16×16三角网格；成功与失败共3495件。";
    if(key.startsWith("endpoint"))return"每件CAD的同owner端点最近邻间隙，是局部连接代理；原始历史测量含GT辅助闭合标记，不能当成实际wire闭合证明。每CAD一个值，均值按对象等权。";
    return"原始预测距离为每条曲线16个点到owner面三角网格的距离。均值先按边-owner关系等权，再按CAD等权；最大值取对象所有边-owner的最大距离。未知单列，不以零补齐。";
  }
  async function renderDistribution(){
    const key=$("distribution-metric").value,dist=M.distributions?.[key]||{},isReplay=key.startsWith("replay"),log=$("distribution-log").checked;
    const series=[{key:"failed",label:"失败",color:colors.failed,total:M.cases.length},{key:"valid",label:"合法",color:colors.valid,total:3179}].filter(s=>!isReplay||s.key==="failed");
    const traces=[],stats=[];for(const s of series){const all=(dist[s.key]||[]).filter(finite),values=all.filter(v=>!log||v>0).sort((a,b)=>a-b),unknown=dist[`unknown_${s.key}`]??Math.max(0,s.total-all.length);if(state.distribution==="cdf")traces.push({type:"scatter",mode:"lines",x:values,y:values.map((_,i)=>(all.length-values.length+i+1)/all.length*100),name:`${s.label} · n=${all.length}`,line:{color:s.color,width:2.4},hovertemplate:"距离≤%{x:.6g}<br>累计 %{y:.2f}%<extra>%{fullData.name}</extra>"});else traces.push({type:"histogram",x:values,name:`${s.label} · n=${all.length}`,histnorm:"percent",nbinsx:42,opacity:.65,marker:{color:s.color},hovertemplate:"距离 %{x}<br>比例 %{y:.2f}%<extra>%{fullData.name}</extra>"});stats.push(`<div class="stat-block"><strong style="color:${s.color}">${s.label}</strong> · 已测 ${all.length} / 未知 ${unknown}<br>均值 <strong class="mono">${fmt(mean(all))}</strong>　中位数 <strong class="mono">${fmt(percentile(all,.5))}</strong>　P90 <strong class="mono">${fmt(percentile(all,.9))}</strong>${log?`<br>对数轴另隐藏 ${all.length-values.length} 个零值；CDF分母仍含零值，直方图按正值子集归一；汇总含零值。`:""}</div>`);}
    const selected=current?.metrics?.[key]??selectedMeta()?.[key],shapes=[],annotations=[];if(finite(selected)&&(!log||selected>0)){shapes.push({type:"line",xref:"x",yref:"paper",x0:selected,x1:selected,y0:0,y1:1,line:{color:"#263e5a",width:1.5,dash:"dot"}});annotations.push({xref:"x",yref:"paper",x:selected,y:1.04,text:`当前 ${fmt(selected)}`,showarrow:false,font:{...font,size:10,color:"#263e5a"},xanchor:"left"});}
    await Plotly.react($("distribution"),traces,{height:280,margin:{l:58,r:25,t:40,b:45},font,paper_bgcolor:"#fff",plot_bgcolor:"#fff",barmode:"overlay",showlegend:true,legend:{orientation:"h",x:0,y:1.21},shapes,annotations,xaxis:{type:log?"log":"linear",title:{text:"归一化长度",font:{size:11}},tickformat:".2~g",gridcolor:"#eef1f5",zeroline:false},yaxis:{title:{text:state.distribution==="cdf"?"累计比例 (%)":"对象比例 (%)",font:{size:11}},range:state.distribution==="cdf"?[0,102]:undefined,gridcolor:"#eef1f5",zeroline:false}}, {...plotConfig,displayModeBar:false,scrollZoom:false});
    $("distribution-stats").innerHTML=stats.join("");$("distribution-note").textContent=distributionDefinition(key)+" 分布和查看位置不会改变任何OCC容差或正式评估结论。";
    $("distribution-subtitle").textContent=isReplay?"拟合后：只测量失败全集，未对成功组重放。":"原始预测：合法3179件与失败316件；不随左侧浏览筛选改变。";
  }
  async function refresh(reset=false){
    renderHeading();renderSelectors();renderMetrics();renderHistory();await renderGeometry(reset);await renderDistribution();
  }
  async function loadCase(id){
    if(!M.cases.some(c=>c.id===id))throw new Error(`未知案例 ${id}`);
    const token=++requestToken;state.loading=true;state.error=null;$("loading").hidden=false;$("loading").textContent="正在读取几何…";
    try{const value=await decodeCase(id);if(token!==requestToken)return;current=value;state.id=id;state.source=value.views?.fitted?"fitted":value.views?.raw?"raw":Object.keys(value.views||{})[0];state.face=failedFaces()[0]??null;state.path=Math.max(0,(value.diagnostics?.paths?.length||0)-1);state.focus="all";state.camera=null;state.revision++;chooseWorstEdge();renderList();await refresh(true);if(token!==requestToken)return;$("loading").hidden=true;try{history.replaceState(null,"",`#${id}`);}catch(_){}state.loading=false;}
    catch(err){if(token!==requestToken)return;state.loading=false;state.error=String(err);$("loading").hidden=false;$("loading").textContent=`几何未载入：${err.message}`;$("replay-status").textContent=String(err);$("replay-status").className="status-banner error";console.error(err);}
  }
  async function setSelection(face,edge,choose=false){state.face=face;state.edge=edge;if(choose)chooseWorstEdge();state.focus="all";state.revision++;await refresh();}
  function nextCase(delta){const i=filtered.findIndex(c=>c.id===state.id),target=filtered[i+delta];if(target)loadCase(target.id);}
  function applyFilter(){renderList();if(filtered.length&&!filtered.some(c=>c.id===state.id))loadCase(filtered[0].id);}
  for(const id of["search","stage-filter","cohort-filter","sort"])$(id).addEventListener(id==="search"?"input":"change",applyFilter);
  $("case-list").addEventListener("click",e=>{const b=e.target.closest("[data-id]");if(b)loadCase(b.dataset.id);});
  $("previous").addEventListener("click",()=>nextCase(-1));$("next").addEventListener("click",()=>nextCase(1));
  document.addEventListener("keydown",e=>{if(/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)||e.ctrlKey||e.altKey||e.metaKey)return;if(e.key==="ArrowRight"){e.preventDefault();nextCase(1);}if(e.key==="ArrowLeft"){e.preventDefault();nextCase(-1);}});
  $("source").addEventListener("change",async()=>{state.source=$("source").value;chooseWorstEdge();state.focus="all";state.revision++;await refresh();});
  $("face").addEventListener("change",()=>setSelection($("face").value==="all"?null:Number($("face").value),null,true));
  $("edge").addEventListener("change",async()=>{state.edge=$("edge").value==="all"?null:Number($("edge").value);await refresh();});
  $("failed-faces").addEventListener("click",e=>{const b=e.target.closest("[data-face]");if(b)setSelection(Number(b.dataset.face),null,true);});
  function ownerPick(e){const r=e.target.closest("tr[data-edge]");if(r)setSelection(Number(r.dataset.owner),Number(r.dataset.edge));}
  $("owner-table").addEventListener("click",ownerPick);$("owner-table").addEventListener("keydown",e=>{if(e.key==="Enter")ownerPick(e);});
  $("history-path").addEventListener("change",()=>{state.path=Number($("history-path").value);renderHistory();});
  for(const id of["isolate","show-projections","show-endpoints","show-gt","opacity"])$(id).addEventListener("change",()=>renderGeometry());
  $("endpoint-mode").addEventListener("change",()=>{renderMetrics();renderGeometry();});
  for(const focus of["all","face","edge","gap"])$("focus-"+focus).addEventListener("click",()=>{state.focus=focus;state.revision++;renderGeometry();});
  $("camera-top").addEventListener("click",()=>{state.camera={eye:{x:0,y:0,z:2},up:{x:0,y:1,z:0}};state.revision++;renderGeometry();});$("camera-front").addEventListener("click",()=>{state.camera={eye:{x:0,y:-2,z:0},up:{x:0,y:0,z:1}};state.revision++;renderGeometry();});
  $("distribution-metric").addEventListener("change",renderDistribution);$("distribution-log").addEventListener("change",renderDistribution);
  for(const mode of["hist","cdf"])$("distribution-"+mode).addEventListener("click",()=>{state.distribution=mode;for(const m of["hist","cdf"])$("distribution-"+m).setAttribute("aria-pressed",String(mode===m));renderDistribution();});
  window.viewer={state,loadCase,setSelection,refresh,renderDistribution,get current(){return current;},get filtered(){return filtered;},get cacheSize(){return cache.size;},maximumGap};
  if(!M?.cases?.length){$("loading").textContent="没有内嵌案例清单";return;}
  $("total-failures").textContent=M.cases.length;$("trim-count").textContent=M.cases.filter(c=>c.stage==="trimming").length;$("failed-face-count").textContent=M.cases.reduce((n,c)=>n+(c.failed_faces||[]).length,0);
  renderList();const hash=location.hash.slice(1);window.viewer.ready=loadCase(M.cases.some(c=>c.id===hash)?hash:M.cases[0].id);
})();
