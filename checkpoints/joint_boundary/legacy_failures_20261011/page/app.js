/* Offline case browser. Geometry is loaded as local scripts to support file://. */
(() => {
  "use strict";
  const data = window.LEGACY_FAILURE_DATA;
  const byId = id => document.getElementById(id);
  if (!data || !Array.isArray(data.cases) || !data.cases.length) {
    byId("case-title").textContent = "案例索引尚未生成";
    byId("viewer-status").textContent = "请按 README 运行页面构建命令。";
    return;
  }
  const names = {0: "Compound", 1: "CompSolid", 2: "Solid", 3: "Shell", 4: "Face"};
  const notesKey = "hola-legacy-failures-20261011-notes";
  let notes = {};
  try { notes = JSON.parse(localStorage.getItem(notesKey) || "{}"); } catch (_) {}
  let current = null, currentGeometry = null, selectedPanelFace = null;
  let filtered = [...data.cases], loadingScript = null;
  const cameraState = {azim: -58, elev: 24, zoom: 1};
  const panelViews = {};
  const escapeText = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const finite = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const number = (value, digits=3) => finite(value) ? Number(value).toFixed(digits) : "未取得";
  const scientific = value => finite(value) ? Number(value).toExponential(3) : "未取得";
  const yesNo = value => value === true ? "通过" : value === false ? "未通过" : "未取得";
  const kind = item => (names[item.post_legacy.shape_type] || "Unknown").toLowerCase();
  const isExact = item => Number(item.raw_legacy.topology_fp) === 0 && Number(item.raw_legacy.topology_fn) === 0;
  const diagnostic = (item, key) => item.diagnostics?.panels?.[key] || item.diagnostics?.[key] || {};
  const postCounts = item => item.geometry_counts?.post || diagnostic(item, "post").counts || {};
  const population = data.population || {intended: 2417, failures: 369, successes: 2048, topology_exact_failures: 268};
  byId("population").innerHTML = `<span><strong>${population.intended}</strong>测试 CAD</span><span><strong>${population.failures}</strong>未通过有效 solid 检查</span><span><strong>${data.cases.length}</strong>固定案例</span><span><strong>${population.topology_exact_failures}</strong>失败件的合并无向面 pair 仍精确</span>`;

  function renderList() {
    const query = byId("search").value.trim().toLowerCase();
    const filter = byId("filter").value;
    filtered = data.cases.filter(item => item.id.toLowerCase().includes(query) && (
      filter === "all" || filter === kind(item) ||
      (filter === "topology_exact" && isExact(item)) ||
      (filter === "mesh_missing" && Number(postCounts(item).missing_mesh_faces || 0) > 0)
    ));
    byId("list-count").textContent = `${filtered.length} / ${data.cases.length} 件`;
    byId("case-list").replaceChildren();
    for (const item of filtered) {
      const button = document.createElement("button");
      button.className = "case-item";
      button.dataset.id = item.id;
      button.setAttribute("aria-current", String(current?.id === item.id));
      button.innerHTML = `<strong>${escapeText(item.id)}</strong><small>${item.raw_legacy.gt_face_count} 面 · ${escapeText(names[item.post_legacy.shape_type] || "?")}</small>`;
      button.addEventListener("click", () => selectCase(item.id));
      byId("case-list").appendChild(button);
    }
    if (!filtered.length) byId("case-list").innerHTML = '<p class="empty-state">没有符合条件的案例。</p>';
    refreshNavigation();
  }
  function refreshNavigation() {
    const index = filtered.findIndex(item => item.id === current?.id);
    byId("previous").disabled = index <= 0;
    byId("next").disabled = index < 0 || index >= filtered.length - 1;
  }
  function metricRow(label, value) {
    return `<div class="metric-row"><dt>${escapeText(label)}</dt><dd>${escapeText(value)}</dd></div>`;
  }
  function observations(item, geometry) {
    const raw = item.raw_legacy, post = item.post_legacy;
    const values = [
      `保存结果为 ${names[post.shape_type] || "未知形状"}；BRepCheck ${yesNo(post.native_brepcheck_valid)}，有效 solid ${yesNo(post.native_valid_solid)}。`,
      isExact(item) ? "Raw 合并无向面 pair 与 GT 完全一致；这不保证完整 B-Rep 拓扑、裁剪环或实体有效。" : `Raw 面邻接存在 ${raw.topology_fp ?? "?"} 个多余 pair、${raw.topology_fn ?? "?"} 个遗漏 pair。`,
      `解码有 ${raw.pred_face_count} 个面；后处理输出 ${post.num_faces} 个面。数量相同也不保证逐面几何或裁剪正确。`
    ];
    const counts = geometry?.panels?.post?.counts || postCounts(item);
    if (finite(counts.missing_mesh_faces) && Number(counts.missing_mesh_faces) > 0) values.push(`后处理有 ${counts.missing_mesh_faces} 个面无法取得显示网格；视图保留其余面和可读边，不代表全部曲面已显示。`);
    const free = counts.free_edges ?? diagnostic(item, "post").incidence?.free_edges;
    if (finite(free)) values.push(`后处理包含 ${free} 条拓扑上未与另一面共享的非 seam、非退化边（可在视图中标红）。`);
    if (finite(counts.shells)) values.push(`输出包含 ${counts.shells} 个 shell、${counts.solids} 个 solid 子形状；子形状计数不替代整体有效性检查。`);
    const check = diagnostic(item,"post").brepcheck_subshapes;
    const statuses = [...new Set((check?.with_errors || []).flatMap(entry => [...(entry.statuses || []), ...(entry.contexts || []).flatMap(context => context.statuses || [])]))].filter(name => name !== "BRepCheck_NoError");
    if (statuses.length) values.push(`原生子形状检查报告：${statuses.join("、")}。这是内核状态，不单独证明神经网络的错误来源。`);
    if (check?.inspection_errors?.length) values.push(`有 ${check.inspection_errors.length} 项子形状诊断未能完成，不能把未取得状态当作检查通过。`);
    if (item.extent?.status === "ok" && item.extent.min_over_max_span < .01) values.push(`GT 整体采样包围跨度很扁，最小／最大跨度比 ${number(item.extent.min_over_max_span*100,2)}%；这描述整体尺度，不是壁厚测量。`);
    if (item.fit_legacy.native_valid_solid) values.push("同一原始预测在 fit-only 下通过有效 solid 检查，optimized 下失败：这里需要单独检查优化／后处理敏感性。");
    if (post.compat_valid_solid && !post.native_valid_solid) values.push("宽松兼容容差下通过，但原生读入检查失败：这是容差敏感案例，不能按 success.txt 视为通过。");
    byId("observations").innerHTML = values.map(value => `<li>${escapeText(value)}</li>`).join("");
    const gaps = (diagnostic(item,"post").wire_endpoint_gaps?.wires || []).map(wire => wire.max).filter(finite);
    byId("metrics").innerHTML = [
      metricRow("Raw 面 XYZ MAE", scientific(raw.face_mae)),
      metricRow("Raw 面 CD² / 边 CD²", `${scientific(raw.face_cd_sq)} / ${scientific(raw.edge_cd_sq)}`),
      metricRow("面邻接 precision / recall", `${number(raw.topology_precision)} / ${number(raw.topology_recall)}`),
      metricRow("互为反向的半边 CD²：最大", scientific(raw.legacy_reciprocal_cd_sq_max)),
      metricRow("STEP 非流形 / 退化边", `${diagnostic(item,"post").incidence?.nonmanifold_edges ?? "未取得"} / ${diagnostic(item,"post").incidence?.degenerate_edges ?? "未取得"}`),
      metricRow("STEP wire 端点间隙：最大", scientific(gaps.length ? Math.max(...gaps) : null)),
      metricRow("GT 整体采样跨度 X / Y / Z", item.extent?.gt_sampled_aabb_span_xyz?.map(value => number(value,4)).join(" / ") || "未取得"),
      metricRow("后处理面 CD² / 边 CD²", `${scientific(post.face_cd)} / ${scientific(post.edge_cd)}`),
      metricRow("后处理几何指标可计算", post.geometry_status === "ok" ? "是" : "否，保留失败记录"),
      metricRow("同件 Joint 输出：有效 solid", yesNo(item.post_joint.native_valid_solid))
    ].join("");
  }
  function showTrace(item) {
    const trace = item.trace;
    if (!trace) {
      byId("trace-summary").textContent = "未取得额外阶段诊断；原始 STEP 与原评估结论仍保留。";
      byId("trace-json").textContent = "null";
      return;
    }
    const statements = item.trace_observations || trace.observations || trace.summary_text || [];
    byId("trace-summary").replaceChildren();
    for (const statement of (Array.isArray(statements) ? statements : [statements])) {
      const p = document.createElement("p");
      p.textContent = typeof statement === "string" ? statement : JSON.stringify(statement);
      byId("trace-summary").appendChild(p);
    }
    if (!byId("trace-summary").children.length) byId("trace-summary").textContent = "已记录选边、几何优化、拟合、裁剪与缝合过程。展开查看逐阶段证据；阶段失败不等于已经定位模型根因。";
    if (trace.comparison?.same_recorded_outcome === false) {
      const warning = document.createElement("p");
      warning.style.color = "var(--red)";
      warning.textContent = "重放结果与原评估不同：页面几何和案例分类始终使用原始评估，不能用这次重放替代。";
      byId("trace-summary").prepend(warning);
    }
    byId("trace-json").textContent = JSON.stringify(trace, null, 2);
  }
  function selectCase(id) {
    const item = data.cases.find(entry => entry.id === id);
    if (!item) return;
    current = item;
    currentGeometry = null;
    selectedPanelFace = null;
    byId("case-position").textContent = `CASE ${String(data.cases.indexOf(item)+1).padStart(2,"0")} / ${data.cases.length}`;
    byId("case-title").textContent = item.id;
    byId("case-subtitle").textContent = `${names[item.post_legacy.shape_type]} · ${item.raw_legacy.gt_face_count} 个 GT 面 · ${isExact(item) ? "合并无向面 pair 精确" : "面 pair 存在错误"}`;
    byId("notes").value = notes[item.id] || "";
    byId("notes-status").textContent = "";
    byId("still-image").src = `../visualizations/${item.id}.png`;
    byId("still-link").href = byId("still-image").getAttribute("src");
    byId("diagnostic-link").href = `../diagnostics/${item.id}.json`;
    byId("source-link").href = `../../comparison_20261011/metadata/post/optimized/samples/legacy/${item.id}.json`;
    observations(item);
    showTrace(item);
    renderList();
    try { history.replaceState(null, "", `#${id}`); } catch (_) {}
    byId("viewer-status").textContent = `正在载入 ${id} 的几何…`;
    for (const key of ["gt", "raw", "post"]) {
      clearPanel(key);
      byId(`detail-${key}`).textContent = "载入中…";
    }
    if (loadingScript) loadingScript.remove();
    loadingScript = document.createElement("script");
    loadingScript.src = `geometry/${id}.js`;
    loadingScript.onerror = () => {
      if (current.id !== id) return;
      byId("viewer-status").textContent = "几何未载入，可展开下方固定截图。";
    };
    document.body.appendChild(loadingScript);
  }

  function createPanel(key) {
    if (!window.THREE) return;
    const host = byId(`canvas-${key}`);
    try {
      const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true, preserveDrawingBuffer: true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.setClearColor(0xffffff, 0);
      renderer.domElement.setAttribute("aria-label", `${key} 三维视图，可拖动旋转；固定截图提供替代查看方式`);
      host.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      scene.add(new THREE.HemisphereLight(0xffffff, 0x8292a2, 2));
      const light = new THREE.DirectionalLight(0xffffff, 2.2);
      light.position.set(3, -4, 6); scene.add(light);
      const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .001, 1000);
      camera.up.set(0, 0, 1);
      const group = new THREE.Group(); scene.add(group);
      const view = {renderer, scene, camera, group, host, surfaces:[], edges:[]};
      panelViews[key] = view;
      const observer = new ResizeObserver(() => updateCameras()); observer.observe(host);
      let pointer = null;
      host.addEventListener("pointerdown", event => {
        pointer = {x: event.clientX, y: event.clientY, moved: 0};
        host.setPointerCapture(event.pointerId);
      });
      host.addEventListener("pointermove", event => {
        if (!pointer) return;
        const dx = event.clientX-pointer.x, dy = event.clientY-pointer.y;
        pointer.moved += Math.abs(dx)+Math.abs(dy);
        cameraState.azim -= dx*.45;
        cameraState.elev = Math.max(-89.9, Math.min(89.9, cameraState.elev+dy*.4));
        pointer.x = event.clientX; pointer.y = event.clientY;
        document.querySelectorAll("[data-camera]").forEach(button => button.setAttribute("aria-pressed", "false"));
        updateCameras();
      });
      host.addEventListener("pointerup", event => {
        if (pointer?.moved < 5 && currentGeometry) {
          const rect = renderer.domElement.getBoundingClientRect();
          const mouse = new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1, -(event.clientY-rect.top)/rect.height*2+1);
          const ray = new THREE.Raycaster(); ray.setFromCamera(mouse, camera);
          const hits = ray.intersectObjects(view.surfaces.filter(mesh => mesh.visible));
          selectedPanelFace = hits.length ? {panel:key,id:hits[0].object.userData.faceId} : null;
          applyDisplay();
          byId("viewer-status").textContent = selectedPanelFace ? `${key.toUpperCase()} 局部面编号 ${selectedPanelFace.id}（编号不跨面板对应）` : "三栏相机已同步。";
        }
        pointer = null;
      });
      host.addEventListener("pointercancel", () => {pointer = null;});
      host.addEventListener("wheel", event => {
        event.preventDefault();
        cameraState.zoom = Math.max(.15, Math.min(8, cameraState.zoom*Math.exp(-event.deltaY*.001)));
        updateCameras();
      }, {passive:false});
    } catch (error) {
      host.innerHTML = '<div class="fallback">3D 显示不可用<br>请查看下方固定多视图</div>';
    }
  }
  function clearPanel(key) {
    const view = panelViews[key];
    if (!view) return;
    for (const child of [...view.group.children]) {
      child.geometry?.dispose();
      if (Array.isArray(child.material)) child.material.forEach(material => material.dispose());
      else child.material?.dispose();
      view.group.remove(child);
    }
    view.surfaces = []; view.edges = [];
    view.renderer.render(view.scene, view.camera);
  }
  function fillPanel(key, panel) {
    const view = panelViews[key];
    if (!view) return;
    clearPanel(key);
    for (const face of panel.faces || []) {
      if (!face.positions?.length || !face.indices?.length) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(face.positions, 3));
      geometry.setIndex(face.indices); geometry.computeVertexNormals();
      const material = new THREE.MeshStandardMaterial({color:0x9eb8ce, roughness:.8, metalness:.03, side:THREE.DoubleSide, polygonOffset:true,polygonOffsetFactor:1,polygonOffsetUnits:1});
      const mesh = new THREE.Mesh(geometry, material); mesh.userData.faceId = face.id;
      view.group.add(mesh); view.surfaces.push(mesh);
    }
    for (const edge of panel.edges || []) {
      if (!edge.points?.length) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(edge.points, 3));
      const material = new THREE.LineBasicMaterial({color:0x254764, transparent:true, opacity:.9});
      const line = new THREE.Line(geometry, material);
      line.userData = edge; view.group.add(line); view.edges.push(line);
    }
  }
  function applyDisplay() {
    const wire = byId("wire-only").checked, xray = byId("xray").checked;
    const highlight = byId("highlight").checked, colors = byId("face-colors").checked;
    const palette = [0x8eafc7,0xa6c4b2,0xd4b28e,0xb5a4cc,0xc7be8f,0x9bb8ba,0xcaa2a4];
    for (const [key, view] of Object.entries(panelViews)) {
      for (const mesh of view.surfaces) {
        mesh.visible = !wire;
        const selected = selectedPanelFace?.panel === key && selectedPanelFace.id === mesh.userData.faceId;
        mesh.material.color.setHex(selected ? 0xe8b656 : colors ? palette[mesh.userData.faceId % palette.length] : key === "gt" ? 0xa7bfba : key === "raw" ? 0xa6b9cc : 0xb4b8c5);
        mesh.material.transparent = xray;
        mesh.material.opacity = xray ? .27 : 1;
        mesh.material.depthWrite = !xray;
      }
      for (const line of view.edges) {
        line.material.color.setHex(highlight && line.userData.free ? 0xd0362f : highlight && line.userData.nonmanifold ? 0xa535a0 : 0x344d60);
        line.material.depthTest = !xray;
        line.renderOrder = xray ? 2 : 0;
      }
      view.renderer.render(view.scene, view.camera);
    }
  }
  function updateCameras() {
    const bounds = currentGeometry?.bounds || {center:[0,0,0],radius:1};
    const radius = Math.max(Number(bounds.radius) || 1, .001);
    const center = bounds.center || [0,0,0];
    const azim = cameraState.azim*Math.PI/180, elev=cameraState.elev*Math.PI/180;
    for (const view of Object.values(panelViews)) {
      const width = view.host.clientWidth, height = view.host.clientHeight;
      if (!width || !height) continue;
      view.renderer.setSize(width, height, false);
      const extent = radius*1.15/cameraState.zoom, aspect = width/height;
      view.camera.left = -extent*aspect; view.camera.right=extent*aspect;
      view.camera.top=extent; view.camera.bottom=-extent;
      view.camera.near=radius*.001; view.camera.far=radius*100;
      const distance=radius*6;
      view.camera.position.set(center[0]+distance*Math.cos(elev)*Math.cos(azim),center[1]+distance*Math.cos(elev)*Math.sin(azim),center[2]+distance*Math.sin(elev));
      view.camera.lookAt(...center); view.camera.updateProjectionMatrix();
      view.renderer.render(view.scene, view.camera);
    }
  }
  window.LEGACY_GEOMETRY_CALLBACK = payload => {
    if (payload.id !== current?.id) return;
    currentGeometry = payload;
    for (const key of ["gt", "raw", "post"]) {
      const panel = payload.panels[key];
      if (!panel) { byId(`detail-${key}`).textContent="没有可读取几何"; continue; }
      fillPanel(key, panel);
      const c = panel.counts || {};
      const missing = Number(c.missing_mesh_faces || 0);
      const free = diagnostic(current,key).incidence?.free_edges;
      byId(`detail-${key}`).innerHTML = `${c.faces ?? panel.faces.length} 面 · ${c.edges ?? panel.edges.length} 边${finite(free) ? ` · ${free} 开放边` : ""}${missing ? `<br><span class="warning">${missing} 个面未能显示网格</span>` : ""}${panel.error ? `<br><span class="warning">${escapeText(panel.error)}</span>` : ""}`;
    }
    observations(current, payload);
    updateCameras(); applyDisplay();
    byId("viewer-status").textContent = Object.keys(panelViews).length === 3 ? "三栏相机已同步；点击曲面可标出局部面编号。" : "3D 显示不完整，请展开固定多视图截图。";
    window.LEGACY_VIEWER_READY = payload.id;
  };
  ["gt","raw","post"].forEach(createPanel);
  ["wire-only","xray","highlight","face-colors"].forEach(id => byId(id).addEventListener("change", applyDisplay));
  document.querySelectorAll("[data-camera]").forEach(button => button.addEventListener("click", () => {
    const value = {iso:[-58,24],front:[-90,0],side:[0,0],top:[-90,89.999],back:[90,0]}[button.dataset.camera];
    cameraState.azim=value[0];cameraState.elev=value[1];cameraState.zoom=1;
    document.querySelectorAll("[data-camera]").forEach(peer => peer.setAttribute("aria-pressed", String(peer===button)));
    updateCameras();
  }));
  byId("protocol-toggle").addEventListener("click", () => { const opened=byId("protocol").hidden; byId("protocol").hidden=!opened;byId("protocol-toggle").setAttribute("aria-expanded",String(opened)); });
  byId("search").addEventListener("input", renderList);
  byId("filter").addEventListener("change", renderList);
  byId("previous").addEventListener("click",()=>{const index=filtered.findIndex(item=>item.id===current.id);if(index>0)selectCase(filtered[index-1].id);});
  byId("next").addEventListener("click",()=>{const index=filtered.findIndex(item=>item.id===current.id);if(index>=0&&index<filtered.length-1)selectCase(filtered[index+1].id);});
  byId("notes").addEventListener("input",()=>{
    notes[current.id]=byId("notes").value;
    try {localStorage.setItem(notesKey,JSON.stringify(notes));byId("notes-status").textContent="已保存到当前浏览器";}catch(_){byId("notes-status").textContent="浏览器不允许本地保存，请导出备注。";}
  });
  byId("export-notes").addEventListener("click",()=>{
    const blob=new Blob([JSON.stringify({collection:"legacy_failures_20261011",exported_at:new Date().toISOString(),notes},null,2)],{type:"application/json"});
    const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download="hola-failure-notes.json";link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  window.addEventListener("hashchange",()=>{const id=location.hash.slice(1);if(id!==current?.id)selectCase(id);});
  const initial=data.cases.find(item=>item.id===location.hash.slice(1)) || data.cases[0];
  selectCase(initial.id);
})();
