// ============================================================
// Navigation & Interactivity
// ============================================================
function switchTab(tab) {
  if (tab === "graph") {
    scheduleGraphRerender(30);
  }
}

function switchMobileView(view) {
  const section = document.getElementById(`panel-${view}`);
  if (section) {
    section.scrollIntoView({ behavior: 'smooth' });
  }
}

// ============================================================
// D3 KNOWLEDGE GRAPH (Empathetic Theme)
// ============================================================
const GRAPH = {
  nodes: [],
  edges: [],
  nodeMap: {},
  svg: null,
  gRoot: null,
  linkEls: null,
  dynamicLinkEls: null,
  nodeEls: null,
  labelEls: null,
  simulation: null,
  initialized: false,
  resizeTimer: null,
  width: 0,
  height: 0,
};

const GRAPH_STATE = {
  symptoms: [],
  conditions: [],
  journeyEdges: [],
};

const THEME = {
  bg: "transparent",
  dimmed: "rgba(0, 50, 98, 0.1)",
  dimmedNode: "#cbd5e1",
  activeSymp: "#3b82f6",
  activeCond: "#10b981",
  topCond: "#FF9966",
  edge: "#FF9966",
};

async function loadFullGraph() {
  try {
    const res = await fetch("/graph-data");
    const data = await res.json();
    GRAPH.nodes = data.nodes;
    GRAPH.edges = data.edges;
    GRAPH.nodeMap = {};
    GRAPH.nodes.forEach((n) => (GRAPH.nodeMap[n.id] = n));
    GRAPH.initialized = true;
    initD3Graph();
  } catch (e) {
    console.error("Failed to load full graph", e);
  }
}

function rememberGraphState(symptoms = [], conditions = [], journeyEdges = []) {
  GRAPH_STATE.symptoms = [...(symptoms || [])];
  GRAPH_STATE.conditions = [...(conditions || [])];
  GRAPH_STATE.journeyEdges = [...(journeyEdges || [])];
}

function hasGraphState() {
  return (
    GRAPH_STATE.symptoms.length > 0 ||
    GRAPH_STATE.conditions.length > 0 ||
    GRAPH_STATE.journeyEdges.length > 0
  );
}

function resetGraphView() {
  if (!GRAPH.initialized || !GRAPH.gRoot) return;
  const hint = document.getElementById("graph-empty-hint");
  if (hint) hint.style.opacity = "1";

  GRAPH.dynamicLinkEls = GRAPH.gRoot.select(".dyn-edges").selectAll("line").data([]).join("line");
  GRAPH.nodeEls.interrupt().attr("fill", THEME.dimmedNode).attr("r", (d) => (d.type === "condition" ? 7 : 5));
  GRAPH.labelEls.interrupt().attr("fill", "var(--navy)").attr("opacity", 0.4).attr("font-size", "8px");
  GRAPH.linkEls.interrupt().attr("stroke", THEME.dimmed).attr("stroke-width", 1).style("opacity", 0.5);
  document.getElementById("graph-stats").textContent = "Traversal Graph — awaiting input";
}

function initD3Graph() {
  const panel = document.getElementById("graph-panel");
  const W = panel.clientWidth || 800;
  const H = panel.clientHeight || 700;
  if (GRAPH.simulation) GRAPH.simulation.stop();
  GRAPH.width = W; GRAPH.height = H;
  const existing = document.getElementById("graph-svg");
  if (existing) existing.innerHTML = "";

  const svg = d3.select("#graph-svg").attr("width", "100%").attr("height", "100%").attr("viewBox", `0 0 ${W} ${H}`).style("font-family", "'Public Sans', sans-serif");
  GRAPH.svg = svg;
  const defs = svg.append("defs");
  defs.append("marker").attr("id", "timeline-arrow").attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto-start-reverse").append("path").attr("d", "M 0 0 L 10 5 L 0 10 z").attr("fill", THEME.edge);

  const g = svg.append("g").attr("class", "graph-root");
  GRAPH.gRoot = g;
  svg.call(d3.zoom().scaleExtent([0.1, 4]).on("zoom", (ev) => g.attr("transform", ev.transform)));

  GRAPH.simulation = d3.forceSimulation(GRAPH.nodes)
    .force("link", d3.forceLink(GRAPH.edges).id((d) => d.id).distance(60))
    .force("charge", d3.forceManyBody().strength(-280))
    .force("x", d3.forceX(W / 2).strength(0.04))
    .force("y", d3.forceY((d) => (d.type === "condition" ? H - 150 : 150)).strength(0.6))
    .force("collide", d3.forceCollide().radius(22));

  GRAPH.linkEls = g.append("g").selectAll("line").data(GRAPH.edges).join("line").attr("stroke", THEME.dimmed).attr("stroke-width", 1).style("opacity", 0.5);
  GRAPH.dynamicLinkEls = g.append("g").attr("class", "dyn-edges").selectAll("line");
  GRAPH.nodeEls = g.append("g").selectAll("circle").data(GRAPH.nodes).join("circle").attr("r", (d) => (d.type === "condition" ? 6 : 4)).attr("fill", THEME.dimmedNode).attr("stroke", "rgba(0,0,0,0.1)").attr("stroke-width", 1).call(d3.drag().on("start", (ev) => { if (!ev.active) GRAPH.simulation.alphaTarget(0.3).restart(); ev.subject.fx = ev.subject.x; ev.subject.fy = ev.subject.y; }).on("drag", (ev) => { ev.subject.fx = ev.x; ev.subject.fy = ev.y; }).on("end", (ev) => { if (!ev.active) GRAPH.simulation.alphaTarget(0); ev.subject.fx = null; ev.subject.fy = null; }));
  GRAPH.labelEls = g.append("g").selectAll("text").data(GRAPH.nodes).join("text").text((d) => d.id.replace(/_/g, " ")).attr("font-size", "8px").attr("fill", "var(--navy)").attr("opacity", 0.4).attr("dx", 10).attr("dy", 3).style("pointer-events", "none");

  GRAPH.simulation.on("tick", () => {
    GRAPH.linkEls.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    if (GRAPH.dynamicLinkEls) GRAPH.dynamicLinkEls.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    GRAPH.nodeEls.attr("cx", d => d.x).attr("cy", d => d.y);
    GRAPH.labelEls.attr("x", d => d.x).attr("y", d => d.y);
  });

  if (hasGraphState()) highlightGraph(GRAPH_STATE.symptoms, GRAPH_STATE.conditions, GRAPH_STATE.journeyEdges);
  else resetGraphView();
}

function resolveGraphNodeId(item) {
  if (!item) return null;
  const base = (typeof item === 'string' ? item : (item.condition_id || item.display || item.id || "")).toString().trim();
  if (!base) return null;
  
  // Try exact match, snake_case, and space_separated with both casings
  const lower = base.toLowerCase();
  const snake = lower.replace(/\s+/g, "_");
  const space = lower.replace(/_/g, " ");
  const candidates = [lower, snake, space, base];
  
  for (const id of candidates) {
    if (GRAPH.nodeMap[id]) return id;
    // Check if any node display matches
    const foundByDisplay = GRAPH.nodes.find(n => n.display && n.display.toLowerCase() === lower);
    if (foundByDisplay) return foundByDisplay.id;
  }
  return snake;
}

function resolveGraphNode(item) {
  const id = resolveGraphNodeId(item);
  return id ? GRAPH.nodeMap[id] || null : null;
}

function highlightGraph(symptoms, conditions, journeyEdges) {
  rememberGraphState(symptoms, conditions, journeyEdges);
  if (!GRAPH.initialized) return;
  const hint = document.getElementById("graph-empty-hint");
  if (hint) hint.style.opacity = "0";

  const activeSymptoms = new Set((symptoms || []).map(resolveGraphNodeId).filter(Boolean));
  const activeConditions = new Set((conditions || []).map(resolveGraphNodeId).filter(Boolean));
  const topCondId = conditions && conditions.length > 0 ? resolveGraphNodeId(conditions[0]) : null;

  const dynamicEdges = [];
  for (const edge of journeyEdges || []) {
    const fromNode = resolveGraphNode(edge.from);
    const toNode = resolveGraphNode(edge.to);
    if (fromNode && toNode) dynamicEdges.push({ source: fromNode, target: toNode, edge_type: edge.edge_type });
  }

  GRAPH.dynamicLinkEls = GRAPH.gRoot.select(".dyn-edges").selectAll("line").data(dynamicEdges).join("line")
    .attr("stroke", d => d.edge_type === "FIRST_SYMPTOM_TO_CONDITION" ? "#ea580c" : "#fb923c")
    .attr("stroke-width", d => d.edge_type === "FIRST_SYMPTOM_TO_CONDITION" ? 5 : 3)
    .attr("marker-end", "url(#timeline-arrow)");

  GRAPH.nodeEls.transition().duration(500)
    .attr("fill", d => d.id === topCondId ? THEME.topCond : (activeConditions.has(d.id) ? THEME.activeCond : (activeSymptoms.has(d.id) ? THEME.activeSymp : THEME.dimmedNode)))
    .attr("r", d => d.id === topCondId ? 12 : ((activeConditions.has(d.id) || activeSymptoms.has(d.id)) ? 8 : (d.type === "condition" ? 7 : 5)));

  GRAPH.labelEls.transition().duration(500).attr("opacity", d => (activeConditions.has(d.id) || activeSymptoms.has(d.id)) ? 1 : 0.4).attr("font-size", d => d.id === topCondId ? "12px" : "8px");

  document.getElementById("graph-stats").textContent = `Global Graph · ${activeSymptoms.size + activeConditions.size} active nodes · ${dynamicEdges.length} connections`;
}

function scheduleGraphRerender(delay = 120) {
  if (!GRAPH.initialized) return;
  window.clearTimeout(GRAPH.resizeTimer);
  GRAPH.resizeTimer = window.setTimeout(() => initD3Graph(), delay);
}

// ============================================================
// CHAT & DASHBOARD LOGIC
// ============================================================
const chatArea = document.getElementById("chat-area");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

let history = [];
let allSymptoms = [];
let isLoading = false;
let sessionId = null;

function initSession() {
  sessionId = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2, 15);
}
initSession();

inputEl.addEventListener("input", () => { sendBtn.disabled = inputEl.value.trim() === "" || isLoading; });
inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
sendBtn.addEventListener("click", sendMessage);

function addMessage(role, text) {
  const isUrgent = text.toUpperCase().includes("URGENT:");
  const msgDiv = document.createElement("div");
  msgDiv.className = `msg ${role}${isUrgent ? " urgent" : ""} flex gap-4 mb-6`;
  if (role === 'user') msgDiv.classList.add('justify-end');

  const avatar = document.createElement("div");
  avatar.className = `w-8 h-8 rounded-lg flex items-center justify-center text-[10px] font-bold shrink-0 ${role === 'bot' ? 'bg-blue-50 text-navy' : 'bg-navy text-white'}`;
  avatar.textContent = role === "bot" ? "AI" : "ME";

  const content = document.createElement("div");
  content.className = "max-w-[85%]";
  const bubble = document.createElement("div");
  bubble.className = `msg-bubble p-4 rounded-2xl text-sm leading-relaxed ${role === 'bot' ? 'bg-white border border-blue-50 text-navy shadow-sm' : 'bg-navy text-white'}`;

  if (role === "bot") bubble.innerHTML = DOMPurify.sanitize(marked.parse(text));
  else bubble.textContent = text;

  if (role === 'bot') { msgDiv.appendChild(avatar); msgDiv.appendChild(content); content.appendChild(bubble); }
  else { msgDiv.appendChild(content); content.appendChild(bubble); msgDiv.appendChild(avatar); }

  chatArea.appendChild(msgDiv);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function updateDashboard(data) {
  // Symptoms
  const sympList = document.getElementById("symp-list");
  if (data.extracted_symptoms?.length > 0) {
    sympList.innerHTML = data.extracted_symptoms.map(s => `<span class="px-3 py-1 bg-blue-50 text-navy text-[10px] font-bold rounded-full border border-blue-100">${s}</span>`).join("");
  } else sympList.innerHTML = '<div class="opacity-40 text-xs italic">Awaiting description...</div>';

  // Conditions
  const condList = document.getElementById("cond-list");
  if (data.top_conditions?.length > 0) {
    condList.innerHTML = data.top_conditions.map((c, index) => {
      const pct = Math.round(c.score * 100);
      let riskColor = c.severity === 'high' ? 'text-red-500' : (c.severity === 'medium' ? 'text-orange' : 'text-green-500');
      let barColor = c.severity === 'high' ? 'bg-red-500' : (c.severity === 'medium' ? 'bg-orange' : 'bg-green-500');

      // Robust confidence & symptom matching logic
      let badgeLabel = pct >= 70 ? "High Likelihood" : (pct >= 40 ? "Possible Match" : "Low Probability");
      let pillsHtml = "";
      if (index === 0 && (c.matched_symptoms || c.missing_symptoms)) {
        const matched = c.matched_symptoms || [];
        const atypical = c.missing_symptoms || [];
        pillsHtml = `<div class="mt-4 flex flex-wrap gap-2">`;
        if (matched.length > 0) pillsHtml += matched.map(s => `<span class="px-2 py-0.5 bg-green-50 text-green-700 text-[9px] font-bold rounded uppercase border border-green-100">✓ ${s}</span>`).join("");
        if (atypical.length > 0) pillsHtml += atypical.map(s => `<span class="px-2 py-0.5 bg-orange-50 text-orange text-[9px] font-bold rounded uppercase border border-orange-100">? ${s}</span>`).join("");
        pillsHtml += `</div>`;
      }

      return `
        <div class="condition-item">
          <div class="flex justify-between items-end mb-3">
            <div>
              <h5 class="text-lg font-bold text-navy mb-1">${c.display}</h5>
              <span class="${riskColor} text-[9px] font-bold uppercase tracking-widest">${badgeLabel}</span>
            </div>
            <span class="text-navy font-bold text-xs opacity-40">${pct}%</span>
          </div>
          <div class="h-2 w-full bg-navy/5 rounded-full overflow-hidden">
            <div class="h-full ${barColor} rounded-full transition-all duration-1000" style="width: ${pct}%"></div>
          </div>
          ${pillsHtml}
        </div>`;
    }).join("");
  } else condList.innerHTML = '<div class="opacity-40 text-xs italic">Awaiting analysis...</div>';

  // Traversal & Reasoning
  const travList = document.getElementById("trav-list");
  const chainList = document.getElementById("reasoning-chain-list");
  
  if (data.traversal_path?.length > 0) {
    const end = data.traversal_path[data.traversal_path.length - 1]?.to || "";
    travList.innerHTML = `
      <div class="p-6 bg-white/60 rounded-3xl border border-blue-50 shadow-sm">
        <p class="text-sm text-navy opacity-70 leading-relaxed mb-4">
          Reasoning points toward <strong class="text-orange">${end}</strong> as a primary clinical pattern based on the connections found.
        </p>
        <button class="text-[10px] font-bold text-orange uppercase tracking-widest hover:text-navy transition-colors" onclick="document.getElementById('panel-graph').scrollIntoView({behavior:'smooth'})">
          View Reasoning Graph →
        </button>
      </div>`;

    // Populate Reasoning Chain Timeline
    // Group traversal path into a few distinct steps for better legibility
    const steps = data.traversal_path.length > 5 ? [data.traversal_path[0], ...data.traversal_path.slice(-4)] : data.traversal_path;
    chainList.innerHTML = steps.map((step, i) => {
      const isStart = i === 0;
      const isEnd = i === steps.length - 1;
      const label = isEnd ? `Identified: ${step.to}` : `Input: ${step.from}`;
      const sub = isEnd ? "Conclusion based on matched indicators" : (isStart ? "Primary symptom triggered reasoning" : `Mapped to clinical pathway via graph`);
      const icon = isEnd ? '<i class="ti ti-check text-[10px]"></i>' : (isStart ? '<i class="ti ti-point text-[10px]"></i>' : '<div class="w-1 h-1 bg-navy/20 rounded-full"></div>');
      
      return `
        <div class="chain-step ${isEnd ? 'active' : ''}">
          <div class="chain-dot">${icon}</div>
          <span class="chain-label">${label}</span>
          <span class="chain-sub">${sub}</span>
        </div>`;
    }).join("");

    // Add Clinical Note to Chat if high severity
    if (data.top_conditions?.[0]?.severity === 'high') {
      const lastBotMsg = chatArea.querySelector('.msg.bot:last-child .msg-bubble');
      if (lastBotMsg && !lastBotMsg.querySelector('.clinical-note')) {
        const note = document.createElement("div");
        note.className = "clinical-note";
        note.innerHTML = `
          <i class="ti ti-alert-triangle"></i>
          <span class="clinical-note-text">Clinical Note: Priority evaluation for ${data.top_conditions[0].display} suggested based on reasoning chain.</span>
        `;
        lastBotMsg.appendChild(note);
      }
    }
  } else {
    travList.innerHTML = '<div class="opacity-40 text-xs italic">Awaiting analysis...</div>';
    chainList.innerHTML = '<div class="opacity-40 text-xs italic">Awaiting traversal...</div>';
  }

  // Red Flags
  const rfCard = document.getElementById("card-redflags");
  const rfList = document.getElementById("rf-list");
  if (data.red_flags_detected?.length > 0) {
    rfCard.style.display = "flex";
    rfList.innerHTML = data.red_flags_detected.map(f => `<span class="px-3 py-1 bg-red-500 text-white text-[10px] font-bold rounded-full shadow-lg shadow-red-200">${f}</span>`).join("");
  } else rfCard.style.display = "none";
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isLoading) return;
  inputEl.value = ""; isLoading = true; sendBtn.disabled = true;
  addMessage("user", text);
  history.push({ role: "user", content: text });
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history, session_id: sessionId, deep_research: document.getElementById('deep-research-toggle').checked })
    });
    const data = await res.json();
    sessionId = data.session_id; allSymptoms = data.symptom_timeline || data.extracted_symptoms || [];
    history.push({ role: "assistant", content: data.reply });
    addMessage("bot", data.reply);
    updateDashboard(data);
    highlightGraph(allSymptoms, data.top_conditions, data.journey_edges || []);
  } catch (err) { addMessage("bot", "System disruption. Please retry."); console.error(err); }
  isLoading = false; sendBtn.disabled = inputEl.value.trim() === "";
}

function clearChat() {
  if (sessionId) fetch("/session/clear", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) });
  initSession(); chatArea.innerHTML = ""; history = []; allSymptoms = [];
  updateDashboard({ extracted_symptoms: [], top_conditions: [], red_flags_detected: [] });
  resetGraphView();
  addMessage("bot", "Let's start fresh. What symptoms are you experiencing?");
}

// PDF Export Logic (Restored full configuration)
function generateClinicalPdf(sessionData, downloadButton) {
  const template = document.getElementById("clinical-report-template");
  if (!template || !sessionData || !downloadButton) return;
  template.querySelector("#pdf-gen-date").textContent = new Date().toUTCString();
  template.querySelector("#pdf-session-id").textContent = `${sessionId.substring(0, 8)}...`;
  const redFlags = sessionData.red_flags || sessionData.red_flags_detected || [];
  const rfSection = template.querySelector("#pdf-red-flags");
  if (redFlags.length > 0) { rfSection.style.display = "block"; template.querySelector("#pdf-rf-list").innerHTML = redFlags.map(rf => `<div class="report-item">• ${String(rf).toUpperCase()}</div>`).join(""); }
  else rfSection.style.display = "none";
  template.querySelector("#pdf-symptom-list").innerHTML = (sessionData.symptoms || []).map(s => {
    const name = String(s.name || s).replace(/_/g, " ").title();
    const details = [];
    if (s.duration) details.push(s.duration);
    if (s.severity) details.push(`${s.severity} severity`);
    const detailStr = details.length > 0 ? ` (${details.join(", ")})` : "";
    return `<div class="report-item">• ${name}${detailStr}</div>`;
  }).join("");
  template.querySelector("#pdf-condition-list").innerHTML = (sessionData.top_conditions || []).map((c, i) => `
    <div class="report-condition">
      <div class="report-condition-header"><strong>${i+1}. ${c.display}</strong> - ${String(c.severity).toUpperCase()} severity</div>
      <div class="report-condition-desc" style="font-size: 11px; opacity: 0.8; margin-top: 4px;">${c.description || ''}</div>
      <div class="report-condition-match" style="font-size: 10px; margin-top: 6px;">Matched Indicators: ${(c.matched_symptoms || []).join(", ")}</div>
    </div>`).join("");
  
  const sources = sessionData.rag_sources || sessionData.sources || [];
  template.querySelector("#pdf-source-list").innerHTML = sources.length > 0 
    ? sources.map(s => `<div class="report-item" style="font-size: 10px; margin-bottom: 4px;">• ${s}</div>`).join("")
    : '<div class="report-item" style="font-size: 10px; opacity: 0.5;">No external research references cited.</div>';
  
  const opt = { margin: 10, filename: `SymptomAssist_Report_${sessionId.substring(0,5)}.pdf`, image: { type: 'jpeg', quality: 0.98 }, html2canvas: { scale: 2 }, jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' } };
  const originalText = downloadButton.textContent;
  downloadButton.textContent = "Generating..."; downloadButton.disabled = true;
  html2pdf().set(opt).from(template).save().finally(() => { downloadButton.textContent = originalText; downloadButton.disabled = false; });
}

window.onload = () => {
  loadFullGraph();
  addMessage("bot", "Welcome to SymptomAssist AI. Describe your symptoms in detail to begin traversal.");
  
  const confirmModal = document.getElementById("confirmModal");
  document.getElementById("newChatBtn").addEventListener("click", () => confirmModal.showModal());
  document.getElementById("confirmBtn").addEventListener("click", () => { clearChat(); confirmModal.close(); });
  document.getElementById("cancelBtn").addEventListener("click", () => confirmModal.close());
  
  const summaryModal = document.getElementById("summaryModal");
  const lastSummaryData = { current: null };
  document.querySelectorAll("#viewSummaryBtn, #viewSummaryBtnTrigger").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!sessionId || allSymptoms.length === 0) return alert("Describe symptoms first.");
      summaryModal.showModal();
      document.getElementById("summary-text-area").textContent = "Assembling summary...";
      try {
        const res = await fetch(`/summary/${sessionId}`);
        const data = await res.json();
        document.getElementById("summary-text-area").textContent = data.text;
        lastSummaryData.current = data.data;
      } catch (e) { document.getElementById("summary-text-area").textContent = "Error loading summary."; }
    });
  });
  document.getElementById("closeSummaryBtn").addEventListener("click", () => summaryModal.close());
  document.getElementById("downloadPdfBtn").addEventListener("click", () => { if (lastSummaryData.current) generateClinicalPdf(lastSummaryData.current, document.getElementById("downloadPdfBtn")); });
  document.getElementById("printSummaryBtn").addEventListener("click", () => window.print());
  document.getElementById("copySummaryBtn").addEventListener("click", () => { navigator.clipboard.writeText(document.getElementById("summary-text-area").textContent); });

  if (localStorage.getItem("theme") === "dark") document.body.classList.add("dark-mode");
  document.getElementById("theme-toggle").addEventListener("click", () => {
    document.body.classList.toggle("dark-mode");
    localStorage.setItem("theme", document.body.classList.contains("dark-mode") ? "dark" : "light");
  });
};

function sendChip(text) { inputEl.value = text; inputEl.dispatchEvent(new Event('input')); sendBtn.click(); }
String.prototype.title = function() { return this.replace(/\b\w/g, l => l.toUpperCase()); };
