// ============================================================
      // Tab Switcher
      // ============================================================
      function switchTab(tab) {
        document
          .getElementById("panel-diag")
          .classList.toggle("active", tab === "diag");
        document
          .getElementById("panel-graph")
          .classList.toggle("active", tab === "graph");
        document
          .getElementById("tab-diag")
          .classList.toggle("active", tab === "diag");
        document
          .getElementById("tab-graph")
          .classList.toggle("active", tab === "graph");
        if (tab === "graph") {
          scheduleGraphRerender(30);
        }
      }

      // ============================================================
      // PRE-DEFINED FULL KNOWLEDGE GRAPH (Force Simulation)
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

      const LAYOUT = {
        split: null,
        resizeObserver: null,
        viewportResizeTimer: null,
      };

      const THEME = {
        bg: "#0f172a",
        dimmed: "rgba(100, 116, 139, 0.25)",
        dimmedNode: "#334155",
        activeSymp: "#3b82f6",
        activeCond: "#10b981",
        topCond: "#ef4444",
        edge: "#f59e0b",
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

      function rememberGraphState(
        symptoms = [],
        conditions = [],
        journeyEdges = [],
      ) {
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
        if (hint) hint.style.display = "";

        GRAPH.dynamicLinkEls = GRAPH.gRoot
          .select(".dyn-edges")
          .selectAll("line")
          .data([])
          .join("line");

        GRAPH.nodeEls
          .interrupt()
          .attr("fill", THEME.dimmedNode)
          .attr("r", (d) => (d.type === "condition" ? 7 : 5));

        GRAPH.labelEls
          .interrupt()
          .attr("fill", "#94a3b8")
          .attr("opacity", 0.7)
          .attr("font-size", "8px")
          .style("font-weight", "400");

        GRAPH.linkEls
          .interrupt()
          .attr("stroke", THEME.dimmed)
          .attr("stroke-width", 1)
          .style("opacity", 0.5);

        document.getElementById("graph-stats").textContent =
          "Traversal Graph — awaiting input";
      }

      function initD3Graph() {
        const panel = document.getElementById("graph-panel");
        const W = panel.clientWidth || 560;
        const H = panel.clientHeight || 600;

        if (GRAPH.simulation) {
          GRAPH.simulation.stop();
        }

        GRAPH.width = W;
        GRAPH.height = H;

        const existing = document.getElementById("graph-svg");
        if (existing) existing.innerHTML = "";

        const svg = d3
          .select("#graph-svg")
          .attr("width", 800)
          .attr("height", 600)
          .attr("viewBox", "0 0 800 600")
          .style("background", THEME.bg)
          .style("font-family", "'Inter', 'Roboto', sans-serif");

        GRAPH.svg = svg;

        const defs = svg.append("defs");
        defs
          .append("marker")
          .attr("id", "timeline-arrow")
          .attr("viewBox", "0 0 10 10")
          .attr("refX", 8)
          .attr("refY", 5)
          .attr("markerWidth", 6)
          .attr("markerHeight", 6)
          .attr("orient", "auto-start-reverse")
          .append("path")
          .attr("d", "M 0 0 L 10 5 L 0 10 z")
          .attr("fill", "#fb923c");

        const g = svg.append("g").attr("class", "graph-root");
        GRAPH.gRoot = g;

        svg.call(
          d3
            .zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (ev) => g.attr("transform", ev.transform)),
        );

        // Forces: Greatly spaced out to prevent overlapping
        GRAPH.simulation = d3
          .forceSimulation(GRAPH.nodes)
          .force(
            "link",
            d3
              .forceLink(GRAPH.edges)
              .id((d) => d.id)
              .distance(60),
          )
          .force("charge", d3.forceManyBody().strength(-280))
          .force("x", d3.forceX(W / 2).strength(0.04))
          .force(
            "y",
            d3
              .forceY((d) => (d.type === "condition" ? H - 150 : 150))
              .strength(0.6),
          )
          .force("collide", d3.forceCollide().radius(22));

        // Edges
        GRAPH.linkEls = g
          .append("g")
          .selectAll("line")
          .data(GRAPH.edges)
          .join("line")
          .attr("stroke", THEME.dimmed)
          .attr("stroke-width", 1)
          .style("opacity", 0.5);

        // Dynamic sequential path edges
        GRAPH.dynamicLinkEls = g
          .append("g")
          .attr("class", "dyn-edges")
          .selectAll("line");

        // Nodes
        GRAPH.nodeEls = g
          .append("g")
          .selectAll("circle")
          .data(GRAPH.nodes)
          .join("circle")
          .attr("r", (d) => (d.type === "condition" ? 6 : 4))
          .attr("fill", THEME.dimmedNode)
          .attr("stroke", "#020617")
          .attr("stroke-width", 1)
          .call(drag(GRAPH.simulation));

        // Labels: Show all distinctly, better spacing
        GRAPH.labelEls = g
          .append("g")
          .selectAll("text")
          .data(GRAPH.nodes)
          .join("text")
          .text((d) => d.id.replace(/_/g, " "))
          .attr("font-size", "8px")
          .attr("fill", "#94a3b8")
          .attr("opacity", 0.7)
          .attr("dx", 10)
          .attr("dy", 3)
          .style("pointer-events", "none")
          .style("text-shadow", "1px 1px 2px #020617");

        GRAPH.simulation.on("tick", () => {
          GRAPH.linkEls
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);

          if (GRAPH.dynamicLinkEls) {
            GRAPH.dynamicLinkEls
              .attr("x1", (d) => d.source.x)
              .attr("y1", (d) => d.source.y)
              .attr("x2", (d) => d.target.x)
              .attr("y2", (d) => d.target.y);
          }

          GRAPH.nodeEls.attr("cx", (d) => d.x).attr("cy", (d) => d.y);

          GRAPH.labelEls.attr("x", (d) => d.x).attr("y", (d) => d.y);
        });

        function drag(sim) {
          return d3
            .drag()
            .on("start", (ev) => {
              if (!ev.active) sim.alphaTarget(0.3).restart();
              ev.subject.fx = ev.subject.x;
              ev.subject.fy = ev.subject.y;
            })
            .on("drag", (ev) => {
              ev.subject.fx = ev.x;
              ev.subject.fy = ev.y;
            })
            .on("end", (ev) => {
              if (!ev.active) sim.alphaTarget(0);
              ev.subject.fx = null;
              ev.subject.fy = null;
            });
        }

        if (hasGraphState()) {
          highlightGraph(
            GRAPH_STATE.symptoms,
            GRAPH_STATE.conditions,
            GRAPH_STATE.journeyEdges,
          );
        } else {
          resetGraphView();
        }
      }

      function normalizeGraphId(text) {
        return (text || "")
          .toString()
          .trim()
          .toLowerCase()
          .replace(/\s+/g, "_");
      }

      function resolveGraphNodeId(item) {
        if (!item) return null;

        const rawValues = [];
        if (typeof item === "string") {
          rawValues.push(item);
        } else {
          if (item.condition_id) rawValues.push(item.condition_id);
          if (item.display) rawValues.push(item.display);
          if (item.id) rawValues.push(item.id);
        }

        for (const raw of rawValues) {
          const base = (raw || "").toString().trim().toLowerCase();
          if (!base) continue;

          const candidates = [
            base,
            base.replace(/\s+/g, "_"),
            base.replace(/_/g, " "),
            base.replace(/[^a-z0-9_\s]/g, "").replace(/\s+/g, "_"),
          ];

          for (const id of candidates) {
            if (GRAPH.nodeMap[id]) return id;
          }
        }

        return rawValues.length > 0 ? normalizeGraphId(rawValues[0]) : null;
      }

      function resolveGraphNode(item) {
        const id = resolveGraphNodeId(item);
        return id ? GRAPH.nodeMap[id] || null : null;
      }

      function highlightGraph(symptoms, conditions, journeyEdges) {
        rememberGraphState(symptoms, conditions, journeyEdges);

        if (!GRAPH.initialized) return;

        if (
          (!symptoms || symptoms.length === 0) &&
          (!conditions || conditions.length === 0) &&
          (!journeyEdges || journeyEdges.length === 0)
        ) {
          resetGraphView();
          return;
        }

        const hint = document.getElementById("graph-empty-hint");
        if (hint) hint.style.display = "none";

        const activeSymptoms = new Set(
          (symptoms || []).map(resolveGraphNodeId).filter(Boolean),
        );
        const activeConditions = new Set(
          (conditions || []).map(resolveGraphNodeId).filter(Boolean),
        );
        const topCondId =
          conditions && conditions.length > 0
            ? resolveGraphNodeId(conditions[0])
            : null;

        // Render dynamic path strictly from backend-provided journey edges.
        const dynamicEdges = [];
        for (const edge of journeyEdges || []) {
          const fromNode = resolveGraphNode(edge.from);
          const toNode = resolveGraphNode(edge.to);
          if (fromNode && toNode) {
            dynamicEdges.push({
              source: fromNode,
              target: toNode,
              edge_type: edge.edge_type,
              score: edge.score,
            });
          }
        }

        // Draw dynamic journey connections over existing graph.
        GRAPH.dynamicLinkEls = GRAPH.gRoot
          .select(".dyn-edges")
          .selectAll("line")
          .data(dynamicEdges, (_, i) => i)
          .join("line")
          .attr("stroke", (d) => {
            if (d.edge_type === "FIRST_SYMPTOM_TO_CONDITION") return "#ea580c";
            if (d.edge_type === "SEQUENTIAL_SYMPTOM") return "#fb923c";
            return THEME.edge;
          })
          .attr("stroke-width", (d) => {
            if (d.edge_type === "FIRST_SYMPTOM_TO_CONDITION") return 5.5;
            if (d.edge_type === "SEQUENTIAL_SYMPTOM") return 4;
            return 2.5;
          })
          .attr("marker-end", "url(#timeline-arrow)")
          .style("pointer-events", "none");

        // Update nodes styling
        GRAPH.nodeEls
          .transition()
          .duration(500)
          .attr("fill", (d) => {
            if (d.id === topCondId) return THEME.topCond;
            if (activeConditions.has(d.id)) return THEME.activeCond;
            if (activeSymptoms.has(d.id)) return THEME.activeSymp;
            return THEME.dimmedNode;
          })
          .attr("r", (d) => {
            if (d.id === topCondId) return 12;
            if (activeConditions.has(d.id) || activeSymptoms.has(d.id))
              return 8;
            return d.type === "condition" ? 7 : 5;
          });

        // Update labels
        GRAPH.labelEls
          .transition()
          .duration(500)
          .attr("fill", (d) => {
            if (activeConditions.has(d.id) || activeSymptoms.has(d.id))
              return "#f8fafc";
            return "#94a3b8";
          })
          .attr("opacity", (d) => {
            return activeConditions.has(d.id) || activeSymptoms.has(d.id)
              ? 1.0
              : 0.7;
          })
          .attr("font-size", (d) => {
            if (d.id === topCondId) return "14px";
            if (activeConditions.has(d.id) || activeSymptoms.has(d.id))
              return "11px";
            return "8px";
          })
          .style("font-weight", (d) =>
            activeConditions.has(d.id) || activeSymptoms.has(d.id)
              ? "700"
              : "400",
          );

        // Dim the static base edges to emphasize the generated pathway
        GRAPH.linkEls
          .transition()
          .duration(500)
          .attr("stroke", THEME.dimmed)
          .attr("stroke-width", 1)
          .style("opacity", 0.3);

        // Update stats
        const hlNodes = activeSymptoms.size + activeConditions.size;
        document.getElementById("graph-stats").textContent =
          `Structured Global Graph · ${hlNodes} active nodes · ${dynamicEdges.length} active connections`;
      }

      function scheduleGraphRerender(delay = 120) {
        if (!GRAPH.initialized) return;

        window.clearTimeout(GRAPH.resizeTimer);
        GRAPH.resizeTimer = window.setTimeout(() => {
          const panel = document.getElementById("graph-panel");
          if (!panel) return;

          const nextWidth = panel.clientWidth || 0;
          const nextHeight = panel.clientHeight || 0;
          if (!nextWidth || !nextHeight) return;

          const widthChanged = Math.abs(nextWidth - GRAPH.width) > 2;
          const heightChanged = Math.abs(nextHeight - GRAPH.height) > 2;
          if (!widthChanged && !heightChanged) return;

          initD3Graph();
        }, delay);
      }

      function getMinPaneWidth() {
        return Math.max(Math.floor(window.innerWidth * 0.25), 260);
      }

      function destroySplitLayout() {
        if (LAYOUT.split) {
          LAYOUT.split.destroy();
          LAYOUT.split = null;
        }

        document.getElementById("chat-section")?.style.removeProperty("width");
        document
          .getElementById("chat-section")
          ?.style.removeProperty("flex-basis");
        document.getElementById("sidebar")?.style.removeProperty("width");
        document.getElementById("sidebar")?.style.removeProperty("flex-basis");
        document.body.classList.remove("is-resizing");
      }

      function initSplitLayout() {
        const isDesktop = window.innerWidth > 1024;

        if (!isDesktop) {
          destroySplitLayout();
          return;
        }

        if (typeof Split !== "function") {
          console.warn(
            "Split.js failed to load; resizable layout unavailable.",
          );
          return;
        }

        const sizes =
          LAYOUT.split && typeof LAYOUT.split.getSizes === "function"
            ? LAYOUT.split.getSizes()
            : [72, 28];

        destroySplitLayout();

        LAYOUT.split = Split(["#chat-section", "#sidebar"], {
          sizes,
          minSize: [getMinPaneWidth(), getMinPaneWidth()],
          gutterSize: 14,
          cursor: "col-resize",
          direction: "horizontal",
          onDragStart: () => {
            document.body.classList.add("is-resizing");
          },
          onDrag: () => {
            scheduleGraphRerender(80);
          },
          onDragEnd: () => {
            document.body.classList.remove("is-resizing");
            scheduleGraphRerender(40);
          },
        });
      }

      function initGraphResizeObserver() {
        const panel = document.getElementById("graph-panel");
        if (!panel || typeof ResizeObserver !== "function") return;

        if (LAYOUT.resizeObserver) {
          LAYOUT.resizeObserver.disconnect();
        }

        LAYOUT.resizeObserver = new ResizeObserver(() => {
          scheduleGraphRerender();
        });

        LAYOUT.resizeObserver.observe(panel);
      }

      function handleViewportResize() {
        window.clearTimeout(LAYOUT.viewportResizeTimer);
        LAYOUT.viewportResizeTimer = window.setTimeout(() => {
          initSplitLayout();
          scheduleGraphRerender();
        }, 140);
      }

      const chatArea = document.getElementById("chat-area");
      const inputEl = document.getElementById("input");
      const sendBtn = document.getElementById("send-btn");

      function generateSessionId() {
        return typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2, 15);
      }
      let history = [];
      let allSymptoms = [];
      let isLoading = false;
      let sessionId = generateSessionId();

      inputEl.addEventListener("input", () => {
        sendBtn.disabled = inputEl.value.trim() === "" || isLoading;
      });
      inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
      sendBtn.addEventListener("click", sendMessage);

      function addMessage(role, text) {
        const isUrgent = text.toUpperCase().includes("URGENT:");
        const msgDiv = document.createElement("div");
        msgDiv.className = `msg ${role}${isUrgent ? " urgent" : ""}`;

        const avatar = document.createElement("div");
        avatar.className = "msg-avatar";
        avatar.textContent = role === "bot" ? "AI" : "ME";

        const content = document.createElement("div");
        content.className = "msg-content";

        const bubble = document.createElement("div");
        bubble.className = "msg-bubble";

        if (role === "bot") {
          bubble.innerHTML = DOMPurify.sanitize(marked.parse(text));
        } else {
          bubble.textContent = text;
        }

        content.appendChild(bubble);
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(content);

        chatArea.appendChild(msgDiv);
        chatArea.scrollTop = chatArea.scrollHeight;
      }

      function showTyping() {
        const dots = document.createElement("div");
        dots.id = "typing-indicator";
        dots.className = "msg bot";
        dots.innerHTML = `
      <div class="msg-avatar">AI</div>
      <div class="msg-content">
        <div class="typing-dots">
          <div class="dot"></div>
          <div class="dot"></div>
          <div class="dot"></div>
        </div>
      </div>
    `;
        chatArea.appendChild(dots);
        chatArea.scrollTop = chatArea.scrollHeight;
      }

      function removeTyping() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
      }

      function clearChat() {
        // Tell the server to drop the session so symptom history is wiped
        if (sessionId) {
          fetch("/session/clear", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
          });
        }
        sessionId = null;
        chatArea.innerHTML = "";
        history = [];
        allSymptoms = [];
        rememberGraphState([], [], []);
        updateDashboard({
          extracted_symptoms: [],
          top_conditions: [],
          rag_sources: [],
          graph_followups: [],
          red_flags_detected: [],
        });
        resetGraphView();
        addMessage("bot", "Let's start fresh. What symptoms are you experiencing?");
      }

      function updateDashboard(data) {
        // Symptoms
        const sympList = document.getElementById("symp-list");
        if (data.extracted_symptoms?.length > 0) {
          sympList.innerHTML = data.extracted_symptoms
            .map((s) => `<span class="tag tag-symptom">${s}</span>`)
            .join("");
        } else {
          sympList.innerHTML =
            '<div class="empty-state">None identified yet</div>';
        }

        // Conditions
        const condList = document.getElementById("cond-list");
        if (data.top_conditions?.length > 0) {
          condList.innerHTML = data.top_conditions
            .map((c) => {
              const pct = Math.round(c.score * 100);
              return `
          <div class="condition-item">
            <div class="condition-header">
              <span class="condition-name">${c.display}</span>
              <span class="severity sev-${c.severity}">${c.severity}</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
            })
            .join("");
        } else {
          condList.innerHTML =
            '<div class="empty-state">Awaiting context...</div>';
        }

        // RAG
        const ragList = document.getElementById("rag-list");
        if (data.rag_sources?.length > 0) {
          ragList.innerHTML = data.rag_sources
            .map(
              (s) =>
                `<div class="source-item"><div class="source-bullet"></div>${s}</div>`,
            )
            .join("");
        } else {
          ragList.innerHTML = '<div class="empty-state">No docs matched</div>';
        }

        // Follow-ups
        const fuList = document.getElementById("fu-list");
        if (data.graph_followups?.length > 0) {
          fuList.innerHTML = data.graph_followups
            .map((q) => `<div class="q-item">${q}</div>`)
            .join("");
        } else {
          fuList.innerHTML = '<div class="empty-state">None currently</div>';
        }

        // BFS Traversal Path
        const travList = document.getElementById("trav-list");
        const path = data.traversal_path || [];
        if (path.length > 0) {
          const shown = path.slice(0, 12);
          travList.innerHTML =
            `<div class="traversal-header">Visited ${path.length} edges · showing first ${shown.length}</div>` +
            shown
              .map(
                (step, i) => `
          <div class="traversal-step" style="animation-delay:${i * 40}ms">
            <span class="ts-from">${step.from}</span>
            <span class="ts-arrow">→</span>
            <span class="ts-to">${step.to}</span>
            <span class="ts-weight">w=${step.weight}</span>
          </div>`,
              )
              .join("");
        } else {
          travList.innerHTML =
            '<div class="empty-state">No traversal yet</div>';
        }

        // Red Flags
        const rfCard = document.getElementById("card-redflags");
        const rfList = document.getElementById("rf-list");
        if (data.red_flags_detected?.length > 0) {
          rfCard.style.display = "block";
          rfList.innerHTML = data.red_flags_detected
            .map((f) => `<span class="tag tag-flag">${f}</span>`)
            .join("");
        } else {
          rfCard.style.display = "none";
        }
      }

      async function sendMessage() {
        const text = inputEl.value.trim();
        if (!text || isLoading) return;

        inputEl.value = "";
        sendBtn.disabled = true;
        isLoading = true;

        addMessage("user", text);
        history.push({ role: "user", content: text });
        showTyping();

        try {
          const res = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              messages: history,
              session_id: sessionId, // server echoes this back; null on first turn
            }),
          });

          if (!res.ok) throw new Error(`Status ${res.status}`);
          const data = await res.json();

          removeTyping();
          // Server now owns the symptom timeline; we just store the session_id
          sessionId = data.session_id;
          allSymptoms = data.symptom_timeline || data.extracted_symptoms || [];
          history.push({ role: "assistant", content: data.reply });
          addMessage("bot", data.reply);
          data.traversal_path = data.top_conditions?.[0]?.traversal_path || [];
          updateDashboard(data);
          highlightGraph(
            data.symptom_timeline || data.extracted_symptoms || [],
            data.top_conditions,
            data.journey_edges || [],
          );
        } catch (err) {
          removeTyping();
          addMessage("bot", "System disruption. Please retry.");
          console.error(err);
        }

        isLoading = false;
        sendBtn.disabled = inputEl.value.trim() === "";
        inputEl.focus();
      }

      // Init
      window.onload = () => {
        initSplitLayout();
        initGraphResizeObserver();
        window.addEventListener("resize", handleViewportResize);
        loadFullGraph();

        addMessage(
          "bot",
          "Welcome to SymptomAssist AI. Describe your symptoms in detail — the medical knowledge graph (built from 41 conditions and 130+ symptoms from a real dataset) will be traversed using BFS to identify likely conditions.\n\nHow are you feeling today?",
        );

        // ✅ MODAL FIX STARTS HERE
        const modal = document.getElementById("confirmModal");
        const confirmBtn = document.getElementById("confirmBtn");
        const cancelBtn = document.getElementById("cancelBtn");
        const newChatBtn = document.getElementById("newChatBtn");

        // OPEN modal
        newChatBtn.addEventListener("click", () => {
          modal.classList.add("show");
        });

        // CONFIRM
        confirmBtn.addEventListener("click", () => {
          clearChat();
          modal.classList.remove("show");
        });

        // CANCEL
        cancelBtn.addEventListener("click", () => {
          modal.classList.remove("show");
        });

        // CLICK OUTSIDE
        modal.addEventListener("click", (e) => {
          if (e.target === modal) {
            modal.classList.remove("show");
          }
        });
        // ✅ MODAL FIX ENDS HERE
        
        // --- Summary Modal Logic ---
        const summaryModal = document.getElementById("summaryModal");
        const viewSummaryBtn = document.getElementById("viewSummaryBtn");
        const closeSummaryBtn = document.getElementById("closeSummaryBtn");
        const copySummaryBtn = document.getElementById("copySummaryBtn");
        const printSummaryBtn = document.getElementById("printSummaryBtn");
        const summaryTextArea = document.getElementById("summary-text-area");

        viewSummaryBtn.addEventListener("click", async () => {
          if (!sessionId || allSymptoms.length === 0) {
            alert("Please describe some symptoms first to generate a summary.");
            return;
          }
          
          summaryModal.classList.add("show");
          summaryTextArea.textContent = "Assembling clinical summary...";
          
          try {
            const res = await fetch(`/summary/${sessionId}`);
            if (!res.ok) throw new Error("Failed to fetch summary");
            const data = await res.json();
            summaryTextArea.textContent = data.text;
          } catch (err) {
            summaryTextArea.textContent = "Error loading summary. Please try again later.";
            console.error(err);
          }
        });

        closeSummaryBtn.addEventListener("click", () => {
          summaryModal.classList.remove("show");
        });

        copySummaryBtn.addEventListener("click", () => {
          const text = summaryTextArea.textContent;
          navigator.clipboard.writeText(text).then(() => {
            const originalText = copySummaryBtn.textContent;
            copySummaryBtn.textContent = "Copied!";
            setTimeout(() => {
              copySummaryBtn.textContent = originalText;
            }, 2000);
          });
        });

        printSummaryBtn.addEventListener("click", () => {
          window.print();
        });

        summaryModal.addEventListener("click", (e) => {
          if (e.target === summaryModal) {
            summaryModal.classList.remove("show");
          }
        });

        // Theme toggle
        const toggleBtn = document.getElementById("theme-toggle");

        if (localStorage.getItem("theme") === "dark") {
          document.body.classList.add("dark-mode");
          toggleBtn.textContent = "☀️";
        }

        toggleBtn.addEventListener("click", () => {
          document.body.classList.toggle("dark-mode");

          if (document.body.classList.contains("dark-mode")) {
            localStorage.setItem("theme", "dark");
            toggleBtn.textContent = "☀️";
          } else {
            localStorage.setItem("theme", "light");
            toggleBtn.textContent = "🌙";
          }
        });
      };