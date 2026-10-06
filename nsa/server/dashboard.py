"""Real-time Cognitive Visualization Dashboard for Neural State Architecture (NSA)."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NSA Cognitive Runtime & System 1 Visualizer</title>
  <style>
    :root {
      --bg-primary: #0A0D14;
      --bg-secondary: #101522;
      --bg-card: rgba(19, 26, 43, 0.7);
      --border-card: rgba(255, 255, 255, 0.08);
      --accent-cyan: #00F0FF;
      --accent-blue: #3B82F6;
      --accent-purple: #8B5CF6;
      --accent-emerald: #10B981;
      --accent-amber: #F59E0B;
      --accent-rose: #F43F5E;
      --text-main: #F1F5F9;
      --text-dim: #94A3B8;
      --text-muted: #64748B;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif; }
    body {
      background: radial-gradient(circle at 50% 0%, #151c33 0%, var(--bg-primary) 70%);
      color: var(--text-main);
      min-height: 100vh;
      padding: 24px;
      overflow-x: hidden;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border-card);
      margin-bottom: 24px;
    }
    .title-group h1 {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .badge-live {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-emerald);
      border: 1px solid rgba(16, 185, 129, 0.3);
      font-size: 11px;
      font-weight: 600;
      padding: 4px 8px;
      border-radius: 999px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .badge-live::before {
      content: "";
      width: 7px;
      height: 7px;
      background: var(--accent-emerald);
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
      100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .header-stats {
      display: flex;
      gap: 16px;
    }
    .stat-pill {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      backdrop-filter: blur(12px);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
    }
    .stat-pill span { color: var(--accent-cyan); font-weight: 600; }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 20px;
    }
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-card);
      backdrop-filter: blur(16px);
      border-radius: 14px;
      padding: 20px;
      position: relative;
      overflow: hidden;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .card::before {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
    }
    .card-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-dim);
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .col-4 { grid-column: span 4; }
    .col-6 { grid-column: span 6; }
    .col-8 { grid-column: span 8; }
    .col-12 { grid-column: span 12; }

    /* Heartbeat animation */
    .heartbeat-orb {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      background: radial-gradient(circle, var(--accent-cyan) 0%, rgba(0, 240, 255, 0.1) 70%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px rgba(0, 240, 255, 0.4);
      transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .heartbeat-active {
      transform: scale(1.15);
      box-shadow: 0 0 35px rgba(0, 240, 255, 0.8);
    }

    /* Decision indicators */
    .metric-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      font-size: 13px;
    }
    .progress-bar-bg {
      width: 100%;
      height: 7px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
    }
    .progress-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
      border-radius: 999px;
      transition: width 0.4s ease;
    }
    .memory-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      background: rgba(139, 92, 246, 0.15);
      color: var(--accent-purple);
      border: 1px solid rgba(139, 92, 246, 0.3);
    }
    .channel-bars {
      display: flex;
      gap: 12px;
      height: 110px;
      align-items: flex-end;
      padding-top: 10px;
    }
    .channel-col {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      height: 100%;
      justify-content: flex-end;
    }
    .channel-bar {
      width: 100%;
      background: linear-gradient(180deg, var(--accent-cyan), var(--accent-purple));
      border-radius: 4px;
      min-height: 4px;
      transition: height 0.3s ease;
    }
    .channel-lbl {
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 6px;
    }
    .channel-val {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 4px;
    }
    
    /* Memory list */
    .memory-list {
      max-height: 220px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .memory-item {
      background: rgba(255, 255, 255, 0.03);
      border-left: 3px solid var(--accent-purple);
      padding: 8px 12px;
      border-radius: 0 6px 6px 0;
      font-size: 12px;
    }
    .memory-item .meta {
      font-size: 10px;
      color: var(--text-muted);
      margin-bottom: 3px;
    }

    /* Sensory stimulus tester */
    .sensor-input-box {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }
    .sensor-input-box input {
      flex: 1;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-card);
      color: var(--text-main);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
    }
    .sensor-input-box input:focus { border-color: var(--accent-cyan); }
    .sensor-btn {
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      color: #fff;
      border: none;
      padding: 0 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      transition: opacity 0.2s;
    }
    .sensor-btn:hover { opacity: 0.9; }

    @media (max-width: 900px) {
      .col-4, .col-6, .col-8 { grid-column: span 12; }
    }
  </style>
</head>
<body>

  <div class="header">
    <div class="title-group">
      <h1>🧠 Neural State Architecture (NSA) <span style="font-size: 14px; color: var(--accent-purple);">CCE 6.4</span></h1>
      <div class="badge-live" id="live-indicator">Cognitive Substrate Active</div>
    </div>
    <div class="header-stats">
      <div class="stat-pill">Backend: <span id="hdr-backend">TRANSFORMERS</span></div>
      <div class="stat-pill">System 1: <span id="hdr-s1-model">Qwen2.5-0.5B</span></div>
      <div class="stat-pill">System 2: <span id="hdr-s2-model">Qwen2.5-3B</span></div>
      <div class="stat-pill">Safety Kernel: <span id="hdr-verdict" style="color: var(--accent-emerald);">COMMIT</span></div>
    </div>
  </div>

  <div class="grid">
    <!-- SYSTEM 1 COGNITIVE CONTROLLER -->
    <div class="card col-6">
      <div class="card-title">
        <span>⚡ System 1 Background Controller</span>
        <span class="memory-badge" id="s1-active-tag">Frozen HF Logits</span>
      </div>
      <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 20px;">
        <div class="heartbeat-orb" id="heartbeat-orb">
          <span style="font-size: 18px;">⚡</span>
        </div>
        <div>
          <div style="font-size: 12px; color: var(--text-dim);">Background Autonomous Heartbeat</div>
          <div style="font-size: 18px; font-weight: 700;" id="s1-heartbeat-hz">0.5 Hz (1 tick/2s)</div>
          <div style="font-size: 11px; color: var(--accent-cyan);" id="s1-route">Model Route: Fast System 1</div>
        </div>
      </div>

      <div class="metric-row">
        <span>Salience Assessment</span>
        <span id="s1-salience-val" style="font-weight: 600;">0.000</span>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="s1-salience-bar" style="width: 0%;"></div></div>
      <div style="height: 12px;"></div>

      <div class="metric-row">
        <span>Escalation Risk to System 2</span>
        <span id="s1-risk-val" style="font-weight: 600;">0.0%</span>
      </div>
      <div class="progress-bar-bg"><div class="progress-bar-fill" id="s1-risk-bar" style="width: 0%; background: linear-gradient(90deg, var(--accent-emerald), var(--accent-amber));"></div></div>
      <div style="height: 12px;"></div>

      <div class="metric-row">
        <span>Active Memory Policy</span>
        <span class="memory-badge" id="s1-memory-policy">working</span>
      </div>
      <div class="metric-row">
        <span>Safety Kernel Verdict</span>
        <span style="color: var(--accent-emerald); font-weight: 600;" id="s1-kernel-verdict">COMMIT [Step #0]</span>
      </div>
    </div>

    <!-- CCE CONTINUOUS DYNAMICS -->
    <div class="card col-6">
      <div class="card-title">
        <span>🧠 Continuous Cognitive Engine (CCE $X_t$)</span>
        <span style="font-size: 12px; color: var(--accent-cyan);" id="cce-ticks">#0 updates</span>
      </div>

      <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
        <div>
          <div style="font-size: 11px; color: var(--text-dim);">Wall-Clock Elapsed</div>
          <div style="font-size: 20px; font-weight: 700;" id="cce-elapsed">0.0s</div>
        </div>
        <div>
          <div style="font-size: 11px; color: var(--text-dim);">Epistemic Uncertainty</div>
          <div style="font-size: 20px; font-weight: 700; color: var(--accent-amber);" id="cce-uncertainty">0.0%</div>
        </div>
        <div>
          <div style="font-size: 11px; color: var(--text-dim);">Active Cognitive Goal</div>
          <div style="font-size: 16px; font-weight: 600; color: var(--accent-purple);" id="cce-goal">conversation</div>
        </div>
      </div>

      <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 4px;">Soft State Working Channels (4-D Latent Field)</div>
      <div class="channel-bars">
        <div class="channel-col">
          <div class="channel-val" id="c0-val">0.00</div>
          <div class="channel-bar" id="c0-bar" style="height: 20%;"></div>
          <div class="channel-lbl">W_0</div>
        </div>
        <div class="channel-col">
          <div class="channel-val" id="c1-val">0.00</div>
          <div class="channel-bar" id="c1-bar" style="height: 20%;"></div>
          <div class="channel-lbl">W_1</div>
        </div>
        <div class="channel-col">
          <div class="channel-val" id="c2-val">0.00</div>
          <div class="channel-bar" id="c2-bar" style="height: 20%;"></div>
          <div class="channel-lbl">W_2</div>
        </div>
        <div class="channel-col">
          <div class="channel-val" id="c3-val">0.00</div>
          <div class="channel-bar" id="c3-bar" style="height: 20%;"></div>
          <div class="channel-lbl">W_3</div>
        </div>
      </div>
    </div>

    <!-- SELECTIVE MEMORY VAULT -->
    <div class="card col-6">
      <div class="card-title">
        <span>💾 Selective Memory Vault</span>
        <span style="font-size: 12px; color: var(--text-dim);" id="mem-count">0 items stored</span>
      </div>
      <div class="memory-list" id="memory-items-container">
        <div style="color: var(--text-muted); font-size: 12px; padding: 10px;">No memories stored yet. Converse with the agent or inject stimulus.</div>
      </div>
    </div>

    <!-- SENSORY INGRESS & LIVE TESTER -->
    <div class="card col-6">
      <div class="card-title">
        <span>📡 Inject Sensory Stimulus to Continuous Engine</span>
        <span style="font-size: 12px; color: var(--accent-emerald);">Live Ingress</span>
      </div>
      <p style="font-size: 12px; color: var(--text-dim); line-height: 1.5;">
        Send sensory inputs directly to the Continuous Cognitive Engine without calling the full LLM. Observe System 1's real-time salience scoring and latent state perturbation instantly:
      </p>
      <div class="sensor-input-box">
        <input type="text" id="sensor-input" placeholder="e.g. Server CPU temperature spiked to 85°C..." />
        <button class="sensor-btn" onclick="sendStimulus()">Inject</button>
      </div>
      <div id="sensor-ack" style="font-size: 11px; color: var(--accent-cyan); margin-top: 8px; min-height: 16px;"></div>
    </div>
  </div>

  <script>
    let lastTickCount = -1;

    async function updateDashboard() {
      try {
        const res = await fetch("/health");
        if (!res.ok) return;
        const data = await res.ok ? await res.json() : {};

        // Headers
        document.getElementById("hdr-backend").textContent = (data.backend || "transformers").toUpperCase();
        document.getElementById("hdr-s1-model").textContent = (data.system_one_model || "Qwen2.5-0.5B").split("/").pop();
        document.getElementById("hdr-s2-model").textContent = (data.active_model || "Qwen2.5-3B").split("/").pop();
        document.getElementById("hdr-verdict").textContent = data.last_kernel_verdict || "COMMIT";

        // System 1
        if (data.system_one) {
          const s1 = data.system_one;
          document.getElementById("s1-salience-val").textContent = (s1.salience || 0).toFixed(3);
          document.getElementById("s1-salience-bar").style.width = Math.min(100, (s1.salience || 0) * 100) + "%";

          const esc = s1.decisions && s1.decisions.escalation ? s1.decisions.escalation : {};
          const risk = (esc.risk || 0) * 100;
          document.getElementById("s1-risk-val").textContent = risk.toFixed(1) + "%";
          document.getElementById("s1-risk-bar").style.width = Math.min(100, risk) + "%";

          document.getElementById("s1-memory-policy").textContent = s1.memory_policy || "working";
          document.getElementById("s1-route").textContent = "Model Route: " + (s1.selected_model || "Qwen2.5-0.5B").split("/").pop();
        }
        document.getElementById("s1-kernel-verdict").textContent = (data.last_kernel_verdict || "COMMIT") + " [Step #" + (data.state_step || 0) + "]";

        // CCE
        if (data.cce) {
          const cce = data.cce;
          document.getElementById("cce-elapsed").textContent = (cce.elapsed_seconds || 0).toFixed(1) + "s";
          document.getElementById("cce-uncertainty").textContent = ((cce.uncertainty || 0) * 100).toFixed(1) + "%";
          document.getElementById("cce-ticks").textContent = "#" + (cce.update_count || 0) + " updates";
          document.getElementById("cce-goal").textContent = cce.active_goal || "conversation";

          // Working channels
          const w = cce.working_state || [0, 0, 0, 0];
          for (let i = 0; i < 4; i++) {
            const val = w[i] !== undefined ? w[i] : 0;
            const elVal = document.getElementById("c" + i + "-val");
            const elBar = document.getElementById("c" + i + "-bar");
            if (elVal) elVal.textContent = val.toFixed(2);
            if (elBar) {
              const h = Math.max(5, Math.min(100, Math.abs(val) * 100));
              elBar.style.height = h + "%";
            }
          }

          // Heartbeat pulse animation
          if (cce.update_count !== lastTickCount) {
            lastTickCount = cce.update_count;
            const orb = document.getElementById("heartbeat-orb");
            orb.classList.add("heartbeat-active");
            setTimeout(() => orb.classList.remove("heartbeat-active"), 200);
          }
        }

        // Memory count
        document.getElementById("mem-count").textContent = (data.selective_memory_items || 0) + " items stored";

      } catch (e) {
        console.error("Dashboard poll failed", e);
      }
    }

    async function updateMemories() {
      try {
        const res = await fetch("/api/memory");
        if (!res.ok) return;
        const data = await res.json();
        const container = document.getElementById("memory-items-container");
        if (data.items && data.items.length > 0) {
          container.innerHTML = data.items.map(m => `
            <div class="memory-item">
              <div class="meta"><span class="memory-badge">${m.kind}</span> ${m.memory_id}</div>
              <div>${escapeHtml(m.content)}</div>
            </div>
          `).reverse().join("");
        }
      } catch (e) {}
    }

    async function sendStimulus() {
      const input = document.getElementById("sensor-input");
      const text = input.value.trim();
      if (!text) return;
      try {
        const res = await fetch("/api/cce/sensor", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: text, source: "dashboard_ui", importance: 0.8 })
        });
        const d = await res.json();
        document.getElementById("sensor-ack").textContent = "✓ Ingested: Salience=" + (d.salience ? d.salience.toFixed(3) : "n/a") + " (Triggered=" + d.triggered + ")";
        input.value = "";
        updateDashboard();
      } catch (e) {
        document.getElementById("sensor-ack").textContent = "Error: " + e.message;
      }
    }

    function escapeHtml(s) {
      return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Auto-poll
    setInterval(updateDashboard, 1000);
    setInterval(updateMemories, 3000);
    updateDashboard();
    updateMemories();
  </script>
</body>
</html>
"""
