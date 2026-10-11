const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const popChart = document.getElementById("popChart");
const brainCanvas = document.getElementById("brainCanvas");
const brainCtx = brainCanvas.getContext("2d");

let snap = null;
let selectedId = null;
let brainTrace = null;
let brainTraceTimer = null;
const history = { pop: [], t: [] };
const HISTORY = 240;
const trails = new Map();

function resize() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.floor(r.width * dpr);
  canvas.height = Math.floor(r.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize);

const ws = new WebSocket(
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`
);
ws.onmessage = (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "state") {
    snap = msg.payload;
    pushHistory();
    render();
    renderPanels();
  }
};

canvas.addEventListener("click", (e) => {
  if (!snap) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left;
  const y = e.clientY - r.top;
  let best = null, bd = 22;
  for (const a of snap.agents) {
    const d = Math.hypot(a.x - x, a.y - y);
    if (d < bd) { bd = d; best = a; }
  }
  selectedId = best ? best.id : null;
  fetch("/api/trace", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ agentId: selectedId }),
  });
  if (brainTraceTimer) clearInterval(brainTraceTimer);
  if (selectedId != null) {
    brainTraceTimer = setInterval(fetchBrainTrace, 200);
    fetchBrainTrace();
  } else {
    brainTrace = null;
    drawBrain();
  }
  renderPanels();
});

async function fetchBrainTrace() {
  if (selectedId == null) return;
  try {
    const r = await fetch(`/api/brain/${selectedId}`).then((r) => r.json());
    if (r.ok) {
      brainTrace = r.trace;
      drawBrain();
    }
  } catch {}
}

// ---- Ollama admin ----
async function loadOllama() {
  try {
    const r = await fetch("/api/admin/ollama").then((r) => r.json());
    if (r.ok) {
      document.getElementById("ollamaHost").value = r.config.host;
      document.getElementById("ollamaModel").value = r.config.model;
      document.getElementById("ollamaEnabled").checked = r.config.enabled;
    }
  } catch {}
}

async function saveOllama() {
  const body = {
    host: document.getElementById("ollamaHost").value,
    model: document.getElementById("ollamaModel").value,
    enabled: document.getElementById("ollamaEnabled").checked,
  };
  await fetch("/api/admin/ollama", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  loadOllama();
}

document.getElementById("ollamaSave").onclick = saveOllama;
loadOllama();

document.getElementById("pause").onclick = () =>
  fetch("/api/control", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action: snap?.paused ? "resume" : "pause" }),
  });

document.getElementById("reset").onclick = () => {
  if (confirm("Reset the world?"))
    fetch("/api/control", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ action: "reset" }),
    });
};

function pushHistory() {
  if (!snap) return;
  history.t.push(snap.t);
  history.pop.push(snap.stats.population);
  while (history.t.length > HISTORY) {
    history.t.shift();
    history.pop.shift();
  }
}

function drawChart(c, data, color) {
  const W = c.width, H = c.height;
  const g = c.getContext("2d");
  g.clearRect(0, 0, W, H);
  if (data.length < 2) return;
  const lo = Math.min(...data), hi = Math.max(...data);
  const span = Math.max(1e-6, hi - lo);
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((data[i] - lo) / span) * (H - 8) - 4;
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  }
  g.stroke();
}

function drawBrain() {
  const W = brainCanvas.width, H = brainCanvas.height;
  brainCtx.clearRect(0, 0, W, H);
  brainCtx.fillStyle = "#080c14";
  brainCtx.fillRect(0, 0, W, H);

  if (!brainTrace) {
    brainCtx.fillStyle = "#4a5568";
    brainCtx.font = "12px system-ui";
    brainCtx.fillText("Select an agent…", 12, 22);
    return;
  }

  const layers = [
    { name: "input", data: brainTrace.input, rgb: "74,212,138" },
    { name: "h1", data: brainTrace.h1, rgb: "106,169,255" },
    { name: "h2", data: brainTrace.h2, rgb: "167,139,250" },
    { name: "logits", data: brainTrace.logits, rgb: "245,158,11" },
    { name: "policy", data: brainTrace.probs, rgb: "239,68,68" },
  ];

  const labelW = 48;
  const rowH = (H - 14) / layers.length;
  const maxLen = Math.max(...layers.map((l) => l.data.length));
  const cellW = (W - labelW) / maxLen;

  brainCtx.font = "10px ui-monospace, monospace";
  for (let r = 0; r < layers.length; r++) {
    const layer = layers[r];
    brainCtx.fillStyle = "#8892a6";
    brainCtx.fillText(layer.name, 4, r * rowH + rowH / 2 + 3);
    for (let i = 0; i < layer.data.length; i++) {
      const v = layer.data[i];
      const intensity = Math.min(1, Math.abs(v));
      const x = labelW + i * cellW;
      const y = r * rowH;
      brainCtx.fillStyle =
        v >= 0
          ? `rgba(${layer.rgb},${intensity})`
          : `rgba(255,80,80,${intensity})`;
      brainCtx.fillRect(x, y + 2, Math.max(1, cellW - 0.5), rowH - 4);
    }
  }

  brainCtx.fillStyle = "#8892a6";
  brainCtx.font = "9px ui-monospace, monospace";
  const acts = brainTrace.actions ?? [];
  for (let i = 0; i < acts.length && i < 12; i++) {
    const x = labelW + i * cellW;
    brainCtx.fillText(acts[i].slice(0, 4), x + 1, H - 3);
  }
}

function render() {
  if (!snap) return;
  const r = canvas.getBoundingClientRect();
  ctx.fillStyle = "#050810";
  ctx.fillRect(0, 0, r.width, r.height);

  ctx.strokeStyle = "#0e1420";
  for (let x = 0; x < r.width; x += 40) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, r.height); ctx.stroke();
  }
  for (let y = 0; y < r.height; y += 40) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(r.width, y); ctx.stroke();
  }

  for (const h of snap.hazards) {
    const g = ctx.createRadialGradient(h.x, h.y, 0, h.x, h.y, h.r);
    g.addColorStop(0, "rgba(255,70,70,0.30)");
    g.addColorStop(1, "rgba(255,70,70,0)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(h.x, h.y, h.r, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(255,120,120,0.35)";
    ctx.beginPath(); ctx.arc(h.x, h.y, h.r, 0, Math.PI * 2); ctx.stroke();
  }

  for (const m of snap.mirrors) {
    ctx.strokeStyle = "rgba(140,190,255,0.35)";
    ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(140,190,255,0.06)";
    ctx.beginPath(); ctx.arc(m.x, m.y, m.r, 0, Math.PI * 2); ctx.fill();
  }

  for (const f of snap.food) {
    ctx.fillStyle = "#4ad48a";
    ctx.beginPath(); ctx.arc(f.x, f.y, 2.5, 0, Math.PI * 2); ctx.fill();
  }

  for (const a of snap.agents) {
    let t = trails.get(a.id);
    if (!t) { t = []; trails.set(a.id, t); }
    t.push({ x: a.x, y: a.y });
    if (t.length > 30) t.shift();
    ctx.strokeStyle = `hsla(${a.hue} 70% 60% / 0.15)`;
    ctx.beginPath();
    for (let i = 0; i < t.length; i++) {
      if (i === 0) ctx.moveTo(t[i].x, t[i].y);
      else ctx.lineTo(t[i].x, t[i].y);
    }
    ctx.stroke();

    const rad = 5 + (Math.min(130, a.energy) / 130) * 5;
    const lightness = 40 + a.pleasure * 30;
    ctx.fillStyle = `hsl(${a.hue} 75% ${lightness}%)`;
    ctx.beginPath(); ctx.arc(a.x, a.y, rad, 0, Math.PI * 2); ctx.fill();

    if (a.hasMark) {
      ctx.fillStyle = `hsl(${a.markHue} 90% 60%)`;
      const mx = a.x + Math.cos(a.id * 1.7) * (rad * 0.55);
      const my = a.y + Math.sin(a.id * 1.7) * (rad * 0.55);
      ctx.beginPath(); ctx.arc(mx, my, 2.5, 0, Math.PI * 2); ctx.fill();
      if (a.markTouched) {
        ctx.strokeStyle = "#4ad48a";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(a.x, a.y, rad + 6, 0, Math.PI * 2); ctx.stroke();
        ctx.lineWidth = 1;
      }
    }

    if (a.curiosity > 0.5) {
      ctx.strokeStyle = `rgba(167,139,250,${(a.curiosity - 0.5) * 1.2})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(a.x, a.y, rad + 3, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 1;
    }

    if (a.pleasure > 0.35) {
      ctx.strokeStyle = `rgba(120,255,180,${(a.pleasure - 0.35) * 1.1})`;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(a.x, a.y, rad + 5, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 1;
    }

    if (a.integrity < 70) {
      ctx.strokeStyle = `rgba(255,80,80,${(70 - a.integrity) / 70})`;
      ctx.beginPath(); ctx.arc(a.x, a.y, rad + 2, 0, Math.PI * 2); ctx.stroke();
    }

    if (a.id === selectedId) {
      ctx.strokeStyle = "#6aa9ff";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(a.x, a.y, rad + 8, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 1;
    }

    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = "rgba(220,230,245,0.75)";
    ctx.fillText(a.action, a.x + rad + 3, a.y + 3);

    // Show only ideas from the last four seconds
    if (a.lastIdea && snap.t - a.lastIdeaT < 4) {
      ctx.font = "10px system-ui, sans-serif";
      const text = a.lastIdea.slice(0, 40);
      const w2 = ctx.measureText(text).width + 10;
      // Color by source: LLM = warm yellow, curiosity = purple, mirror = green
      if (a.lastIdeaSource === "llm") {
        ctx.fillStyle = "rgba(245,200,90,0.92)";
      } else if (a.lastIdeaSource === "mirror") {
        ctx.fillStyle = "rgba(120,255,180,0.95)";
      } else {
        ctx.fillStyle = "rgba(167,139,250,0.85)";
      }
      ctx.fillRect(a.x + 10, a.y - 28, w2, 16);
      ctx.fillStyle = "#0b0d12";
      ctx.fillText(text, a.x + 15, a.y - 16);
    }
  }
}

function renderPanels() {
  if (!snap) return;
  const s = snap.stats;
  document.getElementById("stats").innerHTML = `
    <dt>time</dt><dd>${snap.t.toFixed(1)}s</dd>
    <dt>population</dt><dd>${s.population}</dd>
    <dt>births</dt><dd>${s.births}</dd>
    <dt>deaths</dt><dd>${s.deaths}</dd>
    <dt>max generation</dt><dd>${s.maxGeneration}</dd>
    <dt>mirror-mark task ✓</dt><dd>${s.mirrorTestsPassed}</dd>
    <dt>avg pleasure</dt><dd>${s.avgPleasure}</dd>
    <dt>avg reward/s</dt><dd>${s.avgRewardPerSec}</dd>
  `;

  document.getElementById("ollamaStats").textContent =
    `calls: ${s.ollama?.calls ?? 0} · errors: ${s.ollama?.errors ?? 0}` +
    (s.ollama?.lastError ? ` · ${s.ollama.lastError}` : "");

  drawChart(popChart, history.pop, "#6aa9ff");

  const sel = snap.agents.find((a) => a.id === selectedId);
  const selEl = document.getElementById("selected");
  if (!sel) {
    selEl.innerHTML = '<p class="hint">Click an agent to inspect it.</p>';
  } else {
    selEl.innerHTML = `
      <div><b>${sel.name}</b> <span class="badge">${sel.action}</span></div>
      <div class="kv" style="margin-top:6px">
        <dt>generation</dt><dd>${sel.generation}</dd>
        <dt>age</dt><dd>${sel.age}s</dd>
        <dt>energy</dt><dd>${sel.energy}</dd>
        <dt>integrity</dt><dd>${sel.integrity}</dd>
        <dt>pleasure</dt><dd>${sel.pleasure}</dd>
        <dt>curiosity</dt><dd>${sel.curiosity}</dd>
        <dt>novelty</dt><dd>${sel.novelty}</dd>
        <dt>mark</dt><dd>${sel.hasMark ? (sel.markTouched ? "touched ✓" : "active") : "no"}</dd>
        <dt>in mirror</dt><dd>${sel.inMirror ? "yes" : "no"}</dd>
        <dt>mark progress</dt><dd>${sel.markProgress}</dd>
      </div>
      ${sel.narrative ? `<div style="margin-top:8px;padding:8px;background:#0a0e17;border-radius:6px;border:1px solid #1e2a3d;font-style:italic;color:#c9d3e5;font-size:11.5px;line-height:1.5">"${escapeHtml(sel.narrative)}"</div>` : ""}
    `;
  }

  const ideasEl = document.getElementById("ideas");
  if (sel && sel.lastIdea) {
    ideasEl.innerHTML = `<div style="color:#a78bfa">"${escapeHtml(sel.lastIdea)}"</div>`;
  } else {
    ideasEl.textContent = "None yet.";
  }

  const top = snap.agents.slice().sort((a, b) => b.totalReward - a.totalReward).slice(0, 10);
  document.getElementById("agents").innerHTML = top
    .map(
      (a) => `
    <div class="agent ${a.id === selectedId ? "active" : ""}" data-id="${a.id}">
      <div class="name">${a.name} <span class="badge">gen ${a.generation}</span>
        ${a.hasMark ? '<span class="badge" style="color:#f59e0b">mark</span>' : ""}
        ${a.curiosity > 0.5 ? '<span class="badge" style="color:#a78bfa">curious</span>' : ""}
      </div>
      <div class="meta">${a.action} · E ${a.energy.toFixed(0)} · R ${a.totalReward}</div>
      <div class="bar"><i style="width:${Math.min(100, a.pleasure * 100)}%;background:#4ad48a"></i></div>
      <div class="bar"><i style="width:${Math.min(100, a.curiosity * 100)}%;background:#a78bfa"></i></div>
    </div>
  `
    )
    .join("");

  document.querySelectorAll(".agent").forEach((el) => {
    el.addEventListener("click", () => {
      selectedId = Number(el.dataset.id);
      fetch("/api/trace", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ agentId: selectedId }),
      });
      if (brainTraceTimer) clearInterval(brainTraceTimer);
      brainTraceTimer = setInterval(fetchBrainTrace, 200);
      fetchBrainTrace();
      renderPanels();
    });
  });

  const log = document.getElementById("log");
  log.innerHTML = snap.events
    .slice()
    .reverse()
    .slice(0, 40)
    .map((e) => `<div>[${e.t.toFixed(1)}] ${escapeHtml(e.text)}</div>`)
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

resize();