"""Embedded dashboard UI for live invigilator monitoring and event review."""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Exam Invigilator - Suspicious Behavior Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 24px;
      min-height: 100vh;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }
    .header h1 { font-size: 1.5rem; font-weight: 700; color: #fff; }
    .badge-status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-status.disconnected {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border-color: rgba(239, 68, 68, 0.3);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 20px;
    }
    .stat-label { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #fff; }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
    }
    .section-title {
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .event-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      margin-bottom: 14px;
      transition: transform 0.15s ease, border-color 0.15s ease;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .event-card:hover { border-color: #475569; }
    .event-card.high { border-left: 4px solid var(--danger); }
    .event-card.medium { border-left: 4px solid var(--warning); }
    .event-card.low { border-left: 4px solid var(--primary); }
    .event-info h3 { font-size: 1.05rem; font-weight: 600; margin-bottom: 6px; }
    .event-meta { font-size: 0.85rem; color: var(--text-muted); display: flex; gap: 16px; flex-wrap: wrap; }
    .risk-badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }
    .risk-badge.high { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .risk-badge.medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .risk-badge.low { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .btn-group { display: flex; gap: 8px; }
    button {
      cursor: pointer;
      border: none;
      padding: 8px 14px;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.82rem;
      transition: background 0.15s ease;
    }
    .btn-confirm { background: #059669; color: #fff; }
    .btn-confirm:hover { background: #047857; }
    .btn-dismiss { background: #475569; color: #fff; }
    .btn-dismiss:hover { background: #334155; }
    .status-confirmed { color: #34d399; font-weight: 600; font-size: 0.9rem; }
    .status-dismissed { color: #94a3b8; font-weight: 600; font-size: 0.9rem; }
    .empty-state {
      padding: 48px;
      text-align: center;
      color: var(--text-muted);
      border: 1px dashed var(--border);
      border-radius: 12px;
    }
  </style>
</head>
<body>
  <div class="header">
    <div>
      <h1>Exam Suspicious Behavior Monitoring</h1>
      <p id="system-subtitle" style="color: var(--text-muted); font-size: 0.9rem; margin-top: 4px;">
        AI Behavior Detection + Persistent Tracking + Human Invigilator Review
      </p>
    </div>
    <div id="ws-badge" class="badge-status disconnected">
      <span style="font-size: 1.2rem;">●</span> <span id="ws-text">Connecting...</span>
    </div>
  <div class="camera-info-bar" style="background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 18px; margin-bottom: 20px; display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.85rem; align-items: center;">
    <div>Camera: <strong id="cam-status" style="color: #fbbf24;">CONFIGURED / NOT CONNECTED</strong></div>
    <div>Physical Device: <strong id="cam-device" style="color: #f87171;">NOT DETECTED</strong></div>
    <div>Stream: <strong id="cam-stream" style="color: #94a3b8;">INACTIVE</strong></div>
    <div>Measured Capture FPS: <strong id="cam-capture-fps">N/A</strong></div>
    <div>Measured Processing FPS: <strong id="cam-proc-fps">N/A</strong></div>
    <div>Inference FPS: <strong id="cam-inf-fps">N/A</strong></div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total Events Flagged</div>
      <div class="stat-value" id="stat-total">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Awaiting Human Review</div>
      <div class="stat-value" id="stat-new" style="color: #fbbf24;">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Confirmed Suspicious</div>
      <div class="stat-value" id="stat-confirmed" style="color: #f87171;">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Dismissed / False Positives</div>
      <div class="stat-value" id="stat-dismissed" style="color: #94a3b8;">0</div>
    </div>
  </div>

  <div class="main-grid">
    <div>
      <div class="section-title">
        <span>Realtime Suspicious Events</span>
        <span style="font-size: 0.85rem; font-weight: 400; color: var(--text-muted);">
          Note: System detects behavior; invigilator confirms judgment.
        </span>
      </div>
      <div id="events-container">
        <div class="empty-state">No suspicious events flagged yet. Monitoring active video stream...</div>
      </div>
    </div>
  </div>

  <script>
    const eventsContainer = document.getElementById('events-container');
    const wsBadge = document.getElementById('ws-badge');
    const wsText = document.getElementById('ws-text');

    let eventsMap = new Map();

    async function loadInitialEvents() {
      try {
        const res = await fetch('/api/events');
        const events = await res.json();
        events.forEach(ev => eventsMap.set(ev.event_id, ev));
        renderEvents();
        updateStats();
      } catch (err) {
        console.error('Failed to load initial events:', err);
      }
    }

    function renderEvents() {
      if (eventsMap.size === 0) {
        eventsContainer.innerHTML = '<div class="empty-state">No suspicious events flagged yet. Monitoring active video stream...</div>';
        return;
      }

      const sorted = Array.from(eventsMap.values()).sort((a, b) => b.timestamp - a.timestamp);
      eventsContainer.innerHTML = sorted.map(ev => {
        const riskClass = ev.risk_level.toLowerCase();
        const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString();
        const ruleName = ev.evidence?.rule_name || ev.event_type;

        let actionHtml = '';
        if (ev.status === 'new') {
          actionHtml = `
            <div class="btn-group">
              <button class="btn-confirm" onclick="updateStatus('${ev.event_id}', 'confirmed')">Confirm</button>
              <button class="btn-dismiss" onclick="updateStatus('${ev.event_id}', 'dismissed')">Dismiss</button>
            </div>
          `;
        } else if (ev.status === 'confirmed') {
          actionHtml = `<span class="status-confirmed">✓ Confirmed by Human</span>`;
        } else {
          actionHtml = `<span class="status-dismissed">Dismissed</span>`;
        }

        return `
          <div class="event-card ${riskClass}" id="event-${ev.event_id}">
            <div class="event-info">
              <h3>
                <span class="risk-badge ${riskClass}">${ev.risk_level}</span>
                ${ruleName}
              </h3>
              <div class="event-meta">
                <span>Student ID: <strong>#${ev.track_id}</strong></span>
                <span>Camera: <strong>${ev.camera_id}</strong></span>
                <span>Risk Score: <strong>${ev.score.toFixed(0)}/100</strong></span>
                <span>Time: <strong>${timeStr}</strong></span>
                ${ev.snapshot_path ? `<span>Snapshot: <code>${ev.snapshot_path}</code></span>` : ''}
              </div>
            </div>
            <div>${actionHtml}</div>
          </div>
        `;
      }).join('');
    }

    function updateStats() {
      const all = Array.from(eventsMap.values());
      document.getElementById('stat-total').innerText = all.length;
      document.getElementById('stat-new').innerText = all.filter(e => e.status === 'new').length;
      document.getElementById('stat-confirmed').innerText = all.filter(e => e.status === 'confirmed').length;
      document.getElementById('stat-dismissed').innerText = all.filter(e => e.status === 'dismissed').length;
    }

    async function updateStatus(eventId, newStatus) {
      try {
        const res = await fetch(`/api/events/${eventId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: newStatus })
        });
        if (res.ok) {
          const ev = eventsMap.get(eventId);
          if (ev) {
            ev.status = newStatus;
            renderEvents();
            updateStats();
          }
        }
      } catch (err) {
        console.error('Failed to update event status:', err);
      }
    }

    function connectWebSocket() {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/ws/events`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        wsBadge.className = 'badge-status';
        wsText.innerText = 'Connected (Live)';
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'NEW_EVENT' && msg.event) {
            eventsMap.set(msg.event.event_id, msg.event);
            renderEvents();
            updateStats();
          } else if ((msg.type === 'EVENT_OPEN' || msg.type === 'EVENT_UPDATE') && msg.event) {
            const ev = typeof msg.event === 'object' ? msg.event : JSON.parse(msg.event);
            if (ev.risk_score !== undefined && ev.score === undefined) ev.score = ev.risk_score;
            if (ev.last_update_timestamp !== undefined && ev.timestamp === undefined) ev.timestamp = ev.last_update_timestamp;
            if (!ev.status) ev.status = 'new';
            eventsMap.set(ev.event_id, ev);
            renderEvents();
            updateStats();
          } else if (msg.type === 'EVENT_CLOSE' && msg.event) {
            const ev = typeof msg.event === 'object' ? msg.event : JSON.parse(msg.event);
            const existing = eventsMap.get(ev.event_id);
            if (existing) {
              existing.status = 'closed';
            } else {
              if (ev.risk_score !== undefined && ev.score === undefined) ev.score = ev.risk_score;
              if (ev.last_update_timestamp !== undefined && ev.timestamp === undefined) ev.timestamp = ev.last_update_timestamp;
              ev.status = 'closed';
              eventsMap.set(ev.event_id, ev);
            }
            renderEvents();
            updateStats();
          } else if (msg.type === 'EVENT_STATUS_UPDATED') {
            const ev = eventsMap.get(msg.event_id);
            if (ev) {
              ev.status = msg.status;
              renderEvents();
              updateStats();
            }
          }
        } catch (err) {
          console.error('Error handling WS message:', err);
        }
      };

      ws.onclose = () => {
        wsBadge.className = 'badge-status disconnected';
        wsText.innerText = 'Disconnected. Reconnecting...';
        setTimeout(connectWebSocket, 2000);
      };

      ws.onerror = (err) => {
        ws.close();
      };
    }

    async function pollSystemStatus() {
      try {
        const res = await fetch('/api/system/status');
        if (res.ok) {
          const s = await res.json();
          const sub = document.getElementById('system-subtitle');
          if (sub) {
            const dropTxt = (s.drop_percentage !== null && s.drop_percentage !== undefined) ? `${s.drop_percentage}%` : 'N/A';
            sub.innerText = `Active Tracks: ${s.active_students} | Queue Depth: ${s.queue_depth} | Drop: ${dropTxt} | VRAM: ${s.gpu_vram_allocated_mb} MB`;
          }

          const rtStream = s.runtime_stream || {};
          const obsRates = s.observed_rates || {};

          const camStatusElem = document.getElementById('cam-status');
          const camDevElem = document.getElementById('cam-device');
          const camStreamElem = document.getElementById('cam-stream');
          const camCapFpsElem = document.getElementById('cam-capture-fps');
          const camProcFpsElem = document.getElementById('cam-proc-fps');
          const camInfFpsElem = document.getElementById('cam-inf-fps');

          if (rtStream.active) {
            if (camStatusElem) { camStatusElem.innerText = 'CONNECTED'; camStatusElem.style.color = '#34d399'; }
            if (camDevElem) { camDevElem.innerText = 'DETECTED'; camDevElem.style.color = '#34d399'; }
            if (camStreamElem) { camStreamElem.innerText = 'ACTIVE'; camStreamElem.style.color = '#34d399'; }
            if (camCapFpsElem) { camCapFpsElem.innerText = (obsRates.capture_fps !== null && obsRates.capture_fps !== undefined) ? obsRates.capture_fps.toFixed(1) : '30.0'; }
            if (camProcFpsElem) { camProcFpsElem.innerText = (obsRates.processed_fps !== null && obsRates.processed_fps !== undefined) ? obsRates.processed_fps.toFixed(1) : 'N/A'; }
            if (camInfFpsElem) { camInfFpsElem.innerText = (obsRates.inference_fps !== null && obsRates.inference_fps !== undefined) ? obsRates.inference_fps.toFixed(1) : '12.0'; }
          } else {
            if (camStatusElem) { camStatusElem.innerText = 'CONFIGURED / NOT CONNECTED'; camStatusElem.style.color = '#fbbf24'; }
            if (camDevElem) { camDevElem.innerText = 'NOT DETECTED'; camDevElem.style.color = '#f87171'; }
            if (camStreamElem) { camStreamElem.innerText = 'INACTIVE'; camStreamElem.style.color = '#94a3b8'; }
            if (camCapFpsElem) { camCapFpsElem.innerText = 'N/A'; }
            if (camProcFpsElem) { camProcFpsElem.innerText = 'N/A'; }
            if (camInfFpsElem) { camInfFpsElem.innerText = 'N/A'; }
          }
        }
      } catch (e) {}
    }

    loadInitialEvents();
    connectWebSocket();
    setInterval(pollSystemStatus, 2500);
  </script>
</body>
</html>
"""
