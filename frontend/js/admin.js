const API_BASE = (window.API_BASE || window.location.origin || 'http://localhost:8000').replace(/\/$/, '');

const trainImage = document.getElementById('train-image');
const statusEl = document.getElementById('admin-status');
const logEl = document.getElementById('admin-log');
let ws = null;

function logLine(message) {
  if (!logEl) return;
  const line = document.createElement('div');
  line.textContent = message;
  logEl.prepend(line);
}

function ensureSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return ws;
  const wsBase = API_BASE.replace(/^http/, 'ws');
  ws = new WebSocket(`${wsBase}/ws/train-progress`);
  ws.onopen = () => logLine('Connected to training stream.');
  ws.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data);
      if (data.echo) return;
      if (data.type === 'training-start') {
        logLine('Training started.');
        statusEl.textContent = 'Training started — waiting for completion…';
      }
      if (data.type === 'training-progress') {
        const pct = data.progress ? Math.min(100, Math.round(data.progress * 100)) : 0;
        statusEl.textContent = `Training progress: ${pct}%`;
        logLine(`Progress: ${pct}% | CO₂: ${data.co2_g?.toFixed?.(4) || 0} g | farts: ${data.equivalent_farts?.toFixed?.(2) || 0}`);
      }
      if (data.type === 'training-complete') {
        statusEl.textContent = 'Training finished!';
        logLine(`Complete. Runtime ${data.runtime_seconds?.toFixed?.(1) || '?'}s; farts ${data.equivalent_farts?.toFixed?.(2) || 0}`);
      }
      if (data.type === 'training-error') {
        statusEl.textContent = `Error: ${data.message || 'unknown issue'}`;
        logLine(`Error: ${data.message || 'unknown issue'}`);
      }
    } catch (err) {
      console.error('Bad message', err);
    }
  };
  ws.onclose = () => {
    statusEl.textContent = 'WebSocket closed. Reconnecting…';
    setTimeout(ensureSocket, 1000);
  };
  ws.onerror = () => {
    statusEl.textContent = 'WebSocket error — retrying…';
  };
  return ws;
}

async function startTraining() {
  statusEl.textContent = 'Starting training…';
  ensureSocket();
  try {
    const res = await fetch(`${API_BASE}/start-training`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Unable to start training');
    }
    statusEl.textContent = 'Training started! Watching for progress…';
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
  }
}

if (trainImage) {
  trainImage.addEventListener('click', startTraining);
}

ensureSocket();
