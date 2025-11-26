const API_BASE = window.API_BASE || 'http://localhost:8000';

const trainImage = document.getElementById('train-image');
const statusEl = document.getElementById('admin-status');

async function startTraining() {
  statusEl.textContent = 'Starting training…';
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
