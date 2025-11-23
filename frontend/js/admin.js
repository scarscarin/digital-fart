const statusBox = document.getElementById('admin-status');
const trainBtn = document.getElementById('train-now-btn');

function setStatus(text) {
  statusBox.textContent = text;
}

async function loadStatus() {
  try {
    const resp = await fetch('/api/training/status');
    if (!resp.ok) throw new Error('status failed');
    const data = await resp.json();
    const runningText = data.running ? 'Training running' : 'Idle';
    const outputText = data.output_url ? `Last output: ${data.output_url}` : 'No output yet';
    setStatus(`${runningText}. ${outputText}`);
  } catch (err) {
    console.error(err);
    setStatus('Could not load status.');
  }
}

async function startTraining() {
  setStatus('Starting training…');
  try {
    const resp = await fetch('/api/admin/train', { method: 'POST' });
    if (resp.status === 409) {
      setStatus('Training already running.');
      return;
    }
    if (!resp.ok) throw new Error('start failed');
    const data = await resp.json();
    setStatus(`Training started (ID: ${data.training_id}). Open the landing page training section to see logs.`);
  } catch (err) {
    console.error(err);
    setStatus('Failed to start training.');
  }
}

trainBtn?.addEventListener('click', startTraining);
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  setInterval(loadStatus, 5000);
});
