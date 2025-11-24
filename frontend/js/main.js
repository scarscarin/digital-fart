const API_BASE =
  window.API_BASE_URL ||
  (location.hostname === 'localhost' && location.port === '8080'
    ? 'http://localhost:8000'
    : '');

const recordTrigger = document.getElementById('record-trigger');
const statusEl = document.getElementById('record-status');
const trainingBtn = document.getElementById('start-training-view');
const logBox = document.getElementById('training-log');
const normalFartAudio = document.getElementById('normal-fart-audio');

let mediaRecorder;
let chunks = [];
let recordingCountdown;

function updateStatus(message) {
  statusEl.textContent = message;
}

function startCountdown(seconds) {
  let remaining = seconds;
  updateStatus(`Recording… ${remaining}`);
  recordingCountdown = setInterval(() => {
    remaining -= 1;
    if (remaining <= 0) {
      updateStatus('Stopping…');
      clearInterval(recordingCountdown);
    } else {
      updateStatus(`Recording… ${remaining}`);
    }
  }, 1000);
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    updateStatus('Your browser does not support recording.');
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    mediaRecorder.onstop = handleRecordingStop;

    mediaRecorder.start();
    startCountdown(10);

    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 10000);
  } catch (err) {
    console.error(err);
    updateStatus('Microphone permission denied or unavailable.');
  }
}

async function handleRecordingStop() {
  clearInterval(recordingCountdown);
  updateStatus('Preparing upload…');

  const blob = new Blob(chunks, { type: 'audio/webm' });
  const formData = new FormData();
  formData.append('file', blob, `fart-${Date.now()}.webm`);

  updateStatus('Uploading…');
  try {
    const resp = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      throw new Error(`Upload failed: ${resp.status} ${errText}`);
    }
    await resp.json();
    updateStatus('Thanks, your fart has been archived.');

    const audio = document.createElement('audio');
    audio.src = URL.createObjectURL(blob);
    audio.play().catch(() => {});
  } catch (err) {
    console.error(err);
    updateStatus('Upload failed. Please try again.');
  }
}

function appendLog(text, bold = false) {
  const div = document.createElement('div');
  div.textContent = text;
  if (bold) div.style.fontWeight = 'bold';
  logBox.appendChild(div);
  logBox.scrollTop = logBox.scrollHeight;
}

function connectTrainingStream() {
  const evtSource = new EventSource(`${API_BASE}/api/training/stream`);
  appendLog('Connected to training stream…');

  evtSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === 'log') {
        appendLog(payload.message);
      } else if (payload.type === 'done') {
        appendLog('Training complete. Playing the new normal fart…', true);
        if (payload.url) {
          normalFartAudio.src = payload.url;
          normalFartAudio.play().catch(() => {});
        }
      }
    } catch (err) {
      console.error('Bad SSE payload', err, event.data);
    }
  };

  evtSource.onerror = (err) => {
    console.error('SSE connection error', err);
  };
}

recordTrigger?.addEventListener('click', startRecording);
trainingBtn?.addEventListener('click', connectTrainingStream);
