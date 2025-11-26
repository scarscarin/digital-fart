const API_BASE = (window.API_BASE || window.location.origin || 'http://localhost:8000').replace(/\/$/, '');
const RECORD_SECONDS = 5;

let recorder = null;
let stream = null;
let isRecording = false;
let allowAutoPlay = false;
let ws = null;

const statusEl = document.getElementById('status');
const staticFrame = document.querySelector('.static-frame');
const video = document.getElementById('myVideo');
const recImg = document.getElementById('recImg');
const co2Counter = document.getElementById('co2-counter');
const trainingVideo = document.getElementById('training-video');
const trainButton = document.getElementById('train-button');
const generatedAudio = document.getElementById('generated-fart');
const playGenerated = document.getElementById('play-generated');

function setStatus(text) {
  if (statusEl) {
    statusEl.textContent = text;
  }
}

function updateCounter(farts) {
  if (co2Counter) {
    co2Counter.textContent = `CO₂ emitted: ${farts.toFixed(2)} farts equivalent`;
  }
}

async function handleClick() {
  if (isRecording) return;

  allowAutoPlay = window.confirm('Allow the website to automatically play the generated fart when training is finished?');

  try {
    setStatus('could u give me permission to hear your fart?');
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    setStatus('GO! AS FARTY AS YOU CAN!');
    startVideoAnimation();
    startRecording(stream);
    setTimeout(stopRecordingAndUpload, RECORD_SECONDS * 1000);
  } catch (err) {
    console.error(err);
    setStatus('Microphone access denied or unavailable. Maybe do a dance, instead?');
  }
}

function startVideoAnimation() {
  if (!staticFrame || !video) return;
  staticFrame.style.display = 'none';
  video.style.display = 'inline-block';
  video.currentTime = 0;
  video.play();

  video.onended = () => {
    if (isRecording) {
      video.currentTime = 0;
      video.play();
    } else {
      video.style.display = 'none';
      staticFrame.style.display = 'inline-block';
    }
  };
}

function startRecording(mediaStream) {
  isRecording = true;
  recorder = RecordRTC(mediaStream, {
    type: 'audio',
    mimeType: 'audio/webm',
    recorderType: RecordRTC.StereoAudioRecorder,
    numberOfAudioChannels: 1,
    desiredSampRate: 48000,
  });
  recorder.startRecording();
}

function stopRecordingAndUpload() {
  if (!recorder || !isRecording) return;

  setStatus('Wow, what a *prrr*. Im not closing the microphone...');

  recorder.stopRecording(async () => {
    isRecording = false;
    if (video) {
      video.pause();
      video.style.display = 'none';
    }
    if (staticFrame) {
      staticFrame.style.display = 'inline-block';
    }

    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }

    const blob = recorder.getBlob();
    recorder = null;

    setStatus('Uploading to the gassy cloud... 💾');

    try {
      const formData = new FormData();
      formData.append('file', blob, 'fart.webm');
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      if (response.ok && result.success) {
        setStatus(`Thank you! Total farts in the vault: ${result.count}`);
      } else {
        throw new Error(result.detail || 'Upload failed.');
      }
    } catch (err) {
      console.error(err);
      setStatus('Upload error.');
    }
  });
}

function ensureSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return ws;
  const wsBase = API_BASE.replace(/^http/, 'ws');
  ws = new WebSocket(`${wsBase}/ws/train-progress`);
  ws.onopen = () => {
    ws.send('hello');
    setStatus('Connected to training stream.');
  };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.echo) return;
      handleEvent(data);
    } catch (err) {
      console.error('Bad websocket message', err);
    }
  };
  ws.onclose = () => {
    setStatus('Training stream closed. Reconnecting…');
    setTimeout(ensureSocket, 1000);
  };
  ws.onerror = () => {
    setStatus('Training stream error — retrying…');
  };
  return ws;
}

function handleEvent(event) {
  if (event.type === 'training-start') {
    updateCounter(0);
    const duration = event.target_seconds ? Math.round(event.target_seconds / 60) : 10;
    setStatus(`Training started — running for ${duration} minutes. Keep sniffing the counter.`);
    if (trainingVideo) {
      trainingVideo.src = event.video;
      trainingVideo.style.display = 'block';
      trainingVideo.play().catch(() => {});
    }
  }

  if (event.type === 'training-progress') {
    const farts = event.equivalent_farts || 0;
    updateCounter(farts);
    if (typeof event.progress === 'number') {
      const pct = Math.min(100, Math.max(0, Math.round(event.progress * 100)));
      setStatus(`Training in progress: ${pct}%`);
    }
  }

  if (event.type === 'training-complete') {
    const farts = event.equivalent_farts || 0;
    updateCounter(farts);
    setStatus('Training finished!');
    const audioUrl = event.audio ? (event.audio.startsWith('http') ? event.audio : `${API_BASE}${event.audio}`) : '';
    if (audioUrl) {
      generatedAudio.src = audioUrl;
      generatedAudio.load();
      if (allowAutoPlay) {
        generatedAudio.play().catch(() => {});
      } else {
        playGenerated.style.display = 'inline-block';
      }
    }
  }

  if (event.type === 'training-error') {
    setStatus(`Training error: ${event.message || 'unknown issue'}`);
  }
}

async function startTrainingFromClient() {
  ensureSocket();
  updateCounter(0);
  setStatus('Requesting training…');
  try {
    const res = await fetch(`${API_BASE}/start-training`, { method: 'POST' });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Unable to start training');
    }
    setStatus('Training kicked off. Watch the counter.');
  } catch (err) {
    console.error(err);
    setStatus(`Training failed: ${err.message}`);
  }
}

if (recImg) {
  recImg.addEventListener('click', handleClick);
}

if (trainButton) {
  trainButton.addEventListener('click', startTrainingFromClient);
}

if (playGenerated) {
  playGenerated.addEventListener('click', () => {
    generatedAudio.play();
  });
}

ensureSocket();
