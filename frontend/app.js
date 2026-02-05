const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const timerEl = document.getElementById('timer');
const resultEl = document.getElementById('result');
const meterFill = document.getElementById('meterFill');
const statusEl = document.getElementById('status');
const waveCanvas = document.getElementById('waveCanvas');
const waveCtx = waveCanvas.getContext('2d');

let audioContext = null;
let processor = null;
let analyser = null;
let source = null;
let mediaStream = null;
let chunks = [];
let startTime = null;
let timerId = null;
let recordedBlob = null;
let animationId = null;

function formatTime(ms) {
  const total = Math.floor(ms / 1000);
  const minutes = String(Math.floor(total / 60)).padStart(2, '0');
  const seconds = String(total % 60).padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function updateTimer() {
  if (!startTime) return;
  timerEl.textContent = formatTime(Date.now() - startTime);
}

function mergeBuffers(buffers) {
  let length = 0;
  buffers.forEach(buf => {
    length += buf.length;
  });
  const result = new Float32Array(length);
  let offset = 0;
  buffers.forEach(buf => {
    result.set(buf, offset);
    offset += buf.length;
  });
  return result;
}

function floatTo16BitPCM(output, offset, input) {
  for (let i = 0; i < input.length; i += 1) {
    let s = Math.max(-1, Math.min(1, input[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    output.setInt16(offset, s, true);
    offset += 2;
  }
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  floatTo16BitPCM(view, 44, samples);

  return new Blob([view], { type: 'audio/wav' });
}

function resizeCanvas() {
  const rect = waveCanvas.getBoundingClientRect();
  waveCanvas.width = Math.floor(rect.width * window.devicePixelRatio);
  waveCanvas.height = Math.floor(rect.height * window.devicePixelRatio);
}

function drawWaveform() {
  if (!analyser) return;
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteTimeDomainData(dataArray);

  const width = waveCanvas.width;
  const height = waveCanvas.height;
  waveCtx.clearRect(0, 0, width, height);

  waveCtx.lineWidth = 2 * window.devicePixelRatio;
  waveCtx.strokeStyle = '#31e8c6';
  waveCtx.beginPath();

  const sliceWidth = width / bufferLength;
  let x = 0;
  for (let i = 0; i < bufferLength; i += 1) {
    const v = dataArray[i] / 128.0;
    const y = (v * height) / 2;
    if (i === 0) {
      waveCtx.moveTo(x, y);
    } else {
      waveCtx.lineTo(x, y);
    }
    x += sliceWidth;
  }
  waveCtx.stroke();
  animationId = requestAnimationFrame(drawWaveform);
}

async function startRecording() {
  resultEl.textContent = 'Recording...';
  statusEl.textContent = 'Recording';
  recordedBlob = null;
  chunks = [];
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  source = audioContext.createMediaStreamSource(mediaStream);
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = event => {
    const input = event.inputBuffer.getChannelData(0);
    chunks.push(new Float32Array(input));

    let sum = 0;
    for (let i = 0; i < input.length; i += 1) {
      sum += input[i] * input[i];
    }
    const rms = Math.sqrt(sum / input.length);
    meterFill.style.width = `${Math.min(100, rms * 250)}%`;
  };

  source.connect(analyser);
  analyser.connect(processor);
  processor.connect(audioContext.destination);

  resizeCanvas();
  if (animationId) cancelAnimationFrame(animationId);
  drawWaveform();

  startTime = Date.now();
  timerEl.textContent = '00:00';
  timerId = setInterval(updateTimer, 200);
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  analyzeBtn.disabled = true;
}

function stopRecording() {
  if (processor) {
    processor.disconnect();
    processor.onaudioprocess = null;
  }
  if (analyser) analyser.disconnect();
  if (source) source.disconnect();
  if (audioContext) audioContext.close();
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }

  clearInterval(timerId);
  meterFill.style.width = '0%';
  statusEl.textContent = 'Idle';

  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }
  waveCtx.clearRect(0, 0, waveCanvas.width, waveCanvas.height);

  const samples = mergeBuffers(chunks);
  recordedBlob = encodeWAV(samples, audioContext.sampleRate);

  recordBtn.disabled = false;
  stopBtn.disabled = true;
  analyzeBtn.disabled = false;
  resultEl.textContent = 'Ready to analyze.';
}

async function sendForPrediction() {
  if (!recordedBlob) return;
  resultEl.textContent = 'Analyzing...';
  statusEl.textContent = 'Analyzing';
  const formData = new FormData();
  formData.append('file', recordedBlob, 'recording.wav');

  const response = await fetch('/api/predict', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  if (data.error) {
    resultEl.textContent = data.error;
    statusEl.textContent = 'Idle';
    return;
  }

  const status = data.cough_present ? 'Cough Present' : 'No Cough';
  resultEl.textContent = status;
  statusEl.textContent = 'Idle';
}

recordBtn.addEventListener('click', () => {
  startRecording().catch(err => {
    resultEl.textContent = `Mic error: ${err.message}`;
    statusEl.textContent = 'Idle';
  });
});

stopBtn.addEventListener('click', stopRecording);

analyzeBtn.addEventListener('click', () => {
  sendForPrediction().catch(err => {
    resultEl.textContent = `Error: ${err.message}`;
    statusEl.textContent = 'Idle';
  });
});

window.addEventListener('resize', resizeCanvas);
