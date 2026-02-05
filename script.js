const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const startBtn = document.getElementById('startScanBtn');
const stopBtn = document.getElementById('stopScanBtn');
const playSongBtn = document.getElementById('playSongBtn');
const languageSelect = document.getElementById('languageSelect');
const audioPlayer = document.getElementById('audioPlayer');

const eyeAmplitudeEl = document.getElementById('eyeAmplitude');
const eyeWavelengthEl = document.getElementById('eyeWavelength');
const mouthCurvatureEl = document.getElementById('mouthCurvature');
const detectedMoodEl = document.getElementById('detectedMood');
const faceIdentityEl = document.getElementById('faceIdentity');
const moodAnalysisEl = document.getElementById('moodAnalysis');
const nowPlayingEl = document.getElementById('nowPlaying');
const scanStatus = document.getElementById('scanStatus');

let faceMesh;
let camera;
let isScanning = false;
let currentMood = 'neutral';

let baselineEmbedding = null;
let sampleWindow = []; // for eye amplitude waves

const moods = ['happy', 'sad', 'angry', 'surprised', 'neutral', 'sleepy', 'excited', 'stressed'];
const languages = ['english', 'hindi', 'kannada', 'bhojpuri', 'malayalam', 'telugu', 'punjabi', 'tamil'];

const moodDescriptions = {
  happy: 'You look cheerful and balanced. Keep this energy flowing!',
  sad: 'Low facial energy detected. A gentle uplifting playlist may help.',
  angry: 'High tension signs detected around eyes and mouth. Try calming tracks.',
  surprised: 'Wide eyes and raised mouth response suggest surprise/arousal.',
  neutral: 'You seem calm and stable right now.',
  sleepy: 'Lower eye amplitude and slower variation indicate drowsiness.',
  excited: 'Rapid eye activity and positive mouth curvature suggest excitement.',
  stressed: 'Mixed high-frequency eye pattern with tight mouth profile suggests stress.'
};

const songDb = buildSongDb();

startBtn.addEventListener('click', startScanning);
stopBtn.addEventListener('click', stopScanning);
playSongBtn.addEventListener('click', playMoodSong);

async function initFaceMesh() {
  if (faceMesh) return;
  faceMesh = new FaceMesh({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
  });

  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  faceMesh.onResults(onFaceResults);
}

async function startScanning() {
  await initFaceMesh();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    webcam.srcObject = stream;

    camera = new Camera(webcam, {
      onFrame: async () => {
        if (!isScanning) return;
        await faceMesh.send({ image: webcam });
      },
      width: 960,
      height: 540,
    });

    isScanning = true;
    sampleWindow = [];
    scanStatus.textContent = 'Scanning...';
    startBtn.disabled = true;
    stopBtn.disabled = false;

    await camera.start();
  } catch (err) {
    scanStatus.textContent = 'Camera access denied/unavailable';
    console.error(err);
  }
}

function stopScanning() {
  isScanning = false;

  const stream = webcam.srcObject;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    webcam.srcObject = null;
  }

  if (camera && typeof camera.stop === 'function') {
    camera.stop();
  }

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  scanStatus.textContent = 'Stopped';
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

function onFaceResults(results) {
  overlay.width = webcam.videoWidth || 960;
  overlay.height = webcam.videoHeight || 540;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.multiFaceLandmarks?.length) {
    faceIdentityEl.textContent = 'Face not found';
    detectedMoodEl.textContent = 'Unknown';
    return;
  }

  const lm = results.multiFaceLandmarks[0];
  drawContour(lm);

  const eyeAmplitude = computeEyeAmplitude(lm);
  const eyeWavelength = computeEyeWavelength(eyeAmplitude);
  const mouthCurvature = computeMouthCurvature(lm);

  eyeAmplitudeEl.textContent = eyeAmplitude.toFixed(4);
  eyeWavelengthEl.textContent = eyeWavelength.toFixed(2);
  mouthCurvatureEl.textContent = mouthCurvature.toFixed(4);

  identifyFace(lm);
  currentMood = classifyMood({ eyeAmplitude, eyeWavelength, mouthCurvature });

  detectedMoodEl.textContent = currentMood;
  moodAnalysisEl.textContent = moodDescriptions[currentMood];
}

function drawContour(landmarks) {
  ctx.strokeStyle = '#7a8cff';
  ctx.lineWidth = 1;
  for (const p of landmarks) {
    const x = p.x * overlay.width;
    const y = p.y * overlay.height;
    ctx.beginPath();
    ctx.arc(x, y, 1.2, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeEyeAmplitude(lm) {
  const leftV = dist(lm[159], lm[145]);
  const leftH = dist(lm[33], lm[133]);
  const rightV = dist(lm[386], lm[374]);
  const rightH = dist(lm[362], lm[263]);

  const leftEAR = leftV / leftH;
  const rightEAR = rightV / rightH;
  return (leftEAR + rightEAR) / 2;
}

function computeEyeWavelength(eyeAmplitude) {
  const ts = performance.now();
  sampleWindow.push({ value: eyeAmplitude, ts });
  if (sampleWindow.length > 80) sampleWindow.shift();

  if (sampleWindow.length < 20) return 0;

  const avg = sampleWindow.reduce((s, x) => s + x.value, 0) / sampleWindow.length;
  let crossings = 0;

  for (let i = 1; i < sampleWindow.length; i++) {
    const a = sampleWindow[i - 1].value - avg;
    const b = sampleWindow[i].value - avg;
    if ((a < 0 && b >= 0) || (a > 0 && b <= 0)) crossings++;
  }

  const durationSec = (sampleWindow[sampleWindow.length - 1].ts - sampleWindow[0].ts) / 1000;
  if (durationSec <= 0 || crossings < 2) return 0;

  const frequency = crossings / (2 * durationSec); // Hz
  return frequency;
}

function computeMouthCurvature(lm) {
  const leftCorner = lm[61];
  const rightCorner = lm[291];
  const upperLip = lm[13];

  const cornerAvgY = (leftCorner.y + rightCorner.y) / 2;
  return upperLip.y - cornerAvgY;
}

function identifyFace(lm) {
  const embedding = [
    dist(lm[33], lm[263]),
    dist(lm[1], lm[152]),
    dist(lm[61], lm[291]),
    dist(lm[10], lm[152]),
    dist(lm[133], lm[362])
  ];

  if (!baselineEmbedding) {
    baselineEmbedding = embedding;
    faceIdentityEl.textContent = 'User #1 (registered)';
    return;
  }

  const similarity = cosineSimilarity(baselineEmbedding, embedding);
  faceIdentityEl.textContent = similarity > 0.995 ? 'User #1 recognized' : 'Unknown face pattern';
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function classifyMood({ eyeAmplitude, eyeWavelength, mouthCurvature }) {
  if (eyeAmplitude < 0.20 && eyeWavelength < 0.5) return 'sleepy';
  if (eyeAmplitude > 0.33 && mouthCurvature < -0.015 && eyeWavelength > 1.2) return 'excited';
  if (eyeAmplitude > 0.36 && eyeWavelength > 1.8) return 'surprised';
  if (mouthCurvature < -0.012 && eyeWavelength > 1.1) return 'happy';
  if (mouthCurvature > 0.003 && eyeAmplitude < 0.25) return 'sad';
  if (eyeWavelength > 2.2 && mouthCurvature > 0.005) return 'angry';
  if (eyeWavelength > 1.6 && mouthCurvature > 0.001) return 'stressed';
  return 'neutral';
}

function playMoodSong() {
  const language = languageSelect.value;
  const langSongs = songDb[language];
  if (!langSongs) return;

  const song = langSongs[currentMood] || langSongs.neutral;
  audioPlayer.src = song.url;
  nowPlayingEl.textContent = `Now Playing: ${song.title} (${language.toUpperCase()}) for mood: ${currentMood}`;
  audioPlayer.play().catch((err) => {
    console.warn('Autoplay prevented. User interaction required.', err);
  });
}

function buildSongDb() {
  const moodTitles = {
    happy: 'Happy Lift',
    sad: 'Soft Comfort',
    angry: 'Calm Down Flow',
    surprised: 'Energy Spark',
    neutral: 'Balanced Vibes',
    sleepy: 'Wake Up Pulse',
    excited: 'High Voltage Beat',
    stressed: 'Stress Relief Melody'
  };

  const sampleUrls = {
    happy: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
    sad: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3',
    angry: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3',
    surprised: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3',
    neutral: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3',
    sleepy: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3',
    excited: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3',
    stressed: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3',
  };

  const db = {};
  for (const lang of languages) {
    db[lang] = {};
    for (const mood of moods) {
      db[lang][mood] = {
        title: `${capitalize(lang)} ${moodTitles[mood]}`,
        url: sampleUrls[mood]
      };
    }
  }
  return db;
}

function capitalize(word) {
  return word[0].toUpperCase() + word.slice(1);
}
