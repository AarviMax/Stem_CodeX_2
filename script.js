const webcam = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

const startBtn = document.getElementById('startScanBtn');
const stopBtn = document.getElementById('stopScanBtn');
const playSongBtn = document.getElementById('playSongBtn');
const languageSelect = document.getElementById('languageSelect');

const eyeAmplitudeEl = document.getElementById('eyeAmplitude');
const eyeWavelengthEl = document.getElementById('eyeWavelength');
const mouthCurvatureEl = document.getElementById('mouthCurvature');
const detectedMoodEl = document.getElementById('detectedMood');
const faceIdentityEl = document.getElementById('faceIdentity');
const moodAnalysisEl = document.getElementById('moodAnalysis');
const nowPlayingEl = document.getElementById('nowPlaying');
const scanStatus = document.getElementById('scanStatus');
const allMoodsEl = document.getElementById('allMoods');
const songEmbed = document.getElementById('songEmbed');

let faceMesh;
let camera;
let isScanning = false;
let currentMood = 'neutral';
let sampleWindow = [];
let baselineEmbedding = null;

const moods = [
  'happy', 'sad', 'angry', 'surprised', 'neutral', 'sleepy', 'excited', 'stressed',
  'calm', 'confused', 'fearful', 'disgusted', 'bored',
  'depressed', 'anxious', 'frustrated', 'lonely', 'guilty'
  'calm', 'confused', 'fearful', 'disgusted', 'bored'
];
const languages = ['english', 'hindi', 'kannada', 'bhojpuri', 'malayalam', 'telugu', 'punjabi', 'tamil'];

const moodDescriptions = {
  happy: 'AI model sees positive mouth curvature and active eyes: cheerful state.',
  sad: 'AI model sees low facial activation and downturned lips: low-energy state.',
  angry: 'High-frequency eye pattern with tight mouth indicates agitation/tension.',
  surprised: 'High amplitude and rapid motion indicate surprise.',
  neutral: 'Balanced movement and curvature suggest neutral mood.',
  sleepy: 'Low amplitude and slower temporal variation indicate drowsiness.',
  excited: 'Strong activity + smiling profile indicate excitement.',
  stressed: 'Mixed jitter with compressed mouth profile indicates stress.',
  calm: 'Stable, low-variance movements suggest calmness.',
  confused: 'Uneven activity with uncertain mouth profile suggests confusion.',
  fearful: 'High alert-eye pattern with constrained smile suggests fearfulness.',
  disgusted: 'Tight upper-lip profile with irregular eye response suggests disgust.',
  bored: 'Low movement + flat curvature indicates boredom.',
  depressed: 'Very low facial-energy profile suggests a depressed affect state.',
  anxious: 'High micro-variability in eye signal indicates anxious state.',
  frustrated: 'Tension-heavy profile with uneven mouth movement indicates frustration.',
  lonely: 'Muted expression with low engagement suggests loneliness.',
  guilty: 'Constrained smile + unstable eye rhythm indicates possible guilt-like mood.'
};

  bored: 'Low movement + flat curvature indicates boredom.'
};

// AI model (small neural net) trained on synthetic feature prototypes for mood classes.
const moodModel = trainMoodModel();
const songDb = buildEmbeddedSongDb();
allMoodsEl.textContent = moods.join(', ');

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
      width: 640,
      height: 480,
    });

    isScanning = true;
    sampleWindow = [];
    scanStatus.textContent = 'Scanning...';
    startBtn.disabled = true;
    stopBtn.disabled = false;

    await camera.start();
  } catch (error) {
    scanStatus.textContent = 'Camera blocked/unavailable';
    console.error(error);
  }
}

function stopScanning() {
  isScanning = false;
  const stream = webcam.srcObject;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    webcam.srcObject = null;
  }

  if (camera && typeof camera.stop === 'function') camera.stop();

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  scanStatus.textContent = 'Stopped';
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

function onFaceResults(results) {
  overlay.width = webcam.videoWidth || 640;
  overlay.height = webcam.videoHeight || 480;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!results.multiFaceLandmarks?.length) {
    faceIdentityEl.textContent = 'Face not found';
    detectedMoodEl.textContent = 'Unknown';
    return;
  }

  const lm = results.multiFaceLandmarks[0];
  drawLandmarks(lm);

  const eyeAmplitude = computeEyeAmplitude(lm);
  const eyeWavelength = computeEyeWavelength(eyeAmplitude);
  const mouthCurvature = computeMouthCurvature(lm);

  eyeAmplitudeEl.textContent = eyeAmplitude.toFixed(4);
  eyeWavelengthEl.textContent = eyeWavelength.toFixed(2);
  mouthCurvatureEl.textContent = mouthCurvature.toFixed(4);

  identifyFace(lm);
  currentMood = classifyMoodAI({ eyeAmplitude, eyeWavelength, mouthCurvature });
  detectedMoodEl.textContent = currentMood;
  moodAnalysisEl.textContent = moodDescriptions[currentMood] || moodDescriptions.neutral;
}

function drawLandmarks(landmarks) {
  ctx.strokeStyle = 'rgba(111, 212, 255, .85)';
  ctx.lineWidth = 1;
  for (const p of landmarks) {
    const x = p.x * overlay.width;
    const y = p.y * overlay.height;
    ctx.beginPath();
    ctx.arc(x, y, 1.1, 0, Math.PI * 2);
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
  return ((leftV / leftH) + (rightV / rightH)) / 2;
}

function computeEyeWavelength(eyeAmplitude) {
  const ts = performance.now();
  sampleWindow.push({ value: eyeAmplitude, ts });
  if (sampleWindow.length > 90) sampleWindow.shift();
  if (sampleWindow.length < 22) return 0;

  const mean = sampleWindow.reduce((s, v) => s + v.value, 0) / sampleWindow.length;
  let crossings = 0;
  for (let i = 1; i < sampleWindow.length; i++) {
    const a = sampleWindow[i - 1].value - mean;
    const b = sampleWindow[i].value - mean;
    if ((a < 0 && b >= 0) || (a > 0 && b <= 0)) crossings += 1;
  }

  const durationSec = (sampleWindow[sampleWindow.length - 1].ts - sampleWindow[0].ts) / 1000;
  if (durationSec <= 0 || crossings < 2) return 0;
  return crossings / (2 * durationSec);
}

function computeMouthCurvature(lm) {
  const left = lm[61];
  const right = lm[291];
  const upper = lm[13];
  const cornerMeanY = (left.y + right.y) / 2;
  return upper.y - cornerMeanY;
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

  const sim = cosineSimilarity(baselineEmbedding, embedding);
  faceIdentityEl.textContent = sim > 0.995 ? 'User #1 recognized' : 'Unknown face pattern';
}

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function normalizeFeatures(features) {
  return {
    eyeAmplitude: clamp((features.eyeAmplitude - 0.12) / 0.32, 0, 1),
    eyeWavelength: clamp(features.eyeWavelength / 3.0, 0, 1),
    mouthCurvature: clamp((features.mouthCurvature + 0.04) / 0.08, 0, 1)
  };
}

function classifyMoodAI(features) {
  const input = normalizeFeatures(features);
  const output = moodModel.run(input);
  const [mood] = Object.entries(output).sort((a, b) => b[1] - a[1])[0] || ['neutral', 1];
  return moods.includes(mood) ? mood : 'neutral';
}

function trainMoodModel() {
  const net = new brain.NeuralNetwork({ hiddenLayers: [12, 10], activation: 'relu' });

  const net = new brain.NeuralNetwork({ hiddenLayers: [10, 10], activation: 'relu' });

  // Synthetic prototypes to emulate a small AI mood model.
  const prototypes = {
    happy: [0.62, 0.45, 0.12],
    sad: [0.32, 0.25, 0.65],
    angry: [0.75, 0.88, 0.72],
    surprised: [0.92, 0.72, 0.38],
    neutral: [0.5, 0.4, 0.5],
    sleepy: [0.18, 0.12, 0.52],
    excited: [0.88, 0.8, 0.25],
    stressed: [0.68, 0.76, 0.62],
    calm: [0.4, 0.2, 0.45],
    confused: [0.55, 0.55, 0.57],
    fearful: [0.8, 0.84, 0.58],
    disgusted: [0.48, 0.63, 0.75],
    bored: [0.24, 0.18, 0.49],
    depressed: [0.18, 0.16, 0.78],
    anxious: [0.7, 0.93, 0.6],
    frustrated: [0.64, 0.69, 0.68],
    lonely: [0.22, 0.22, 0.62],
    guilty: [0.34, 0.58, 0.64]
    bored: [0.24, 0.18, 0.49]
  };

  const trainingData = [];
  Object.entries(prototypes).forEach(([mood, p]) => {
    for (let i = 0; i < 20; i++) {
    for (let i = 0; i < 18; i++) {
      const input = {
        eyeAmplitude: clamp(p[0] + jitter(0.06), 0, 1),
        eyeWavelength: clamp(p[1] + jitter(0.08), 0, 1),
        mouthCurvature: clamp(p[2] + jitter(0.07), 0, 1)
      };
      const output = Object.fromEntries(moods.map((m) => [m, m === mood ? 1 : 0]));
      trainingData.push({ input, output });
    }
  });

  net.train(trainingData, {
    iterations: 1400,
    iterations: 1200,
    log: false,
    errorThresh: 0.012,
    learningRate: 0.03
  });

  return net;
}

function playMoodSong() {
  const language = languageSelect.value;
  const byLang = songDb[language] || songDb.english;
  const song = byLang[currentMood] || byLang.neutral;

  songEmbed.src = `https://www.youtube.com/embed/${song.videoId}`;
  nowPlayingEl.textContent = `Now showing: ${song.title} (${language}) for mood: ${currentMood}`;
}

function buildEmbeddedSongDb() {
  const languageDefaults = {
    english: 'dQw4w9WgXcQ',
    hindi: 'JGwWNGJdvx8',
    kannada: '6vKucgAeF_Q',
    bhojpuri: 'wWlYU0P2b4E',
    malayalam: 'hQ9Q1K0Gx8I',
    telugu: 'kJQP7kiw5Fk',
    punjabi: 'i3m8U3jjo0o',
    tamil: 'YQHsXMglC9A'
  };

  const languageMoodVideos = {
    english: {
      happy: 'ZbZSe6N_BXs', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: 'kJQP7kiw5Fk', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: '3AtDnEC4zak', depressed: 'RgKAFK5djSk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'CevxZvSJLk8', lonely: 'YykjpeuMNEk', guilty: 'nfs8NYg7yQM'
    },
    hindi: {
      happy: 'JGwWNGJdvx8', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: '3AtDnEC4zak', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: 'ZbZSe6N_BXs', depressed: 'YykjpeuMNEk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'RgKAFK5djSk', guilty: 'CevxZvSJLk8'
    },
    kannada: {
      happy: '6vKucgAeF_Q', sad: 'YykjpeuMNEk', angry: 'nfs8NYg7yQM', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: '3AtDnEC4zak', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: 'ZbZSe6N_BXs', depressed: 'ho9rZjlsyYY', anxious: 'hLQl3WQQoQ0',
      frustrated: 'CevxZvSJLk8', lonely: 'RgKAFK5djSk', guilty: '2OEL4P1Rz04'
    },
    bhojpuri: {
      happy: 'wWlYU0P2b4E', sad: 'YykjpeuMNEk', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: 'kJQP7kiw5Fk', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: '3AtDnEC4zak', depressed: 'RgKAFK5djSk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'ho9rZjlsyYY', guilty: 'CevxZvSJLk8'
    },
    malayalam: {
      happy: 'hQ9Q1K0Gx8I', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: 'kJQP7kiw5Fk', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: '3AtDnEC4zak', depressed: 'YykjpeuMNEk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'RgKAFK5djSk', guilty: 'CevxZvSJLk8'
    },
    telugu: {
      happy: 'kJQP7kiw5Fk', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: '3AtDnEC4zak', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: 'ZbZSe6N_BXs', depressed: 'YykjpeuMNEk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'RgKAFK5djSk', guilty: 'CevxZvSJLk8'
    },
    punjabi: {
      happy: 'i3m8U3jjo0o', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: 'kJQP7kiw5Fk', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: '3AtDnEC4zak', depressed: 'YykjpeuMNEk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'RgKAFK5djSk', guilty: 'CevxZvSJLk8'
    },
    tamil: {
      happy: 'YQHsXMglC9A', sad: 'ho9rZjlsyYY', angry: '2OEL4P1Rz04', surprised: 'fLexgOxsZu0', neutral: 'jfKfPfyJRdk',
      sleepy: '09R8_2nJtjg', excited: 'kJQP7kiw5Fk', stressed: '1ZYbU82GVz4', calm: 'UfcAVejslrU', confused: '5qap5aO4i9A',
      fearful: 'sTANio_2E0Q', disgusted: 'DWcJFNfaw9c', bored: '3AtDnEC4zak', depressed: 'YykjpeuMNEk', anxious: 'hLQl3WQQoQ0',
      frustrated: 'nfs8NYg7yQM', lonely: 'RgKAFK5djSk', guilty: 'CevxZvSJLk8'
    }
  const base = {
    happy: { title: 'Mood Lift Track', videoId: 'ZbZSe6N_BXs' },
    sad: { title: 'Comfort Piano', videoId: 'ho9rZjlsyYY' },
    angry: { title: 'Calm Ambient', videoId: '2OEL4P1Rz04' },
    surprised: { title: 'High Energy', videoId: 'fLexgOxsZu0' },
    neutral: { title: 'Lo-fi Focus', videoId: 'jfKfPfyJRdk' },
    sleepy: { title: 'Wake Up Beat', videoId: '09R8_2nJtjg' },
    excited: { title: 'Party Pulse', videoId: 'kJQP7kiw5Fk' },
    stressed: { title: 'Stress Relief', videoId: '1ZYbU82GVz4' },
    calm: { title: 'Deep Calm', videoId: 'UfcAVejslrU' },
    confused: { title: 'Think Mode', videoId: '5qap5aO4i9A' },
    fearful: { title: 'Grounding Sound', videoId: 'sTANio_2E0Q' },
    disgusted: { title: 'Reset Mood', videoId: 'DWcJFNfaw9c' },
    bored: { title: 'Fresh Vibes', videoId: '3AtDnEC4zak' }
  };

  const db = {};
  for (const lang of languages) {
    db[lang] = {};
    for (const mood of moods) {
      const perLang = languageMoodVideos[lang] || {};
      const videoId = perLang[mood] || perLang.neutral || languageDefaults[lang] || languageDefaults.english;
      db[lang][mood] = {
        title: `${capitalize(lang)} ${mood} recommendation`,
        videoId
      const item = base[mood] || base.neutral;
      db[lang][mood] = {
        title: `${capitalize(lang)} ${item.title}`,
        videoId: item.videoId
      };
    }
  }
  return db;
}

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

function jitter(scale) {
  return (Math.random() * 2 - 1) * scale;
}

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}
