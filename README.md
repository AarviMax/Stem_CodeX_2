# MoodWave AI (Localhost)

A local web app that:
- Uses webcam face landmarks (MediaPipe FaceMesh).
- Uses an in-browser **AI mood model** (small neural network via Brain.js) to classify mood from:
  - eye amplitude
  - eye wavelength
  - mouth curvature
- Performs lightweight face recognition (user pattern match).
- Supports language selection: Hindi, English, Kannada, Bhojpuri, Malayalam, Telugu, Punjabi, Tamil.
- Recommends songs using **embedded YouTube links** (iframe), not raw MP3 URLs, with language-specific mappings.
- Includes **Start Scanning** and **Stop Scanning** controls.
- Supports both positive and negative moods (e.g., stressed, depressed, anxious, lonely, frustrated, guilty).

## Run on localhost

```bash
python3 -m http.server 8000
```

Open:

`http://localhost:8000`

## Kill localhost server

If you launched the server in background:

```bash
pkill -f "python3 -m http.server 8000"
```

Or kill by port:

```bash
kill -9 $(lsof -ti:8000)
```

> Allow webcam permission in the browser.
