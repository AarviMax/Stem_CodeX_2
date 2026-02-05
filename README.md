# MoodWave AI (Localhost)

A local web app that:
- Uses webcam face landmarks (MediaPipe FaceMesh).
- Estimates mood from **eye amplitude**, **eye wavelength**, and **mouth curvature**.
- Performs lightweight face recognition (user pattern match).
- Supports language selection: Hindi, English, Kannada, Bhojpuri, Malayalam, Telugu, Punjabi, Tamil.
- Recommends and plays mood-based songs.
- Includes **Start Scanning** and **Stop Scanning** controls.

## Run on localhost

```bash
python3 -m http.server 8000
```

Then open:

`http://localhost:8000`

> Allow webcam permission in the browser.
