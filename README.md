# 🛡️ NSFW Content Monitor

Monitor your screen in real time to detect and alert on NSFW content using a pre-trained MobileNet model.

## 📌 About the Project

This project uses a combination of:

- 🧠 A MobileNetV2 model trained to classify NSFW content
- 🖥️ Screen capture using the `mss` Python library
- 📊 Live visual feedback using OpenCV
- 🔊 Audio alert when inappropriate content is detected

It aims to provide **a lightweight, free, and private parental control tool** powered by open-source machine learning.

---

## 🚀 Features

- ⚡ Real-time NSFW content detection
- 🪪 Categories: `drawings`, `hentai`, `neutral`, `porn`, `sexy`
- 📉 Adjustable sensitivity threshold
- 📋 Live overlay window with category scores
- 🔔 Audio alert system for sensitive content
- 🔓 Fully local and privacy-safe (no data leaves your machine)

---

## 🛠️ How It Works

1. Captures the entire screen periodically (every 0.5s)
2. Runs each frame through a MobileNetV2 model trained to detect NSFW content
3. Shows category confidence scores in a floating info window
4. Triggers a sound alert if the sum of NSFW category scores exceeds a defined threshold

---

## 🧰 Requirements

- Python 3.7+
- Packages:
  - `tensorflow`
  - `tensorflow-hub`
  - `opencv-python`
  - `numpy`
  - `mss`
  - `pywin32` (for Windows users)

Install them with:

```bash
pip install -r requirements.txt
