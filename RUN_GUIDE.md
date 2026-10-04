# How to Run the PAASR Research Prototype

This project provides an interactive Telugu-to-English Speech-to-Speech Translation research prototype under the **PAASR (Phonology-Aware Adaptive Speech Representation)** framework.

---

## 1. Quick Start: Launching the Application

Activate your virtual environment and launch the Gradio web interface with:

```powershell
python app.py
```
*(Or `python demo_app.py`, which routes directly to `app.py`)*

Once started, open your web browser at:
```
http://127.0.0.1:7860
```

---

## 2. End-to-End Real Translation Pipeline

When you record or upload spoken Telugu audio, the application executes the live translation pipeline:

1. **Telugu Speech Input:** Accepts `.wav`, `.mp3`, or direct browser microphone recordings, normalized and resampled to 16 kHz mono.
2. **Telugu ASR:** Transcribes spoken Telugu directly into Telugu Unicode text using a fine-tuned Whisper model (`vasista22/whisper-telugu-tiny`).
3. **Telugu → English Translation:** Translates recognized Telugu text into English using `Helsinki-NLP/opus-mt-dra-en` (MarianMT Dravidian-to-English model).
4. **English Speech Synthesis (TTS):** Converts translated English text into natural playable English speech (`gTTS`).
5. **Real-time Diagnostics:** Displays input audio duration, estimated SNR, per-module execution latency, and step-by-step transcriptions.

---

## 3. Web Interface Features

- **🚀 Live S2ST Prototype:**
  - Record voice or upload custom audio.
  - Includes 3 one-click real Telugu speech examples (Agriculture, Healthcare, Education).
  - Outputs playable synthesized English speech, the Telugu transcript, the English translation, and latency statistics.
- **📝 Community Field Survey:**
  - Interactive 10-question questionnaire for rural & semi-urban field studies, logging non-identifying responses to `survey_responses.jsonl`.
- **📐 Mathematical Model & Novelty:**
  - Complete theoretical formulation of PAASR multi-stream decomposition, CARF dynamic gating, and PCL contrastive invariance.

---

## 4. Dependencies & Hugging Face Spaces Deployment

All necessary dependencies are listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

To deploy on **Hugging Face Spaces**:
1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `requirements.txt`, and the sample audio files (`sample_telugu.mp3`, etc.).
3. The Space will automatically build and start the web interface.
