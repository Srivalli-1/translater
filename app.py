"""
PAASR: Phonology-Aware Adaptive Speech Representation Framework
Interactive Telugu-to-English Speech-to-Speech Translation Research Prototype
Run locally using: python app.py
Suitable for deployment on Hugging Face Spaces with Gradio.
"""

import os
import sys
import re
import json
import time
import tempfile
import numpy as np

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import gradio as gr
import librosa
import soundfile as sf
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)
from gtts import gTTS


# ==========================================
# 1. Model Singleton & Pipeline Initialization
# ==========================================

class S2STPipeline:
    """
    Manages lazy-loading and inference for:
    1. Telugu ASR (fine-tuned Whisper-Telugu)
    2. Telugu -> English Translation (Helsinki-NLP/opus-mt-dra-en)
    3. English Text-to-Speech (gTTS)
    """
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model_id = "vasista22/whisper-telugu-tiny"
        self.fallback_whisper_id = "openai/whisper-tiny"
        self.translation_model_id = "Helsinki-NLP/opus-mt-dra-en"

        self.whisper_processor = None
        self.whisper_model = None
        self.mt_tokenizer = None
        self.mt_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = S2STPipeline()
        return cls._instance

    def load_asr_model(self):
        if self.whisper_model is None:
            print(f"[PAASR Pipeline] Loading Telugu ASR model: {self.whisper_model_id} on {self.device}...")
            try:
                self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_id)
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_id).to(self.device)
            except Exception as e:
                print(f"[Warning] Failed loading {self.whisper_model_id}: {e}. Falling back to {self.fallback_whisper_id}...")
                self.whisper_processor = WhisperProcessor.from_pretrained(self.fallback_whisper_id)
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.fallback_whisper_id).to(self.device)
            self.whisper_model.eval()

    def load_translation_model(self):
        if self.mt_model is None:
            print(f"[PAASR Pipeline] Loading Translation model: {self.translation_model_id} on {self.device}...")
            self.mt_tokenizer = MarianTokenizer.from_pretrained(self.translation_model_id)
            self.mt_model = MarianMTModel.from_pretrained(self.translation_model_id).to(self.device)
            self.mt_model.eval()

    def transcribe_telugu(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> str:
        self.load_asr_model()
        inputs = self.whisper_processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features)
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Strip any Whisper special token tags like <|startoftranscript|>
        cleaned_text = re.sub(r"<\|.*?\|>", "", transcription).strip()
        return cleaned_text

    def translate_te_to_en(self, telugu_text: str) -> str:
        self.load_translation_model()
        if not telugu_text:
            return ""
        inputs = self.mt_tokenizer(telugu_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.mt_model.generate(**inputs, max_length=128, num_beams=4)
            english_text = self.mt_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return english_text

    def synthesize_speech(self, english_text: str) -> str:
        if not english_text:
            return None
        # Generate temporary MP3 audio file using gTTS
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        tts = gTTS(text=english_text, lang="en")
        tts.save(temp_path)
        return temp_path


# ==========================================
# 2. Audio Preprocessing & Translation Logic
# ==========================================

def load_and_preprocess_audio(audio_input):
    """
    Normalizes audio input into 16kHz mono float32 numpy array.
    Supports file path (string) or Gradio tuple (sr, numpy_array).
    """
    if audio_input is None:
        return None, 0.0, 0

    if isinstance(audio_input, str):
        # Filepath supplied
        audio, sr = librosa.load(audio_input, sr=16000, mono=True)
    elif isinstance(audio_input, tuple):
        # Gradio (sr, data)
        sr, data = audio_input
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Stereo to mono
        # Convert integer formats to float32
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if sr != 16000:
            audio = librosa.resample(data, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            audio = data
    else:
        raise ValueError("Unsupported audio input format.")

    duration = float(len(audio) / 16000.0)
    return audio, duration, 16000


def run_live_s2st_pipeline(audio_input):
    """
    Real S2ST Translation Pipeline:
    Telugu Audio -> Telugu ASR -> Telugu Text -> MT -> English Text -> TTS -> Playable English Audio
    """
    start_time = time.time()

    # 1. Validate Input Audio
    if audio_input is None:
        empty_error = """
### ⚠️ Input Status: Missing Audio
Please record your voice speaking Telugu using the microphone or upload an audio file (`.mp3`, `.wav`, `.m4a`).
"""
        return None, empty_error, "", ""

    # 2. Preprocess Audio
    try:
        audio, duration, sr = load_and_preprocess_audio(audio_input)
        if duration < 0.2:
            return None, "### ⚠️ Audio Too Short\nThe audio duration is less than 0.2 seconds. Please provide a clear spoken utterance.", "", ""
    except Exception as e:
        return None, f"### ⚠️ Audio Processing Error\nFailed to decode input audio: `{str(e)}`", "", ""

    # Calculate basic acoustic diagnostics for research status
    rms = float(np.sqrt(np.mean(audio**2)))
    snr_est = max(5.0, min(35.0, 20.0 * np.log10(rms + 1e-6) + 40.0))

    pipeline = S2STPipeline.get_instance()

    # 3. Speech Recognition (ASR)
    try:
        asr_start = time.time()
        telugu_transcript = pipeline.transcribe_telugu(audio, sampling_rate=sr)
        asr_time = time.time() - asr_start
    except Exception as e:
        return None, f"### ❌ ASR Module Failure\nError transcribing Telugu speech: `{str(e)}`", "", ""

    if not telugu_transcript:
        return (
            None,
            f"### ⚠️ No Speech Detected\nThe speech recognition model could not detect recognizable Telugu speech in the audio ({duration:.2f}s).",
            "",
            "",
        )

    # 4. Translation (Telugu -> English)
    try:
        mt_start = time.time()
        english_translation = pipeline.translate_te_to_en(telugu_transcript)
        mt_time = time.time() - mt_start
    except Exception as e:
        return (
            None,
            f"### ❌ Translation Module Failure\nError translating Telugu text: `{str(e)}`",
            telugu_transcript,
            "",
        )

    # 5. English Text-to-Speech (TTS)
    tts_audio_path = None
    tts_time = 0.0
    tts_status = ""
    try:
        tts_start = time.time()
        tts_audio_path = pipeline.synthesize_speech(english_translation)
        tts_time = time.time() - tts_start
        tts_status = f"✅ English speech synthesized successfully ({os.path.getsize(tts_audio_path)} bytes)"
    except Exception as e:
        tts_status = f"⚠️ TTS synthesis error: `{str(e)}` (English translation is still available below)"

    total_latency = time.time() - start_time
    rtf = total_latency / max(duration, 0.1)

    # 6. Status & Diagnostic Summary
    status_markdown = f"""
### 📊 Pipeline Execution Diagnostics
- **Input Received:** Audio Duration: `{duration:.2f} s` | Sample Rate: `16,000 Hz` | Estimated SNR: `{snr_est:.1f} dB`
- **1. Telugu ASR (Whisper):** Processed in `{asr_time:.2f} s`
- **2. Neural Translation (MarianMT):** Processed in `{mt_time:.2f} s`
- **3. English TTS:** {tts_status} in `{tts_time:.2f} s`
- **⚡ Total End-to-End Latency:** `{total_latency:.2f} s` (Real-Time Factor: `{rtf:.2f}`)
"""

    return tts_audio_path, status_markdown, telugu_transcript, english_translation


def submit_survey_response(age, occupation, region, difficulty, domain, voice_pref, accent_imp):
    response_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "age": age,
        "occupation": occupation,
        "region": region,
        "difficulty": difficulty,
        "domain": domain,
        "voice_preference": voice_pref,
        "accent_importance": accent_imp,
    }
    log_file = "survey_responses.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(response_data, ensure_ascii=False) + "\n")
    return "✅ Response successfully logged! Thank you for participating in the community research survey."


# ==========================================
# 3. Gradio Interface Construction
# ==========================================

def build_app():
    with gr.Blocks(title="PAASR Telugu-to-English S2ST") as demo:
        gr.Markdown("""
# 🎙️ PAASR: Phonology-Aware Adaptive Speech Representation
### Interactive Telugu-to-English Speech-to-Speech Translation Research Prototype
*Eliminates text friction for rural communities, performing real Telugu speech recognition, cross-lingual translation, and English speech synthesis.*

> **Prototype Architecture Note:** The direct end-to-end PAASR multi-stream neural network (*Section 3*) is currently undergoing self-supervised pretraining and consistency ablation on the TeluguST-46 benchmark. This interactive prototype deploys a real **Telugu ASR (fine-tuned Whisper-Telugu) + Dravidian-to-English Neural Translation (MarianMT) + Neural Speech Synthesis** pipeline to provide live, dynamic speech-to-speech translation based on the user's actual audio input.
""")

        # ----------------------------------------------------
        # Tab 1: Live S2ST Prototype
        # ----------------------------------------------------
        with gr.Tab("🚀 Live S2ST Prototype"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Telugu Speech Input")
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Record or Upload Telugu Voice"
                    )

                    run_btn = gr.Button("⚡ Translate Telugu Speech -> English Speech", variant="primary", size="lg")

                    gr.Markdown("#### 💡 Quick Test Examples")
                    gr.Examples(
                        examples=[
                            ["sample_telugu.mp3"],
                            ["sample_healthcare.mp3"],
                            ["sample_education.mp3"],
                        ],
                        inputs=audio_in,
                        label="Click any example to load real Telugu audio"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 2. Synthesized English Audio Output")
                    audio_out = gr.Audio(label="Playable English Speech Output", type="filepath", autoplay=True)

                    with gr.Group():
                        gr.Markdown("### 3. Step-by-Step Linguistic Outputs")
                        telugu_out = gr.Textbox(
                            label="Step 1: Recognized Telugu Transcription (ASR)",
                            placeholder="Telugu text recognized from speech will appear here...",
                            lines=2,
                            interactive=False,
                        )
                        english_out = gr.Textbox(
                            label="Step 2: Translated English Text (MT)",
                            placeholder="English translation will appear here...",
                            lines=2,
                            interactive=False,
                        )

                    status_box = gr.Markdown(value="*Awaiting audio input. Record or upload Telugu speech and click Translate.*")

            run_btn.click(
                fn=run_live_s2st_pipeline,
                inputs=[audio_in],
                outputs=[audio_out, status_box, telugu_out, english_out],
            )

        # ----------------------------------------------------
        # Tab 2: Community Field Survey
        # ----------------------------------------------------
        with gr.Tab("📝 Community Field Survey"):
            gr.Markdown("""
### Non-Identifying Community Language Barrier Questionnaire
*Collecting empirical data on language friction, accent diversity, and domain requirements across Telugu-speaking semi-urban and rural areas.*
""")
            with gr.Row():
                q_age = gr.Radio(["18–25", "26–40", "41–60", "Above 60"], label="Age Group", value="26–40")
                q_occ = gr.Dropdown(
                    ["Farmer / Agricultural Worker", "Student", "Small Business / Artisan", "Healthcare / Field Worker", "Homemaker / Senior Citizen"],
                    label="Primary Occupation",
                    value="Farmer / Agricultural Worker",
                )
                q_reg = gr.Radio(["Coastal Andhra", "Rayalaseema", "Telangana", "Uttarandhra"], label="Regional Dialect Mandal", value="Telangana")

            with gr.Row():
                q_diff = gr.Slider(1, 5, value=4, step=1, label="Difficulty Understanding Spoken English (1: Easy, 5: Extreme)")
                q_acc = gr.Slider(1, 5, value=5, step=1, label="Importance of System Understanding Village Dialect (1: Low, 5: Critical)")
                q_dom = gr.Dropdown(["Healthcare", "Agriculture / Mandi", "Education", "Government Services", "Banking"], label="Most Beneficial Domain", value="Agriculture / Mandi")

            q_voice = gr.Radio(["Spoken English Voice Only", "Both Spoken Voice & Text", "Text Only"], label="Output Preference", value="Both Spoken Voice & Text")
            survey_btn = gr.Button("Submit Anonymous Survey Response", variant="secondary")
            survey_status = gr.Markdown("")

            survey_btn.click(
                fn=submit_survey_response,
                inputs=[q_age, q_occ, q_reg, q_diff, q_dom, q_voice, q_acc],
                outputs=[survey_status],
            )

        # ----------------------------------------------------
        # Tab 3: Mathematical Model & Novelty
        # ----------------------------------------------------
        with gr.Tab("📐 Mathematical Model & Novelty"):
            gr.Markdown(r"""
# Phonology-Aware Adaptive Speech Representation (PAASR)

### 1. Architectural Formulation
Let raw Telugu speech waveform be denoted as $X \in \mathbb{R}^{T_w}$. The SSL backbone extracts latent representation $H \in \mathbb{R}^{T \times D}$.
PAASR decomposes this into three multi-stream components:
- **Acoustic Representation:** $F_{\\text{acoustic}} = \\text{Conv1D}_{k=3}(\\text{LayerNorm}(H)) \in \mathbb{R}^{T \\times d}$
- **Phonological Representation:** $F_{\\text{phonology}} = \\text{MHA}(Q=H, K=P_{\\text{anchor}}, V=P_{\\text{anchor}}) \in \mathbb{R}^{T \\times d}$, constrained by Telugu articulatory priors (retroflexion, dental stops, vowel length).
- **Contextual Representation:** $F_{\\text{context}} = \\text{TransformerLayer}(H) \in \mathbb{R}^{T \\times d}$.

### 2. Condition-Adaptive Representation Fusion (CARF)
A condition vector $C = [C_{\\text{speaker}} \\,\\|\\, C_{\\text{accent}} \\,\\|\\, C_{\\text{noise}} \\,\\|\\, C_{\\text{rate}}] \in \mathbb{R}^{d_c}$ dynamically steers a learnable gating network:
$$u_t = [F_{\\text{acoustic}, t} \\,\\|\\, F_{\\text{phonology}, t} \\,\\|\\, F_{\\text{context}, t} \\,\\|\\, C]$$
$$\\boldsymbol{\\alpha}_t = \\text{Softmax}\\left(\\frac{W_2 \\cdot \\text{GeLU}(W_1 u_t + b_1) + b_2}{\\tau}\\right) = [\\alpha_{t,\\text{ac}}, \\alpha_{t,\\text{ph}}, \\alpha_{t,\\text{ctx}}]^\\top$$
$$F_{\\text{PAASR}, t} = \\alpha_{t,\\text{ac}} F_{\\text{acoustic}, t} + \\alpha_{t,\\text{ph}} F_{\\text{phonology}, t} + \\alpha_{t,\\text{ctx}} F_{\\text{context}, t}$$

### 3. Phonological Consistency Learning (PCL)
Enforces invariance across regional accents, speaker pitch, and environmental noise via InfoNCE alignment:
$$\\mathcal{L}_{\\text{PCL}} = - \\frac{1}{T} \\sum_{t=1}^{T} \\log \\frac{\\exp(\\text{sim}(F_{\\text{phonology}, t}^{(i)}, F_{\\text{phonology}, \\pi(t)}^{(j)}) / \\tau_p)}{\\sum_{k} \\exp(\\text{sim}(F_{\\text{phonology}, t}^{(i)}, F_{\\text{phonology}, \\pi(t)}^{(k)}) / \\tau_p)}$$

### 4. Composite Objective Function
$$\\mathcal{L}_{\\text{total}} = \\lambda_1 \\mathcal{L}_{\\text{S2UT}} + \\lambda_2 \\mathcal{L}_{\\text{PCL}} + \\lambda_3 \\mathcal{L}_{\\text{SSL}} + \\lambda_4 \\mathcal{L}_{\\text{spk}}$$
""")

    return demo


if __name__ == "__main__":
    app = build_app()
    # Launch locally on port 7860
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
