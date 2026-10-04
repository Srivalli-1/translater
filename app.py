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
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import gradio as gr

try:
    import librosa
except Exception as exc:  # pragma: no cover - runtime fallback for constrained environments
    librosa = None
    LIBROSA_IMPORT_ERROR = exc
else:
    LIBROSA_IMPORT_ERROR = None

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - runtime fallback for constrained environments
    sf = None
    SOUNDFILE_IMPORT_ERROR = exc
else:
    SOUNDFILE_IMPORT_ERROR = None

try:
    import torch
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
except Exception as exc:  # pragma: no cover - runtime fallback for constrained environments
    torch = None
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

try:
    from gtts import gTTS
except Exception as exc:  # pragma: no cover - runtime fallback for constrained environments
    gTTS = None
    GTTS_IMPORT_ERROR = exc
else:
    GTTS_IMPORT_ERROR = None

# Route Hugging Face cache to D: drive if available to avoid low disk space issues on C:
if "HF_HOME" not in os.environ and os.path.exists("D:\\"):
    os.environ["HF_HOME"] = r"D:\huggingface"


def find_nllb_model_path() -> str:
    """Resolve the local path or HF hub ID for facebook/nllb-200-distilled-600M."""
    candidate_paths = [
        os.environ.get("NLLB_MODEL_PATH", ""),
        r"D:\huggingface\models--facebook--nllb-200-distilled-600M\snapshots\f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        r"D:\hf-cache\hub\models--facebook--nllb-200-distilled-600M\snapshots\f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    ]
    for p in candidate_paths:
        if p and os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "pytorch_model.bin")) or
            os.path.exists(os.path.join(p, "model.safetensors"))
        ):
            return p

    for root_dir in [r"D:\huggingface", r"D:\hf-cache\hub", r"D:\hf-test-cache-translater\hub"]:
        model_dir = os.path.join(root_dir, "models--facebook--nllb-200-distilled-600M", "snapshots")
        if os.path.isdir(model_dir):
            for snap in os.listdir(model_dir):
                snap_path = os.path.join(model_dir, snap)
                if os.path.isdir(snap_path) and (
                    os.path.exists(os.path.join(snap_path, "pytorch_model.bin")) or
                    os.path.exists(os.path.join(snap_path, "model.safetensors"))
                ):
                    return snap_path

    return "facebook/nllb-200-distilled-600M"


# ==========================================
# 1. Model Singleton & Pipeline Initialization
# ==========================================

class S2STPipeline:
    """
    Manages lazy-loading and inference for:
    1. Telugu ASR (fine-tuned Whisper-Telugu)
    2. Telugu -> English Translation (facebook/nllb-200-distilled-600M)
    3. English Text-to-Speech (gTTS)
    """
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.whisper_model_id = "vasista22/whisper-telugu-tiny"
        self.translation_model_id = "facebook/nllb-200-distilled-600M"

        self.whisper_processor = None
        self.whisper_model = None
        self.translation_tokenizer = None
        self.translation_model = None
        self.mt_tokenizer = None
        self.mt_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = S2STPipeline()
        return cls._instance

    def load_asr_model(self):
        if torch is None or WhisperProcessor is None or WhisperForConditionalGeneration is None:
            raise RuntimeError(
                "ASR dependencies are not available. Install the project requirements or verify the Python environment."
            )
        if self.whisper_model is None:
            print(f"[PAASR Pipeline] Loading Telugu ASR model: {self.whisper_model_id} on {self.device}...")
            self.whisper_processor = WhisperProcessor.from_pretrained(self.whisper_model_id)
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_id).to(self.device)

            from transformers.models.whisper.tokenization_whisper import LANGUAGES, TASK_IDS, TO_LANGUAGE_CODE

            lang_dict = {}
            for lang_name, lang_code in TO_LANGUAGE_CODE.items():
                tok_id = self.whisper_processor.tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")
                if tok_id is not None:
                    lang_dict[lang_name] = tok_id
                    lang_dict[lang_code] = tok_id
                    lang_dict[f"<|{lang_code}|>"] = tok_id
            for lang_code in LANGUAGES:
                tok_id = self.whisper_processor.tokenizer.convert_tokens_to_ids(f"<|{lang_code}|>")
                if tok_id is not None:
                    lang_dict[lang_code] = tok_id
                    lang_dict[f"<|{lang_code}|>"] = tok_id

            task_dict = {}
            for t in TASK_IDS:
                tok_id = self.whisper_processor.tokenizer.convert_tokens_to_ids(f"<|{t}|>")
                if tok_id is not None:
                    task_dict[t] = tok_id
                    task_dict[f"<|{t}|>"] = tok_id

            self.whisper_model.generation_config.lang_to_id = lang_dict
            self.whisper_model.generation_config.task_to_id = task_dict
            self.whisper_model.generation_config.is_multilingual = True

            self.whisper_model.eval()

    def load_translation_model(self):
        if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise RuntimeError(
                "Translation dependencies are not available. Install the project requirements or verify the Python environment."
            )
        if self.translation_model is not None and self.translation_tokenizer is not None:
            return

        model_path = find_nllb_model_path()
        print(f"[PAASR Pipeline] Loading Translation model: {self.translation_model_id} (source: {model_path}) on {self.device}...")
        self.translation_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            src_lang="tel_Telu",
        )
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.translation_model.eval()
        self.mt_tokenizer = self.translation_tokenizer
        self.mt_model = self.translation_model

    def transcribe_telugu(self, audio_array: np.ndarray, sampling_rate: int = 16000) -> str:
        self.load_asr_model()
        audio_array = np.asarray(audio_array, dtype=np.float32)
        if audio_array.ndim > 1:
            if audio_array.shape[0] <= 8 and audio_array.shape[0] < audio_array.shape[1]:
                audio_array = np.mean(audio_array, axis=0)
            else:
                audio_array = np.mean(audio_array, axis=1)
        if sampling_rate != 16000:
            if librosa is not None:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            else:
                from scipy.signal import resample
                target_len = max(1, int(round(len(audio_array) * 16000 / sampling_rate)))
                audio_array = resample(audio_array, target_len)
            sampling_rate = 16000

        audio_array = audio_array.astype(np.float32)
        if audio_array.size == 0:
            raise ValueError("Whisper received an empty audio waveform.")
        if sampling_rate != 16000:
            raise ValueError(f"Whisper requires 16000 Hz audio; received {sampling_rate} Hz.")
        if not np.all(np.isfinite(audio_array)):
            raise ValueError("Whisper received audio containing NaN or infinite samples.")

        print("========== WHISPER INPUT ==========")
        print("sample_rate:", sampling_rate)
        print("shape:", audio_array.shape)
        print("dtype:", audio_array.dtype)
        print("duration:", len(audio_array) / sampling_rate)
        print("min:", float(audio_array.min()))
        print("max:", float(audio_array.max()))
        print("RMS:", float(np.sqrt(np.mean(audio_array ** 2))))
        print("===================================")

        inputs = self.whisper_processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                input_features,
                language="telugu",
                task="transcribe",
            )
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        print("========== WHISPER RESULT ==========")
        print("Raw transcription:", repr(transcription))
        print("====================================")
        cleaned_text = re.sub(r"<\|.*?\|>", "", transcription).strip()
        return cleaned_text

    def translate_te_to_en(self, telugu_text: str) -> str:
        if not telugu_text or not telugu_text.strip():
            raise ValueError("No Telugu transcription available")

        self.load_translation_model()

        device_str = "GPU" if "cuda" in str(self.device).lower() else "CPU"
        print("========== NLLB INPUT ==========")
        print("Telugu text:")
        print(telugu_text)
        print()
        print("Characters:")
        print(len(telugu_text))
        print()
        print("Model:")
        print("facebook/nllb-200-distilled-600M")
        print()
        print("Device:")
        print(device_str)
        print()
        print("Starting NLLB generation...")
        print("=================================")

        try:
            inputs = self.translation_tokenizer(
                telugu_text,
                return_tensors="pt",
                truncation=True,
                max_length=128
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.translation_model.generate(
                    **inputs,
                    forced_bos_token_id=self.translation_tokenizer.convert_tokens_to_ids("eng_Latn"),
                    max_new_tokens=64,
                    num_beams=2
                )

            english_text = self.translation_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).strip()

            print("NLLB generation:")
            print("SUCCESS")
            print()
            print("English:")
            print(english_text)
            print("================================")
            return english_text

        except Exception as exc:
            import traceback
            print("========== NLLB TRANSLATION ERROR ==========")
            print("Exception type:", type(exc).__name__)
            print("Exception:", str(exc))
            traceback.print_exc()
            print("=============================================")
            raise

    def synthesize_speech(self, english_text: str) -> str:
        if not english_text:
            return None
        if gTTS is None:
            raise RuntimeError("gTTS is not installed in this environment.")
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

def load_user_audio(audio_input):
    """Accept the actual Gradio audio payload returned by the live mic/upload component."""
    if audio_input is None:
        raise ValueError("No audio received.")

    def read_audio_file(audio_path):
        if sf is not None:
            try:
                return sf.read(audio_path, dtype="float32", always_2d=False)
            except Exception:
                if librosa is not None:
                    return librosa.load(audio_path, sr=None, mono=False)
                raise
        elif librosa is not None:
            return librosa.load(audio_path, sr=None, mono=False)
        raise ImportError("Audio decoding libraries are unavailable; install librosa or soundfile.")

    if isinstance(audio_input, (str, os.PathLike)):
        audio, sample_rate = read_audio_file(str(audio_input))
        audio_array = np.asarray(audio)

    elif isinstance(audio_input, dict):
        audio_path = audio_input.get("path") or audio_input.get("filepath") or audio_input.get("name")
        if audio_path and isinstance(audio_path, (str, os.PathLike)) and os.path.exists(str(audio_path)):
            audio, sample_rate = read_audio_file(str(audio_path))
            audio_array = np.asarray(audio)
        elif "data" in audio_input:
            data = audio_input["data"]
            sample_rate = int(audio_input.get("sample_rate") or 16000)
            audio_array = np.asarray(data)
        elif "array" in audio_input:
            data = audio_input["array"]
            sample_rate = int(audio_input.get("sample_rate") or 16000)
            audio_array = np.asarray(data)
        else:
            raise ValueError(f"Unsupported Gradio audio dictionary: {list(audio_input.keys())}")

    elif isinstance(audio_input, tuple):
        if len(audio_input) != 2:
            raise ValueError(f"Unsupported audio tuple structure: {audio_input}")
        sample_rate, data = audio_input
        if data is None:
            raise ValueError("Audio tuple contains no waveform data.")
        audio_array = np.asarray(data)
        sample_rate = int(sample_rate)

    elif isinstance(audio_input, np.ndarray):
        sample_rate = 16000
        audio_array = np.asarray(audio_input, dtype=np.float32)

    else:
        raise ValueError(f"Unsupported audio input format: {type(audio_input)}")

    if audio_array.size == 0:
        raise ValueError("Decoded audio array is empty.")

    if audio_array.ndim > 1:
        if audio_array.shape[0] <= 8 and audio_array.shape[0] < audio_array.shape[1]:
            audio_array = np.mean(audio_array, axis=0)
        else:
            audio_array = np.mean(audio_array, axis=1)

    if np.issubdtype(audio_array.dtype, np.integer):
        limits = np.iinfo(audio_array.dtype)
        audio_array = audio_array.astype(np.float32)
        if limits.min < 0:
            audio_array /= float(max(abs(limits.min), limits.max))
        else:
            audio_array = (audio_array - limits.max / 2) / (limits.max / 2)
    else:
        audio_array = audio_array.astype(np.float32)

    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError(f"Invalid audio sample rate: {sample_rate}")
    sample_rate = int(sample_rate)

    if not np.all(np.isfinite(audio_array)):
        raise ValueError("Decoded audio contains NaN or infinite samples.")

    if sample_rate != 16000:
        if librosa is not None:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        else:
            from scipy.signal import resample
            target_len = max(1, int(round(len(audio_array) * 16000 / sample_rate)))
            audio_array = resample(audio_array, target_len)
        sample_rate = 16000

    audio_array = np.asarray(audio_array, dtype=np.float32)
    peak = float(np.max(np.abs(audio_array))) if audio_array.size > 0 else 0.0
    if peak > 1.0:
        audio_array /= peak

    print("Audio received:", audio_array is not None)
    print("Audio shape:", getattr(audio_array, "shape", None))
    print("Audio dtype:", getattr(audio_array, "dtype", None))
    print("Sample rate:", sample_rate)
    return audio_array.astype(np.float32), 16000


def load_and_preprocess_audio(audio_input):
    """Backward-compatible wrapper that normalizes user audio to 16kHz mono float32."""
    return load_user_audio(audio_input)


def translate_live_audio(audio_input):
    print("========== LIVE AUDIO DEBUG ==========")
    print("audio_input type:", type(audio_input))
    print("audio_input:", audio_input)
    print("======================================")

    if audio_input is None:
        print("[PAASR Debug] No audio was provided.")
        return gr.update(value=None, visible=False), gr.update(value="### ⚠️ Input Status: Missing Audio\nPlease record or upload Telugu speech before clicking the button.", visible=True), "", ""

    try:
        audio, sample_rate = load_user_audio(audio_input)
        duration = len(audio) / sample_rate
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print("Audio duration:", duration)
        print("Audio min:", float(audio.min()))
        print("Audio max:", float(audio.max()))
        print("Audio RMS:", rms)
        print("Starting Telugu ASR...")
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    if audio.size == 0 or rms < 1e-5:
        message = (
            "### ❌ The recording is silent or contains no usable audio.\n"
            f"Sample rate: {sample_rate} Hz · Duration: {duration:.2f} s · "
            f"Shape: {audio.shape} · Dtype: {audio.dtype} · RMS: {rms:.8f}"
        )
        print("[PAASR Debug] Recording is silent; not sending it to Whisper.")
        return gr.update(value=None, visible=False), gr.update(value=message, visible=True), "", ""

    pipeline = S2STPipeline.get_instance()

    try:
        telugu_transcript = pipeline.transcribe_telugu(audio, sampling_rate=sample_rate)
        print("ASR completed.")
        print("Recognized Telugu:")
        print(telugu_transcript)
    except Exception:
        import traceback
        traceback.print_exc()
        raise

    if not telugu_transcript or not telugu_transcript.strip():
        print("[PAASR Debug] ASR returned empty text.")
        message = (
            "### ❌ Whisper returned an empty transcription.\n"
            f"Sample rate: {sample_rate} Hz · Duration: {duration:.2f} s · "
            f"Shape: {audio.shape} · Dtype: {audio.dtype} · RMS: {rms:.8f}"
        )
        return gr.update(value=None, visible=False), gr.update(value=message, visible=True), "", ""

    print()
    print("========== LIVE S2ST ==========")
    print("ASR Telugu:")
    print(telugu_transcript)
    print()

    try:
        print("Starting Telugu → English translation...")
        english_translation = pipeline.translate_te_to_en(telugu_transcript)
        print("English translation:")
        print(english_translation)
    except Exception as exc:
        print(f"[PAASR Debug] Translation failed: {exc}")
        return gr.update(value=None, visible=False), gr.update(value=f"### ❌ Telugu → English translation failed: {exc}", visible=True), telugu_transcript, ""

    if not english_translation or not english_translation.strip():
        print("[PAASR Debug] Translation returned empty text.")
        return gr.update(value=None, visible=False), gr.update(value="### ❌ Telugu → English translation failed: empty output returned.", visible=True), telugu_transcript, ""

    try:
        print("Starting English TTS...")
        english_audio_path = pipeline.synthesize_speech(english_translation)
        print("TTS completed.")
        print(f"Generated audio file: {english_audio_path}")
    except Exception as exc:
        print(f"[PAASR Debug] English speech generation failed: {exc}")
        return gr.update(value=None, visible=False), gr.update(value="⚠️ English speech generation failed.", visible=True), telugu_transcript, english_translation

    return gr.update(value=english_audio_path, visible=True), gr.update(visible=False), telugu_transcript, english_translation


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


_nllb_initialized = False

def init_nllb_at_startup():
    global _nllb_initialized
    if _nllb_initialized:
        return

    print("CUDA available:", torch.cuda.is_available() if torch is not None else False)
    if torch is not None and torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    pipeline = S2STPipeline.get_instance()
    pipeline.load_translation_model()

    tok_loaded = "YES" if pipeline.translation_tokenizer is not None else "NO"
    model_loaded = "YES" if pipeline.translation_model is not None else "NO"
    device_str = "GPU" if "cuda" in str(pipeline.device).lower() else "CPU"

    print("========== NLLB STATUS ==========")
    print("Model: facebook/nllb-200-distilled-600M")
    print(f"Tokenizer loaded: {tok_loaded}")
    print(f"Model loaded: {model_loaded}")
    print(f"Device: {device_str}")
    print("=================================")

    # Test NLLB with simple string first (Requirement 4)
    test_sentence = "నాకు కడుపు నొప్పి వస్తుంది"
    print("\n[PAASR Startup Test] Testing NLLB translation with benchmark string:")
    test_translation = pipeline.translate_te_to_en(test_sentence)
    print(f"[PAASR Startup Test] NLLB verified successfully! Result: '{test_translation}'\n")

    _nllb_initialized = True


# ==========================================
# 3. Gradio Interface Construction
# ==========================================

def build_app():
    init_nllb_at_startup()
    with gr.Blocks(title="PAASR Telugu-to-English S2ST") as demo:
        gr.Markdown("""
# 🎙️ PAASR: Phonology-Aware Adaptive Speech Representation
### Interactive Telugu-to-English Speech-to-Speech Translation Research Prototype
*Eliminates text friction for rural communities, performing real Telugu speech recognition, cross-lingual translation, and English speech synthesis.*

> **Prototype Architecture Note:** The direct end-to-end PAASR multi-stream neural network (*Section 3*) is currently undergoing self-supervised pretraining and consistency ablation on the TeluguST-46 benchmark. This interactive prototype deploys a real **Telugu ASR (fine-tuned Whisper-Telugu) + Telugu-to-English Neural Translation (NLLB-200) + Neural Speech Synthesis** pipeline to provide live, dynamic speech-to-speech translation based on the user's actual audio input.
""")

        # ----------------------------------------------------
        # Tab 1: Live S2ST Prototype
        # ----------------------------------------------------
        with gr.Tab("🚀 Live S2ST Prototype"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_in = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎙️ Record or Upload Telugu Voice"
                    )

                    run_btn = gr.Button("⚡ Translate Telugu Speech → English Speech", variant="primary", size="lg")

                with gr.Column(scale=1):
                    telugu_out = gr.Textbox(
                        label="📝 Recognized Telugu Input",
                        placeholder="Recognized Telugu text will appear here...",
                        lines=2,
                        interactive=False,
                    )
                    english_out = gr.Textbox(
                        label="🌍 Translated English Speech/Text",
                        placeholder="English translation will appear here...",
                        lines=2,
                        interactive=False,
                    )
                    status_box = gr.Markdown(visible=False)
                    audio_out = gr.Audio(
                        label="🔊 English Audio Output",
                        type="filepath",
                        autoplay=True,
                        visible=False,
                    )

            run_btn.click(
                fn=translate_live_audio,
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
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), share=False)
