# PAASR Project Presentation Deck
## A Phonology-Aware Adaptive Speech Representation Framework for Robust Low-Resource Telugu-to-English Direct Speech-to-Speech Translation

---

### Slide 1: Title & Executive Summary
- **Title:** Phonology-Aware Adaptive Speech Representation (PAASR) for Robust Low-Resource Telugu-to-English Direct S2ST
- **Domain:** Multimodal AI, Speech Processing, Low-Resource NLP
- **Core Innovation:** End-to-end direct speech-to-speech translation eliminating intermediate text generation, leveraging self-supervised Telugu representations, articulatory phonological inductive bias, and dynamic condition-adaptive gating.
- **Target Audience:** 95M+ Telugu speakers, especially rural farmers, students, senior citizens, and patients.

---

### Slide 2: The Core Problem & Social Context
- **Linguistic Barrier:** Telugu-speaking rural and semi-urban populations face acute difficulties in healthcare consultations, agricultural advisories, and administrative services.
- **Literacy & UI Friction:** Existing tools require typing in Telugu script or phonetic transliteration, excluding non-literate and elderly demographics.
- **Telugu Linguistic Peculiarities:**
  - Morphologically rich, agglutinative language.
  - Fine-grained retroflex vs. dental phonetic contrasts (/ʈ/ vs. /t̪/).
  - High degree of regional dialect variance (Coastal Andhra, Rayalaseema, Telangana, Uttarandhra).

---

### Slide 3: Limitations of Existing Systems (The Cascade Bottleneck)
- **Cascade Pipeline:** Telugu Audio $\rightarrow$ ASR $\rightarrow$ Telugu Text $\rightarrow$ MT $\rightarrow$ English Text $\rightarrow$ TTS $\rightarrow$ English Audio.
- **Critical Flaws:**
  1. **Error Compounding:** A single ASR misrecognition corrupts translation and synthesis downstream.
  2. **Loss of Paralinguistics:** Tone, emotion, vocal urgency, and speaker voice identity are completely erased.
  3. **High Latency:** Sequential inference creates $>3.5$ s delays, impeding natural conversation.

---

### Slide 4: Proposed Solution: The PAASR Framework
- **Direct Speech-to-Speech (S2ST):** Maps Telugu continuous speech representations directly into target English discrete speech units.
- **Three-Stream Representation:**
  - $F_{\text{acoustic}}$: Captures harmonic, pitch, and energy dynamics.
  - $F_{\text{phonology}}$: Aligned with Telugu articulatory anchor codebooks.
  - $F_{\text{context}}$: Captures long-range syntactic and contextual relations.
- **Condition-Adaptive Representation Fusion (CARF):** Dynamically reweights streams based on estimated speaker, accent, SNR, and speaking rate.

---

### Slide 5: Key Technical Novelties
1. **Phonology-Aware Adaptive Gating:** Dynamic frame-level gating $\boldsymbol{\alpha}_t = \text{Softmax}(W_2 \text{GeLU}(W_1 [F; C] + b_1) + b_2)$.
2. **Phonological Consistency Learning (PCL):** Contrastive InfoNCE invariance loss forcing representations of phonetically identical utterances to align across noisy backgrounds and different regional speakers.
3. **Condition-Adaptive Representation Fusion (CARF):** Explicit extraction of $C = [C_{\text{speaker}}, C_{\text{accent}}, C_{\text{noise}}, C_{\text{rate}}]$ to insulate translation from acoustic noise.
4. **Speaker-Preserving Direct Synthesis:** Conditioning target English HiFi-GAN vocoder on source $C_{\text{speaker}}$ embeddings to clone source vocal identity in the translated English speech.

---

### Slide 6: System Architecture & Workflow
*(Refer to Section 5 Mermaid Flowcharts)*
1. **Source Speech Input** ($16\text{ kHz}$) $\rightarrow$ Audio Augmentation (SpecAugment, Noise, RIR).
2. **Pretrained Telugu SSL Encoder** (XLS-R / HuBERT) $\rightarrow$ Latent frames $H$.
3. **PAASR Multi-Stream & Condition Estimator** $\rightarrow$ CARF Adaptive Gating $\rightarrow$ $F_{\text{PAASR}}$.
4. **S2UT Discrete Unit Decoder** $\rightarrow$ English Speech Tokens.
5. **HiFi-GAN Unit Vocoder** conditioned on $C_{\text{speaker}}$ $\rightarrow$ Natural English Speech Output.

---

### Slide 7: Literature Positioning & Benchmark Foundation
- Grounded on 10 high-impact recent journal studies (2024–2026) across *Nature*, *Speech Communication*, *Computer Speech & Language*, *IEEE Access*, and *EURASIP*.
- Directly benchmarked against **TeluguST-46** (Akkiraju et al., IJCNLP-AACL 2025):
  - 46 hours of manually verified Telugu–English speech translation (30h train, 8h dev, 8h test).
  - Pretrained on $>2,500$ hours of unannotated Telugu speech (IndicTTS, IIIT-H, OpenSLR).

---

### Slide 8: Evaluation Framework & Success Metrics
- **Translation Quality:** BLEU ($\ge +3.5$ over cascade), chrF++, METEOR, COMET.
- **Speech Quality:** UTMOS ($\ge 3.8/5.0$), Human MOS for naturalness and clarity.
- **Speaker Fidelity:** Cosine Similarity of ECAPA-TDNN embeddings between source and target speech ($\ge 0.78$).
- **Robustness:** Performance degradation under 0 dB SNR restricted to $<15\%$ relative BLEU drop.
- **Efficiency:** Real-Time Factor (RTF) $<0.3$, End-to-end latency $<800\text{ ms}$.

---

### Slide 9: Community Validation & Ethical Governance
- **Field Deployments:** 3 rural/semi-urban mandals in Andhra Pradesh and Telangana.
- **Non-Identifying Survey:** Quantitative evaluation of language friction, domain priorities, noise conditions, and voice interface acceptance.
- **Data Governance:** Strict anonymization, no storage of biometric identity without institutional ethics review (IEC/IRB approval).

---

### Slide 10: Milestones & Expected Outcomes
- **Month 1–6:** Data curation, baseline SSL pretraining, community baseline survey.
- **Month 7–12:** PAASR multi-stream architecture, PCL contrastive learning, CARF fusion.
- **Month 13–18:** Direct S2UT translation decoder training, speaker-conditioned HiFi-GAN vocoder.
- **Month 19–24:** Comprehensive ablation benchmark, model quantization/edge deployment, field pilot and final report.
- **Deliverables:** Working open-source prototype, pretrained model checkpoints, community field report, high-impact journal publications.
