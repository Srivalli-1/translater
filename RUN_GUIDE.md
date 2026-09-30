# How to Run the PAASR Project

This repository contains the architecture, research proposal, community evaluation framework, and prototype implementation for:
**"A Phonology-Aware Adaptive Speech Representation Framework for Robust Low-Resource Telugu-to-English Direct Speech-to-Speech Translation (PAASR)"**.

---

## 1. Quick Start: Running the Architecture Verification Script

The script `paasr_model.py` validates the multi-stream decomposition ($F_{\text{acoustic}}, F_{\text{phonology}}, F_{\text{context}}$), condition estimation ($C$), adaptive softmax gating (CARF), and Phonological Consistency Loss (PCL).

### Step 1.1: Install PyTorch & Dependencies
Open PowerShell or Command Prompt in this folder and install PyTorch:
```powershell
pip install torch torchaudio
```
Or install all dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 1.2: Execute the Architecture Test
```powershell
python paasr_model.py
```

**Expected Output:**
```text
=== Testing PAASR Model forward pass ===
Executing on device: cuda (or cpu)

[Shapes Verification]
PAASR Fused Representation : torch.Size([4, 100, 512]) (Expected: [4, 100, 512])
Gating Weights alpha       : torch.Size([4, 100, 3]) (Expected: [4, 100, 3])
Speaker Vector C_spk       : torch.Size([4, 192]) (Expected: [4, 192])
Accent Posterior C_acc     : torch.Size([4, 4]) (Expected: [4, 4])
Target Unit Logits         : torch.Size([4, 30, 1000]) (Expected: [4, 30, 1000])
Phonological Consistency Loss (PCL): 1.3863

=== All PAASR architecture sanity checks passed successfully! ===
```

---

## 2. Running the Interactive Web Demonstration UI

The prototype includes an interactive Gradio web application (`demo_app.py`) for live voice testing, simulated condition analysis, gating visualization, and community survey collection.

### Step 2.1: Install Gradio
```powershell
pip install gradio
```

### Step 2.2: Launch the App
```powershell
python demo_app.py
```
After running, open your web browser at:
```
http://127.0.0.1:7860
```

### Features Available in the Web UI:
1. **Live S2ST Prototype:** Test voice translation across three preset community domains (Healthcare, Agriculture, Education), view real-time extracted condition metrics ($C$), and see dynamic gating weights ($\boldsymbol{\alpha}$).
2. **Community Field Survey:** A digital version of the 10-question survey that logs responses directly to `survey_responses.jsonl`.
3. **Mathematical Model & Novelty:** Full interactive summary of equations and architectural flow.

---

## 3. Viewing the Research Proposal & Presentations

All master documentation files are standard GitHub-flavored Markdown:

| Document | Purpose |
| :--- | :--- |
| **`PAASR_Research_Proposal.md`** | Complete 11-section formal proposal with literature survey, block diagrams (Mermaid), mathematical formulation, and Gantt chart. |
| **`Community_Survey_Questionnaire.md`** | Printable survey form with ethical statements, demographic rubrics, and field observation log. |
| **`PAASR_Presentation_Deck.md`** | 10-slide executive pitch deck for funding review and defense. |

To preview or convert to PDF:
- **In VS Code / IDE:** Press `Ctrl + Shift + V` to preview the rendered document with diagrams.
- **Convert to Word/PDF:** You can use Pandoc:
  ```powershell
  pandoc PAASR_Research_Proposal.md -o PAASR_Research_Proposal.docx
  ```
