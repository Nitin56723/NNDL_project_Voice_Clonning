# NNDL Project — Real-Time Voice Cloning for AAC

A real-time voice cloning system designed for people with progressive speech loss (e.g., ALS). The user records or uploads a short voice sample, and the system synthesizes any typed sentence in that cloned voice — acting as a real-time Augmentative and Alternative Communication (AAC) device.

---

## What We Built

A custom AAC interface (`aac_speak.py`) built on top of **IndexTTS-2.5**, a state-of-the-art zero-shot Text-to-Speech model. The system takes a short reference audio clip (as short as 2–10 seconds) and clones that voice to speak any text typed by the user.

Key features:
- **One-shot voice cloning** — no training required, just upload a short recording
- **Real-time sentence queue** — type the next sentence while the current one is still synthesizing
- **Pre-warm on upload** — speaker embedding is cached when the voice is uploaded, so all subsequent sentences are fast
- **Auto audio conversion** — accepts OGG, MP3, WAV; automatically trims to 10s and converts to WAV
- **Progress tracking** — live color-coded progress bars per sentence (Waiting → Synthesizing → Playing)
- **Missing word fix** — automatically appends punctuation so the model never cuts off the last word
- **No buffer hangs** — uses Windows native `winsound` for glitch-free playback

---

## Pipeline

```
User uploads voice (OGG / WAV / MP3)
        │
        ▼
Convert & trim to 10s WAV (librosa)
        │
        ▼
Pre-warm: compute speaker embedding (IndexTTS-2.5)
  — cached in memory for the entire session
        │
        ▼
User types sentence → press Enter
        │
        ▼
Sentence added to synthesis queue
        │
        ▼
synthesis_worker (background thread)
  ├─ Acquires GPU lock (synthesis_lock)
  ├─ Runs IndexTTS-2.5 inference
  │     ├─ GPT token generation       (~2–6s)
  │     ├─ S2mel (mel spectrogram)    (~1.5–2.5s)
  │     └─ BigVGAN vocoder (CUDA)     (~0.3s)
  └─ Releases GPU lock
        │
        ▼
Play WAV via winsound (blocking, no buffer issues)
        │
        ▼
Next sentence starts automatically
```

### Speaker Embedding Cache

IndexTTS-2.5 caches the speaker embedding by reference file path:
- **First synthesis** (or pre-warm): computes and caches the speaker embedding (~3–5s overhead)
- **All subsequent sentences**: embedding reused from cache — no recomputation
- **Cache resets only** when a new reference voice is uploaded

This means pre-warming once on upload makes every sentence after it significantly faster.

---

## Model

**IndexTTS-2.5** — Zero-shot TTS system

| Component | Role | Size |
|---|---|---|
| GPT (`gpt.pth`) | Autoregressive token generation | 3.1 GB |
| S2mel (`s2mel.pth`) | Token → mel spectrogram | 396 MB |
| BigVGAN (`model.safetensors`) | Mel → waveform (vocoder) | 1.1 GB |
| Codec (`codec.pth`) | Audio tokenizer | 579 MB |
| W2V-BERT 2.0 | Speaker encoder (embedding) | 2.0 GB |
| **Total** | | **~10.4 GB** |

Model runs in **FP16 low-VRAM mode** on the RTX 3050 6GB GPU.

---

## Results

| Metric | Value |
|---|---|
| Reference audio needed | 2–10 seconds |
| First sentence (after pre-warm) | ~8–10 seconds |
| Subsequent sentences | ~8–10 seconds |
| Voice similarity | High (dependent on reference audio quality) |
| GPU used | NVIDIA RTX 3050 6GB Laptop (CUDA, FP16) |
| Playback | Glitch-free via `winsound` |

**Before optimizations** (baseline):
- Synthesis time: 20–26 seconds per sentence
- Cause: concurrent GPU usage between pre-warm thread and synthesis thread

**After optimizations**:
- Synthesis time: 8–10 seconds per sentence
- Fixes applied: GPU mutex lock, queue-based pre-warm, removed wrong-voice cache invalidation, BigVGAN CUDA kernel (`ninja`)

---

## How to Run

### 1. Install dependencies

```bash
pip install uv
uv sync
uv add ninja sounddevice
```

### 2. Download model weights

Download the IndexTTS-2.5 model weights from [HuggingFace](https://huggingface.co/IndexTeam/IndexTTS-2.5) and place them in the `checkpoints/` folder.

### 3. Start the AAC UI

```bash
.\.venv\Scripts\python.exe -u aac_speak.py
```

Then open **http://127.0.0.1:7861** in your browser.

### 4. Use it

1. Upload a short voice recording (OGG/WAV/MP3, 2–10 seconds)
2. Wait ~10s for the **"Voice ready!"** status
3. Type a sentence and press **Enter**
4. The sentence plays in the cloned voice

---

## Project Structure

```
aac_speak.py        # Main AAC real-time voice cloning UI (Gradio)
benchmark.py        # Synthesis timing benchmarks
indextts/           # IndexTTS-2.5 model inference code
webui.py            # Original IndexTTS WebUI (full features)
checkpoints/        # Model weights (not included, download separately)
```

---

## Use Case

This project is aimed at people with **ALS or other progressive speech loss conditions** who want to bank their voice before losing it. By recording a short sample of their natural voice, they can continue communicating in their own voice using this AAC system even after speech loss.
