import torch, time, sys
sys.path.insert(0, ".")

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), "GB")
else:
    print("No GPU - running on CPU")

from indextts.infer_v2_5 import IndexTTS2
print("\nLoading model...")
t0 = time.time()
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

for name, param in list(tts.gpt.named_parameters())[:1]:
    print(f"GPT device: {param.device}")

from indextts.utils.examples_downloader import ensure_examples_available
ensure_examples_available()

print("\n--- Short sentence ---")
t1 = time.time()
tts.infer(spk_audio_prompt="examples/voice_01.wav", text="Hello, I use this system to speak for me.", lang="EN", output_path="benchmark_short.wav")
short_time = time.time()-t1
print(f"Time: {short_time:.1f}s")

print("\n--- Medium sentence ---")
t2 = time.time()
tts.infer(spk_audio_prompt="examples/voice_01.wav", text="Voice banking lets people preserve their natural voice for communication devices before losing it.", lang="EN", output_path="benchmark_medium.wav")
medium_time = time.time()-t2
print(f"Time: {medium_time:.1f}s")

print("\n=== RESULT ===")
print(f"Short : {short_time:.1f}s {'(GPU OK)' if short_time<20 else '(SLOW - CPU?)'}")
print(f"Medium: {medium_time:.1f}s {'(GPU OK)' if medium_time<40 else '(SLOW - CPU?)'}")
