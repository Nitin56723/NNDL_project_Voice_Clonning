# -*- coding: utf-8 -*-
import sys, os, queue, threading, time, tempfile, traceback
sys.path.insert(0, '.')
import soundfile as sf
import gradio as gr
try:
    import sounddevice as sd
    SOUNDDEVICE_OK = True
except Exception as e:
    print('sounddevice not available:', e)
    SOUNDDEVICE_OK = False

tts = None
is_loaded = False
reference_wav_path = None
sentence_queue = queue.Queue()
queue_state = []
queue_lock = threading.Lock()
item_counter = 0

def load_model():
    global tts, is_loaded
    from indextts.infer_v2_5 import IndexTTS2
    print('Loading IndexTTS-2.5...')
    tts = IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_bf16=True)
    is_loaded = True
    print('Model ready! Upload your reference voice to begin.')


def convert_to_wav(input_path, max_duration_sec=10.0):
    import librosa
    try:
        print('Converting', os.path.basename(input_path), 'to WAV (trimmed to', max_duration_sec, 'sec)...')
        # duration= param loads ONLY the first N seconds — avoids loading full 1:32 file
        y, sr = librosa.load(input_path, sr=22050, mono=True, duration=max_duration_sec)
        actual_dur = round(len(y) / sr, 1)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, y, 22050)
        print(f'Converted OK -> {tmp.name} ({actual_dur}s used for voice cloning)')
        return tmp.name
    except Exception as e:
        print('Conversion warning:', e)
        return input_path


def play_audio(path):
    """Play a WAV file reliably on Windows using winsound (no hangs)."""
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return True
    except Exception as e:
        print('winsound error:', e)
    if SOUNDDEVICE_OK:
        try:
            data, sr = sf.read(path)
            duration = len(data) / sr
            sd.play(data, sr)
            time.sleep(duration + 0.3)
            sd.stop()
            return True
        except Exception as e:
            print('sounddevice error:', e)
    return False


# Global mutex — only ONE tts.infer() can run at a time (GPU is single-threaded)
synthesis_lock = threading.Lock()
prewarm_ready = False   # True once speaker embedding is cached for current reference

def run_infer(ref_path, text, out_path):
    """Single guarded inference call."""
    tts.infer(
        spk_audio_prompt=ref_path,
        text=text,
        lang='EN',
        output_path=out_path,
        text_normalization=True,   # needed for abbreviated words like 'u', 'r', 'ur'
    )

def synthesis_worker():
    global queue_state
    while True:
        item = sentence_queue.get()
        if item is None:
            break
        item_id, text, ref_path = item
        t_start = time.time()

        # PREWARM item — silent dummy to cache speaker embedding
        if text == '__PREWARM__':
            try:
                with synthesis_lock:
                    dummy = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                    tts.infer(spk_audio_prompt=ref_path, text='Hi.', lang='EN', output_path=dummy)
                    try: os.unlink(dummy)
                    except: pass
                elapsed = time.time() - t_start
                print(f'Speaker cache warm in {elapsed:.1f}s — sentences will now be fast!')
                global prewarm_ready
                prewarm_ready = True
            except Exception as e:
                print('Pre-warm error:', e)
                prewarm_ready = True  # unblock even on error
            finally:
                sentence_queue.task_done()
            continue

        try:
            with queue_lock:
                for s in queue_state:
                    if s['id'] == item_id:
                        s['status'] = 'synthesizing'
                        s['start_time'] = t_start

            out_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            with synthesis_lock:   # prevent concurrent GPU usage
                run_infer(ref_path, text, out_path)

            elapsed = time.time() - t_start
            print(f'Synthesis done in {elapsed:.1f}s: {text[:50]}')

            with queue_lock:
                for s in queue_state:
                    if s['id'] == item_id:
                        s['status'] = 'playing'
                        s['elapsed'] = elapsed

            play_audio(out_path)
            try: os.unlink(out_path)
            except: pass
            with queue_lock:
                queue_state[:] = [s for s in queue_state if s['id'] != item_id]

        except Exception as e:
            print('Error:', traceback.format_exc())
            with queue_lock:
                for s in queue_state:
                    if s['id'] == item_id:
                        s['status'] = 'error'
                        s['error'] = str(e)[:80]
        finally:
            sentence_queue.task_done()

worker = threading.Thread(target=synthesis_worker, daemon=True)
worker.start()
threading.Thread(target=load_model, daemon=True).start()


def build_queue_html():
    with queue_lock:
        items = list(queue_state)
    if not items:
        return '<p style="color:#aaa;padding:12px;text-align:center">Queue is empty</p>'
    html = ''
    for i, s in enumerate(items):
        td = s['text'][:70] + ('...' if len(s['text']) > 70 else '')
        now = time.time()
        status = s['status']
        if status == 'waiting':
            wait = now - s['added_at']
            color = '#f0a500'; badge = 'Waiting ' + str(int(wait)) + 's'; pct = 0
        elif status == 'synthesizing':
            synth_t = now - s.get('start_time', now)
            pct = min(85, int(synth_t / 14.0 * 100))
            color = '#4a90e2'; badge = 'Synthesizing ' + str(int(synth_t)) + 's'
        elif status == 'playing':
            pct = 100; color = '#2ecc71'; badge = 'Playing (took ' + str(round(s.get('elapsed',0),1)) + 's)'
        else:
            pct = 0; color = '#e74c3c'; badge = 'Error: ' + s.get('error','unknown')[:40]
        html += (
            '<div style="background:#1e1e2e;border-radius:8px;padding:12px;margin-bottom:8px;border-left:4px solid ' + color + '">'
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
            '<span style="color:#eee;font-size:13px"><b>#' + str(i+1) + '</b> ' + td + '</span>'
            '<span style="background:' + color + ';color:#000;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:bold">' + badge + '</span>'
            '</div>'
            '<div style="width:100%;background:#333;border-radius:4px;height:8px">'
            '<div style="width:' + str(pct) + '%;background:' + color + ';height:8px;border-radius:4px"></div>'
            '</div></div>'
        )
    return html

def prewarm_speaker_cache(ref_path):
    """Run a silent dummy synthesis to pre-cache speaker embeddings for the reference audio."""
    try:
        print('Pre-warming speaker cache for reference audio...')
        dummy_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        tts.infer(spk_audio_prompt=ref_path, text='Hi.', lang='EN', output_path=dummy_out)
        try: os.unlink(dummy_out)
        except: pass
        print('Speaker cache ready — first sentence will be fast!')
    except Exception as e:
        print('Speaker pre-warm skipped:', e)

def set_reference(audio_path):
    global reference_wav_path, prewarm_ready
    if audio_path is None:
        return 'No voice uploaded'
    prewarm_ready = False   # reset — new voice needs a new warmup
    reference_wav_path = convert_to_wav(audio_path)
    # Queue a prewarm item — runs through the same synthesis queue (no GPU thrash)
    # After prewarm completes, prewarm_ready=True and all sentences use cached embedding
    if is_loaded and reference_wav_path:
        sentence_queue.put((0, '__PREWARM__', reference_wav_path))
        return 'Voice set — warming up speaker cache (~10s), then sentences will be fast!'
    return 'Voice set: ' + os.path.basename(audio_path) + ' (ready)'

def ensure_ends_with_punctuation(text):
    """Add a period if text has no ending punctuation — fixes missing last word."""
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text = text + '.'
    return text

def queue_sentence(text):
    global item_counter
    text = text.strip()
    if not text:
        return '', build_queue_html(), 'Type a sentence and press Enter'
    if not is_loaded:
        return '', build_queue_html(), 'Model still loading, please wait...'
    if reference_wav_path is None:
        return '', build_queue_html(), 'Upload a reference voice first!'
    if not prewarm_ready:
        # Still warming — queue it anyway, it will wait behind the prewarm
        item_counter += 1
        iid = item_counter
        text = ensure_ends_with_punctuation(text)
        with queue_lock:
            queue_state.append({'id': iid, 'text': text, 'status': 'waiting', 'elapsed': 0, 'added_at': time.time()})
        sentence_queue.put((iid, text, reference_wav_path))
        return '', build_queue_html(), 'Queued (will play after voice warms up)...'
    item_counter += 1
    iid = item_counter
    text = ensure_ends_with_punctuation(text)
    with queue_lock:
        queue_state.append({'id': iid, 'text': text, 'status': 'waiting', 'elapsed': 0, 'added_at': time.time()})
    sentence_queue.put((iid, text, reference_wav_path))
    return '', build_queue_html(), 'Queued sentence #' + str(iid)

def stop_all():
    global queue_state
    if SOUNDDEVICE_OK:
        try: sd.stop()
        except: pass
    while not sentence_queue.empty():
        try: sentence_queue.get_nowait(); sentence_queue.task_done()
        except: break
    with queue_lock:
        queue_state.clear()
    return build_queue_html(), 'Stopped and queue cleared'

def refresh():
    return build_queue_html()

def refresh_ref_status():
    """Update reference status box to reflect warmup completion."""
    if reference_wav_path is None:
        return 'No voice uploaded'
    if prewarm_ready:
        return 'Voice ready! Speaker cached — type your sentence and press Enter.'
    return 'Warming up voice... (computing speaker identity, ~10s) — you can type now, it will queue.'

with gr.Blocks(title='AAC Voice Cloning', theme=gr.themes.Soft()) as demo:
    gr.Markdown('# AAC Real-Time Voice Cloning')
    gr.Markdown('**For people who have lost their voice** - Type a sentence, press **Enter** to speak it in your cloned voice. You can keep typing while the current sentence synthesizes and plays.')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### Step 1: Upload Voice Reference')
            ref_audio = gr.Audio(label='Reference Voice (2-10 sec, any format)', type='filepath', sources=['upload', 'microphone'])
            ref_status = gr.Textbox(label='Reference Status', interactive=False, value='No voice loaded')
            ref_audio.change(set_reference, inputs=ref_audio, outputs=ref_status)
        with gr.Column(scale=2):
            gr.Markdown('### Step 2: Type and Speak')
            text_input = gr.Textbox(label='Type here - press Enter to speak', placeholder='Hello, how are you today?', lines=2, autofocus=True)
            with gr.Row():
                speak_btn = gr.Button('Speak (Enter)', variant='primary', scale=3)
                stop_btn = gr.Button('Stop All', variant='stop', scale=1)
            status_box = gr.Textbox(label='Status', value='Loading model...', interactive=False)
    gr.Markdown('### Synthesis Queue')
    queue_html = gr.HTML(value=build_queue_html())
    speak_btn.click(queue_sentence, inputs=text_input, outputs=[text_input, queue_html, status_box])
    text_input.submit(queue_sentence, inputs=text_input, outputs=[text_input, queue_html, status_box])
    stop_btn.click(stop_all, outputs=[queue_html, status_box])
    t = gr.Timer(value=1)
    t.tick(refresh, outputs=queue_html)
    t.tick(refresh_ref_status, outputs=ref_status)

if __name__ == '__main__':
    print('Open: http://127.0.0.1:7861')
    demo.launch(server_name='0.0.0.0', server_port=7861, show_error=True)
