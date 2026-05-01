"""
main.py — VSynth: Voice → Whisper → ARPABET → klattsch synth
Dear PyGui desktop application
"""

import dearpygui.dearpygui as dpg
import threading
import queue
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
import tempfile
import os
import time
from collections import deque
from phonemes import text_to_arpabet
from synth import render_to_wav

# ── Constants ──────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
CHANNELS     = 1
WAVEFORM_LEN = 300

# ── Shared state ───────────────────────────────────────────────────────────────
_state = {
    "recording":     False,
    "processing":    False,
    "whisper_ready": False,
    "whisper_model": None,
    "audio_chunks":  [],
    "waveform_buf":  deque([0.0] * WAVEFORM_LEN, maxlen=WAVEFORM_LEN),
    "playback_data": None,
    "playback_sr":   None,
}
_ui_queue = queue.Queue()

# ── Voice parameters ───────────────────────────────────────────────────────────
PARAMS = {
    "baseF0":       dict(label="Base Pitch (Hz)",    default=120.0, min=60,   max=400,  step=1.0,  fmt="%.0f"),
    "rate":         dict(label="Speed (ms/phoneme)", default=110.0, min=50,   max=300,  step=1.0,  fmt="%.0f"),
    "scale":        dict(label="Formant Scale",      default=1.0,   min=0.5,  max=2.0,  step=0.01, fmt="%.2f"),
    "vibratoDepth": dict(label="Vibrato Depth",      default=0.0,   min=0.0,  max=30.0, step=0.5,  fmt="%.1f"),
    "vibratoRate":  dict(label="Vibrato Rate (Hz)",  default=5.0,   min=1.0,  max=15.0, step=0.1,  fmt="%.1f"),
    "aspiration":   dict(label="Breathiness",        default=0.0,   min=0.0,  max=1.0,  step=0.01, fmt="%.2f"),
    "tilt":         dict(label="Spectral Tilt",      default=0.0,   min=-1.0, max=1.0,  step=0.01, fmt="%.2f"),
    "effort":       dict(label="Glottal Effort",     default=0.5,   min=0.0,  max=1.0,  step=0.01, fmt="%.2f"),
}
TOOLTIPS = {
    "baseF0":       "Base pitch (Hz) of the synthesized voice.\nLower = deeper robot.  Higher = squeaky.",
    "rate":         "Duration of each phoneme in ms.\nLower = fast clipped speech.  Higher = slow drawl.",
    "scale":        "Scales all formant frequencies.\n1.0 = neutral.  >1 = bright/thin.  <1 = dark/deep.",
    "vibratoDepth": "Vibrato wobble depth in Hz.  0 = none.\nTry 8-15 for a singing / tremolo effect.",
    "vibratoRate":  "Speed of vibrato oscillation in Hz.\nHigher = faster wobble.",
    "aspiration":   "Breathiness mix (0-1).\nHigh = whispery, airy quality.",
    "tilt":         "Spectral tilt (-1 to +1).\nNegative = harsh/bright.  Positive = muffled/dark.",
    "effort":       "Glottal tension (0-1).\nHigh = strained voice.  Low = relaxed/breathy.",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_voice_params() -> dict:
    return {k: dpg.get_value(f"slider_{k}") for k in PARAMS}

def _log(msg: str):
    """Append a timestamped line to the status log (thread-safe)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _ui_queue.put(("log", line))

def _set_label(tag: str, text: str):
    _ui_queue.put(("set_value", tag, text))

def _flush_ui_queue():
    """Called every frame from the render callback."""
    while not _ui_queue.empty():
        msg = _ui_queue.get_nowait()
        if msg[0] == "log":
            cur = dpg.get_value("status_log")
            lines = cur.split("\n") if cur else []
            # Insert newest messages at the top so it doesn't need to scroll down
            lines.insert(0, msg[1])
            if len(lines) > 60:
                lines = lines[:60]
            dpg.set_value("status_log", "\n".join(lines))
        elif msg[0] == "set_value":
            dpg.set_value(msg[1], msg[2])

# ── Whisper loader (background) ────────────────────────────────────────────────
def _load_whisper():
    _log("Loading Whisper model (base.en)…")
    try:
        from faster_whisper import WhisperModel
        # Use English-only model to prevent hallucinations on music, and int8 for tiny footprint
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        _state["whisper_model"] = model
        _state["whisper_ready"] = True
        _log("Whisper ready.")
        dpg.configure_item("record_btn", enabled=True)
        dpg.configure_item("load_audio_btn", enabled=True)
        dpg.set_value("whisper_status", "Whisper: READY")
    except Exception as e:
        _log(f"Whisper load failed: {e}")
        dpg.set_value("whisper_status", "Whisper: ERROR")

# ── Audio recording ────────────────────────────────────────────────────────────
def _recording_thread():
    _state["audio_chunks"] = []
    _log("Recording…")
    dpg.set_value("rec_indicator", "● REC")

    def callback(indata, frames, time_info, status):
        chunk = indata[:, 0].copy()
        _state["audio_chunks"].append(chunk)
        # Update waveform buffer
        ds = chunk[::max(1, len(chunk) // 30)]
        _state["waveform_buf"].extend(ds.tolist())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", callback=callback):
        while _state["recording"]:
            time.sleep(0.05)

    dpg.set_value("rec_indicator", "")
    _log("Recording stopped.")
    _process_audio()

def _process_audio():
    if not _state["audio_chunks"]:
        _log("No audio captured.")
        return
    _state["processing"] = True
    dpg.configure_item("record_btn", enabled=False)
    _log("Transcribing…")

    audio = np.concatenate(_state["audio_chunks"]).astype(np.float32)

    try:
        model = _state["whisper_model"]
        segments, info = model.transcribe(
            audio, 
            language="en",
            vad_filter=True,
            condition_on_previous_text=False
        )
        text = " ".join([segment.text for segment in segments]).strip()
        _log(f"Transcribed: {text!r}")
        dpg.set_value("transcription_text", text)
        _synthesize(text)
    except Exception as e:
        _log(f"Transcription error: {e}")
        _state["processing"] = False
        dpg.configure_item("record_btn", enabled=True)

def _process_audio_file(path: str):
    _state["processing"] = True
    dpg.configure_item("record_btn", enabled=False)
    _log(f"Transcribing file: {os.path.basename(path)}…")
    try:
        model = _state["whisper_model"]
        segments, info = model.transcribe(
            path, 
            language="en",
            vad_filter=True,
            condition_on_previous_text=False
        )
        text = " ".join([segment.text for segment in segments]).strip()
        _log(f"Transcribed: {text!r}")
        dpg.set_value("transcription_text", text)
        _synthesize(text)
    except Exception as e:
        _log(f"Transcription error: {e}")
        _state["processing"] = False
        dpg.configure_item("record_btn", enabled=True)

def _synthesize(text: str):
    _log("Converting to ARPABET…")
    arpabet = text_to_arpabet(text)
    dpg.set_value("arpabet_text", arpabet)
    _log(f"ARPABET: {arpabet[:80]}{'…' if len(arpabet) > 80 else ''}")
    
    # Run in background to not block UI
    threading.Thread(target=_synthesize_arpabet, args=(arpabet, True), daemon=True).start()

def _synthesize_arpabet(arpabet: str, play_audio: bool = True):
    if not arpabet:
        return
    _log("Synthesizing via klattsch…")

    params = _get_voice_params()
    # baseF0 and rate are ints for klattsch
    params["baseF0"] = int(params["baseF0"])
    params["rate"]   = int(params["rate"])

    try:
        wav_path = render_to_wav(arpabet, params)
        if wav_path and play_audio:
            _log("Playing synth output…")
            _play_wav(wav_path)
        return wav_path
    except Exception as e:
        _log(f"Synth error: {e}")
        return None
    finally:
        _state["processing"] = False
        dpg.configure_item("record_btn", enabled=True)

def _play_wav(path: str):
    try:
        sr, data = scipy.io.wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        sd.play(data, samplerate=sr)
        sd.wait()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

def _on_play_text():
    text = dpg.get_value("transcription_text").strip()
    if text and not text.startswith("(transcription"):
        _state["processing"] = True
        dpg.configure_item("record_btn", enabled=False)
        _synthesize(text)

def _on_play_arpabet():
    arpabet = dpg.get_value("arpabet_text").strip()
    if arpabet and not arpabet.startswith("(ARPABET"):
        _state["processing"] = True
        dpg.configure_item("record_btn", enabled=False)
        threading.Thread(target=_synthesize_arpabet, args=(arpabet, True), daemon=True).start()

def _on_save_wav():
    arpabet = dpg.get_value("arpabet_text").strip()
    if arpabet and not arpabet.startswith("(ARPABET"):
        import tkinter as tk
        from tkinter import filedialog
        import datetime
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        
        now = datetime.datetime.now()
        default_name = now.strftime("%m-%d-%H%M%S.wav")
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV Audio", "*.wav")]
        )
        if not save_path:
            return
            
        _log(f"Saving to {os.path.basename(save_path)}…")
        
        def _save_task():
            wav_path = _synthesize_arpabet(arpabet, play_audio=False)
            if wav_path:
                import shutil
                shutil.copy(wav_path, save_path)
                os.unlink(wav_path)
                _log(f"Saved to {save_path}")
        
        threading.Thread(target=_save_task, daemon=True).start()

def _on_load_audio_click():
    if _state["processing"] or not _state.get("whisper_ready"):
        return
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.ogg *.flac")])
    if path:
        threading.Thread(target=_process_audio_file, args=(path,), daemon=True).start()

# ── Button callbacks ───────────────────────────────────────────────────────────
def _on_record_click():
    if _state["processing"]:
        return
    if not _state["recording"]:
        _state["recording"] = True
        dpg.configure_item("record_btn", label="⏹  STOP")
        # Apply red theme to record button
        dpg.bind_item_theme("record_btn", "theme_btn_stop")
        threading.Thread(target=_recording_thread, daemon=True).start()
    else:
        _state["recording"] = False
        dpg.configure_item("record_btn", label="🎙  RECORD")
        dpg.bind_item_theme("record_btn", "theme_btn_record")

def _on_reset_params():
    for k, p in PARAMS.items():
        dpg.set_value(f"slider_{k}", p["default"])

# ── Render callback (every frame) ─────────────────────────────────────────────
def _render_callback():
    _flush_ui_queue()
    # Update waveform — line_series expects [x_list, y_list]
    wf = list(_state["waveform_buf"])
    xs = list(range(len(wf)))
    dpg.set_value("waveform_series", [xs, wf])
    dpg.set_axis_limits("waveform_y", -1.0, 1.0)
    dpg.set_axis_limits("waveform_x", 0, len(wf))

# ── Theme builder ──────────────────────────────────────────────────────────────
BG        = (10,  10,  18,  255)
PANEL     = (16,  16,  28,  255)
PANEL2    = (22,  22,  38,  255)
GREEN     = (0,   210, 100, 255)
GREEN_DIM = (0,   140, 65,  255)
AMBER     = (255, 165, 50,  255)
RED       = (210, 55,  55,  255)
RED_DIM   = (160, 30,  30,  255)
TEXT      = (220, 220, 230, 255)
TEXT_DIM  = (120, 120, 145, 255)
BORDER    = (40,  40,  60,  255)

def _build_themes():
    # Global theme
    with dpg.theme(tag="global_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       BG)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        PANEL2)
            dpg.add_theme_color(dpg.mvThemeCol_Border,         BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        PANEL2)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (30, 30, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (35, 35, 58, 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (0, 255, 120, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,         (28, 28, 46, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (40, 40, 65, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (50, 50, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header,         (0, 160, 75, 180))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (0, 185, 85, 200))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,    PANEL)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,  (50, 50, 75, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,           TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,      GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,  GREEN)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        BG)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  PANEL)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,    6.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    4.0)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     4.0)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding,    6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    12, 12)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     8,  5)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8,  6)

    # Record button (green)
    with dpg.theme(tag="theme_btn_record"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (0, 130, 60, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 160, 75, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (0, 100, 45, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6.0)

    # Stop button (red)
    with dpg.theme(tag="theme_btn_stop"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (180, 40, 40, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 55, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (140, 25, 25, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          (255, 255, 255, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 6.0)

    # Tooltip "?" button
    with dpg.theme(tag="theme_btn_tip"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (30, 30, 50, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 50, 80, 255))
            dpg.add_theme_color(dpg.mvThemeCol_Text,          (120, 120, 160, 255))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10.0)

    # Plot line theme (green waveform)
    with dpg.theme(tag="theme_plot_line"):
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, GREEN, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.5, category=dpg.mvThemeCat_Plots)

# ── UI builder ─────────────────────────────────────────────────────────────────
def _build_ui():
    W, H = 1120, 720

    with dpg.window(tag="main_window", no_title_bar=True, no_move=True,
                    no_resize=True, no_scrollbar=True,
                    width=W, height=H, pos=(0, 0)):

        # ── Header ──────────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_text("VSYNTH", color=GREEN)
            dpg.add_text("  |  Voice → Whisper → klattsch formant synth",
                         color=TEXT_DIM)
            dpg.add_spacer(width=20)
            dpg.add_text("", tag="whisper_status", color=AMBER)
            dpg.add_spacer(width=10)
            dpg.add_text("", tag="rec_indicator", color=RED)
        dpg.add_separator()
        dpg.add_spacer(height=4)

        # ── Three-column layout ─────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ── LEFT: Mic / Record ──────────────────────────────────────────
            with dpg.child_window(tag="left_panel", width=240, height=-1,
                                  border=True, no_scrollbar=True):
                dpg.add_spacer(height=8)
                dpg.add_text("MIC INPUT", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                dpg.add_button(tag="record_btn",
                               label="🎙  RECORD",
                               width=214, height=56,
                               callback=_on_record_click,
                               enabled=False)
                dpg.bind_item_theme("record_btn", "theme_btn_record")

                dpg.add_spacer(height=8)
                dpg.add_button(tag="load_audio_btn",
                               label="📂 LOAD AUDIO FILE",
                               width=214, height=36,
                               callback=_on_load_audio_click,
                               enabled=False)

                dpg.add_spacer(height=12)
                dpg.add_text("WAVEFORM", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=4)

                with dpg.plot(tag="waveform_plot", height=160, width=214,
                              no_title=True, no_mouse_pos=True,
                              no_box_select=True):
                    dpg.add_plot_axis(dpg.mvXAxis, tag="waveform_x",
                                      no_tick_labels=True, no_gridlines=True,
                                      lock_min=True, lock_max=True)
                    with dpg.plot_axis(dpg.mvYAxis, tag="waveform_y",
                                       no_tick_labels=True, no_gridlines=True,
                                       lock_min=True, lock_max=True):
                        dpg.add_line_series(
                            list(range(WAVEFORM_LEN)),
                            [0.0] * WAVEFORM_LEN,
                            tag="waveform_series",
                        )
                        dpg.bind_item_theme("waveform_series", "theme_plot_line")

                dpg.add_spacer(height=12)
                dpg.add_text("HINT", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_text(
                    "1. Press RECORD\n"
                    "2. Speak clearly\n"
                    "3. Press STOP\n"
                    "4. Hear your voice\n"
                    "   as a robot synth!",
                    color=TEXT_DIM,
                    wrap=210,
                )

            dpg.add_spacer(width=6)

            # ── CENTER: Text displays ───────────────────────────────────────
            with dpg.child_window(tag="center_panel", width=-296, height=-1,
                                  border=True, no_scrollbar=True):
                dpg.add_spacer(height=8)
                dpg.add_text("TRANSCRIPTION", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                dpg.add_input_text(
                    tag="transcription_text",
                    default_value="(transcription appears here)",
                    multiline=True, readonly=False,
                    width=-1, height=100,
                )

                dpg.add_spacer(height=10)
                dpg.add_text("ARPABET PHONEMES", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                dpg.add_input_text(
                    tag="arpabet_text",
                    default_value="(ARPABET string appears here)",
                    multiline=True, readonly=False,
                    width=-1, height=80,
                )
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="▶ Play Text", width=120, callback=_on_play_text)
                    dpg.add_button(label="▶ Play Phonemes", width=120, callback=_on_play_arpabet)
                    dpg.add_button(label="💾 Save WAV", width=90, callback=_on_save_wav)

                dpg.add_spacer(height=10)
                dpg.add_text("STATUS LOG", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=4)
                dpg.add_input_text(
                    tag="status_log",
                    default_value="Starting up…",
                    multiline=True, readonly=True,
                    width=-1, height=-1,
                )

            dpg.add_spacer(width=6)

            # ── RIGHT: Voice controls ───────────────────────────────────────
            with dpg.child_window(tag="right_panel", width=284, height=-1,
                                  border=True, no_scrollbar=True):
                dpg.add_spacer(height=8)
                dpg.add_text("VOICE CONTROLS", color=TEXT_DIM)
                dpg.add_separator()
                dpg.add_spacer(height=8)

                for key, p in PARAMS.items():
                    # Label row with tooltip button
                    with dpg.group(horizontal=True):
                        dpg.add_text(p["label"], color=TEXT)
                        dpg.add_spacer(width=8)
                        tip_btn = f"tipbtn_{key}"
                        dpg.add_button(tag=tip_btn, label="?",
                                       width=22, height=18)
                        dpg.bind_item_theme(tip_btn, "theme_btn_tip")
                        with dpg.tooltip(tip_btn):
                            dpg.add_text(TOOLTIPS[key], wrap=200)

                    dpg.add_slider_float(
                        tag=f"slider_{key}",
                        default_value=p["default"],
                        min_value=p["min"],
                        max_value=p["max"],
                        format=p["fmt"],
                        width=-1,
                    )
                    dpg.add_spacer(height=6)

                dpg.add_separator()
                dpg.add_spacer(height=8)

                # Preset buttons
                dpg.add_text("PRESETS", color=TEXT_DIM)
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Default", width=86,
                                   callback=_on_reset_params)
                    dpg.add_button(label="Deep Robot", width=86,
                                   callback=lambda: _apply_preset("deep"))
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Alien", width=86,
                                   callback=lambda: _apply_preset("alien"))
                    dpg.add_button(label="Whispery", width=86,
                                   callback=lambda: _apply_preset("whispery"))

# ── Presets ────────────────────────────────────────────────────────────────────
PRESETS = {
    "deep":     dict(baseF0=70,  rate=130, scale=0.8, vibratoDepth=0,   vibratoRate=5, aspiration=0,    tilt=0.2,  effort=0.7),
    "alien":    dict(baseF0=220, rate=90,  scale=1.4, vibratoDepth=12,  vibratoRate=8, aspiration=0.1,  tilt=-0.3, effort=0.4),
    "whispery": dict(baseF0=130, rate=120, scale=1.0, vibratoDepth=0,   vibratoRate=5, aspiration=0.8,  tilt=0.0,  effort=0.1),
}

def _apply_preset(name: str):
    p = PRESETS.get(name, {})
    for k, v in p.items():
        try:
            dpg.set_value(f"slider_{k}", float(v))
        except Exception:
            pass

# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    dpg.create_context()
    _build_themes()
    _build_ui()

    dpg.bind_theme("global_theme")
    dpg.set_primary_window("main_window", True)

    dpg.create_viewport(
        title="VSynth — Voice to Klatt Synth",
        width=1120, height=720,
        resizable=False,
        small_icon="",
        large_icon="",
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Load Whisper in background
    dpg.set_value("whisper_status", "Whisper: LOADING…")
    threading.Thread(target=_load_whisper, daemon=True).start()

    # Main loop with per-frame callback
    while dpg.is_dearpygui_running():
        _render_callback()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
