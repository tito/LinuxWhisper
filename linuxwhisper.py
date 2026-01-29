#!/usr/bin/env python3
"""
LinuxWhisper - Voice Assistant for Linux
=========================================

A voice-to-text and AI assistant tool that integrates with OpenAI-compatible APIs
(LocalAI, Groq, OpenAI, etc.) for transcription, chat completion, vision analysis, and text-to-speech.

ARCHITECTURE OVERVIEW
---------------------
Section 1: Imports
Section 2: Configuration (Config dataclass - all constants)
Section 3: State Management (AppState dataclass - all mutable state)
Section 4: Services (AudioService, AIService, TTSService, ClipboardService)
Section 5: Managers (HistoryManager, ChatManager)
Section 6: UI Components (GtkOverlay, ChatOverlay)
Section 7: System Tray (TrayManager)
Section 8: Keyboard Handler (KeyboardHandler)
Section 9: Main Entry Point

ADDING A NEW MODE
-----------------
1. Add mode config to Config.MODES dict
2. Add key mapping to KeyboardHandler.KEY_MAPPINGS
3. Create handler method in ModeHandler._handle_<mode>
4. Register in ModeHandler.HANDLERS dispatch dict

HOTKEYS
-------
F3:  Dictation (speech-to-text, types at cursor)
F4:  AI Chat (voice question → AI response)
F7:  AI Rewrite (select text + voice instruction → rewritten text)
F8:  Vision (screenshot + voice question → AI analysis)
F9:  Toggle chat overlay pin mode
F10: Toggle TTS (text-to-speech for AI responses)
"""

# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================
from __future__ import annotations

import argparse
import base64
import io
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)

import cairo
import gi
import numpy as np
import sounddevice as sd
import structlog
from dotenv import load_dotenv
from openai import OpenAI
from scipy.io.wavfile import write as wav_write

# Load .env file from script directory
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

gi.require_version('Gtk', '3.0')
gi.require_version('AyatanaAppIndicator3', '0.1')
gi.require_version('WebKit2', '4.1')
from gi.repository import AyatanaAppIndicator3 as AppIndicator
from gi.repository import Gdk, Gio, GLib, Gtk, WebKit2

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()


# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration.
    
    All constants are centralized here for easy modification.
    To change a setting, edit the default value below.
    """
    # --- Audio Settings ---
    SAMPLE_RATE: int = 44100
    
    # --- History Limits ---
    MAX_TOKENS: int = 32000
    ANSWER_HISTORY_LIMIT: int = 15
    CHAT_MESSAGE_LIMIT: int = 20
    CHAT_AUTO_HIDE_SEC: int = 5
    
    # --- AI Models (defaults for LocalAI, override via .env) ---
    MODEL_CHAT: str = os.environ.get("MODEL_CHAT", "gpt-4")
    MODEL_VISION: str = os.environ.get("MODEL_VISION", "gpt-4-vision-preview")
    MODEL_WHISPER: str = os.environ.get("MODEL_WHISPER", "whisper-1")
    MODEL_TTS: str = os.environ.get("MODEL_TTS", "tts-1")
    
    # --- TTS Voices (LocalAI/Piper voices - adjust based on installed models) ---
    TTS_VOICES: Tuple[str, ...] = ("alloy", "echo", "fable", "onyx", "nova", "shimmer")
    TTS_DEFAULT_VOICE: str = "alloy"
    TTS_MAX_CHARS: int = 4000
    
    # --- Temp File Paths ---
    TEMP_SCREEN_PATH: str = "/tmp/temp_screen.png"
    TEMP_TTS_PATH: str = "/tmp/linuxwhisper_tts.wav"
    
    # --- System Prompt ---
    SYSTEM_PROMPT: str = (
        "Act as a compassionate assistant. Base your reasoning on the principles of "
        "Nonviolent Communication and A Course in Miracles. Apply these frameworks as "
        "your underlying logic without explicitly naming them or forcing them. Let your "
        "output be grounded, clear, and highly concise. Return ONLY the direct response."
    )
    
    # --- Mode Definitions (icon, overlay text, colors) ---
    MODES: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "dictation":  {"icon": "🎙️", "text": "Listening...",    "bg": "#1a1a2e", "fg": "#00d4ff"},
        "ai":         {"icon": "🤖", "text": "AI Listening...", "bg": "#1a1a2e", "fg": "#a855f7"},
        "ai_rewrite": {"icon": "✍️", "text": "Rewrite Mode...", "bg": "#1a1a2e", "fg": "#22c55e"},
        "vision":     {"icon": "📸", "text": "Vision Mode...",  "bg": "#1a1a2e", "fg": "#f59e0b"},
    })

    # format: "id": (Label_for_UI, shortcut_trigger, description)
    # Shortcut triggers use XDG GlobalShortcuts format
    HOTKEY_DEFS: Dict[str, Tuple[str, str, str]] = field(default_factory=lambda: {
        "linuxwhisper-dictation":  ("F3",  "F3",  "Dictation - speech to text"),
        "linuxwhisper-ai":         ("F4",  "F4",  "AI Chat assistant"),
        "linuxwhisper-ai_rewrite": ("F7",  "F7",  "AI Rewrite selected text"),
        "linuxwhisper-vision":     ("F8",  "F8",  "Vision - screenshot analysis"),
        "linuxwhisper-pin":        ("F9",  "F9",  "Toggle chat pin"),
        "linuxwhisper-tts":        ("F10", "F10", "Toggle text-to-speech"),
    })


# Global config instance
CFG = Config()


# ============================================================================
# SECTION 3: STATE MANAGEMENT
# ============================================================================
@dataclass
class AppState:
    """
    Mutable application state.
    
    All runtime state is centralized here for clarity and debugging.
    Reset by creating a new instance: STATE = AppState()
    """
    # --- Recording State ---
    recording: bool = False
    current_mode: Optional[str] = None
    audio_buffer: List[np.ndarray] = field(default_factory=list)
    stream: Optional[sd.InputStream] = None
    viz_queue: queue.Queue = field(default_factory=queue.Queue)
    
    # --- UI Windows ---
    overlay_window: Optional[Any] = None  # GtkOverlay instance
    chat_overlay_window: Optional[Any] = None  # ChatOverlay instance
    
    # --- Chat State ---
    chat_messages: List[Dict[str, str]] = field(default_factory=list)
    chat_pinned: bool = False
    chat_hide_timer: Optional[int] = None
    
    # --- History ---
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    answer_history: List[Dict[str, str]] = field(default_factory=list)
    
    # --- TTS ---
    tts_enabled: bool = True  # Enabled by default
    tts_voice: str = CFG.TTS_DEFAULT_VOICE
    
    # --- System Tray ---
    indicator: Optional[AppIndicator.Indicator] = None
    gtk_menu: Optional[Gtk.Menu] = None
    
    # --- UI Persistence ---
    last_chat_position: Optional[Tuple[int, int]] = None


# Global state instance
STATE = AppState()



# ============================================================================
# SECTION 4: API CLIENT INITIALIZATION
# ============================================================================
def _init_openai_client() -> OpenAI:
    """Initialize OpenAI-compatible API client."""
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key and "openai.com" in base_url:
        log.error("OPENAI_API_KEY missing - check your .env file")
        sys.exit(1)

    # For local APIs like LocalAI, key can be anything
    return OpenAI(base_url=base_url, api_key=api_key or "unused")


API_CLIENT = _init_openai_client()


# ============================================================================
# SECTION 5: UTILITY DECORATORS
# ============================================================================
def safe_execute(operation: str) -> Callable:
    """
    Decorator for consistent error handling.
    
    Usage:
        @safe_execute("Transcription")
        def transcribe_audio(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error("operation_failed", operation=operation, error=str(e))
                return None
        return wrapper
    return decorator


def run_on_main_thread(func: Callable) -> Callable:
    """Decorator to schedule function execution on GTK main thread."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        GLib.idle_add(lambda: func(*args, **kwargs))
    return wrapper


# ============================================================================
# SECTION 6: SERVICES
# ============================================================================

# --- Audio Service ---
class AudioService:
    """Audio recording and transcription service."""
    
    @staticmethod
    def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Capture audio chunks into buffer while recording."""
        if not STATE.recording:
            return
        
        data_copy = indata.copy()
        STATE.audio_buffer.append(data_copy)
        
        # Send downsampled data to visualization queue (skip if full)
        try:
            if STATE.viz_queue.qsize() < 5:
                flat_data = data_copy[:, 0][::10]  # Downsample
                STATE.viz_queue.put_nowait(flat_data)
        except Exception:
            pass
    
    @staticmethod
    def start_recording() -> None:
        """Start audio recording stream."""
        STATE.audio_buffer = []
        AudioService._clear_viz_queue()
        STATE.stream = sd.InputStream(
            samplerate=CFG.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=AudioService.audio_callback
        )
        STATE.stream.start()
        STATE.recording = True
    
    @staticmethod
    def stop_recording() -> Optional[np.ndarray]:
        """Stop recording and return audio data."""
        STATE.recording = False
        if STATE.stream:
            STATE.stream.stop()
            STATE.stream.close()
            STATE.stream = None
        
        if STATE.audio_buffer:
            return np.concatenate(STATE.audio_buffer, axis=0)
        return None
    
    @staticmethod
    def _clear_viz_queue() -> None:
        """Clear the visualization queue."""
        while not STATE.viz_queue.empty():
            try:
                STATE.viz_queue.get_nowait()
            except queue.Empty:
                break
    
    @staticmethod
    @safe_execute("Transcription")
    def transcribe(audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio using Whisper."""
        duration_sec = len(audio_data) / CFG.SAMPLE_RATE
        log.info("transcribe_start", model=CFG.MODEL_WHISPER, audio_duration_sec=round(duration_sec, 1))

        wav_buffer = io.BytesIO()
        wav_buffer.name = "audio.wav"
        wav_write(wav_buffer, CFG.SAMPLE_RATE, audio_data)
        wav_buffer.seek(0)

        transcript = API_CLIENT.audio.transcriptions.create(
            model=CFG.MODEL_WHISPER,
            file=wav_buffer
        )
        result = transcript.text.strip()
        log.info("transcribe_complete", text_length=len(result), text_preview=result[:80] if result else "")
        return result


# --- AI Service ---
class AIService:
    """AI chat and vision completion service."""
    
    @staticmethod
    def build_messages(user_content: str) -> List[Dict[str, Any]]:
        """Build API messages with system prompt and conversation history."""
        messages = [{"role": "system", "content": CFG.SYSTEM_PROMPT}]
        messages.extend(STATE.conversation_history)
        messages.append({"role": "user", "content": user_content})
        return messages
    
    @staticmethod
    @safe_execute("AI Chat")
    def chat(prompt: str) -> Optional[str]:
        """Send chat completion request."""
        log.info("chat_start", model=CFG.MODEL_CHAT, prompt_length=len(prompt))
        messages = AIService.build_messages(prompt)
        response = API_CLIENT.chat.completions.create(
            model=CFG.MODEL_CHAT,
            messages=messages
        )
        result = response.choices[0].message.content
        log.info("chat_complete", response_length=len(result) if result else 0)
        return result
    
    @staticmethod
    @safe_execute("AI Vision")
    def vision(prompt: str, image_base64: str) -> Optional[str]:
        """Send vision completion request with image."""
        log.info("vision_start", model=CFG.MODEL_VISION, prompt_length=len(prompt), image_size_kb=len(image_base64) // 1024)
        messages = AIService.build_messages(prompt)
        # Replace last user message with multimodal content
        messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }
        response = API_CLIENT.chat.completions.create(
            model=CFG.MODEL_VISION,
            messages=messages
        )
        result = response.choices[0].message.content
        log.info("vision_complete", response_length=len(result) if result else 0)
        return result


# --- TTS Service ---
class TTSService:
    """Text-to-speech service using Groq Orpheus."""
    
    @staticmethod
    def speak(text: str) -> None:
        """Convert text to speech and play (async)."""
        if not STATE.tts_enabled or not text:
            return

        def _speak_thread():
            try:
                log.info("tts_start", model=CFG.MODEL_TTS, voice=STATE.tts_voice, text_length=min(len(text), CFG.TTS_MAX_CHARS))
                response = API_CLIENT.audio.speech.create(
                    model=CFG.MODEL_TTS,
                    voice=STATE.tts_voice,
                    input=text[:CFG.TTS_MAX_CHARS],
                    response_format="wav"
                )
                response.write_to_file(CFG.TEMP_TTS_PATH)
                log.info("tts_complete", output_file=CFG.TEMP_TTS_PATH)
                subprocess.run(["aplay", "-q", CFG.TEMP_TTS_PATH], capture_output=True)
            except Exception as e:
                log.error("tts_failed", error=str(e))

        threading.Thread(target=_speak_thread, daemon=True).start()
    
    @staticmethod
    def toggle() -> None:
        """Toggle TTS enabled state."""
        STATE.tts_enabled = not STATE.tts_enabled
        ChatManager.refresh_overlay()


# --- Clipboard Service ---
class ClipboardService:
    """Clipboard operations for typing and pasting text (Wayland native)."""

    @staticmethod
    def type_text(text: str) -> None:
        """Paste text at cursor via clipboard (Wayland)."""
        if not text:
            return

        # Save original clipboard
        try:
            result = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True)
            original = result.stdout if result.returncode == 0 else None
        except Exception:
            original = None

        # Add leading space to prevent word merging
        clean_text = f" {text.strip()}" if not text.startswith(" ") else text

        # Copy to clipboard and paste via wtype
        subprocess.run(["wl-copy", "--", clean_text], check=False)
        subprocess.run(["wtype", "-M", "ctrl", "-P", "v", "-m", "ctrl", "-p", "v"], check=False)

        # Restore original clipboard after short delay
        time.sleep(0.15)
        if original is not None:
            try:
                subprocess.run(["wl-copy", "--", original], check=False)
            except Exception:
                pass

    @staticmethod
    def copy_selected() -> str:
        """Copy currently selected text and return it (Wayland)."""
        # Trigger copy via wtype
        subprocess.run(["wtype", "-M", "ctrl", "-P", "c", "-m", "ctrl", "-p", "c"], check=False)
        time.sleep(0.1)
        result = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    @staticmethod
    def paste_text(text: str) -> None:
        """Paste text directly via clipboard (Wayland)."""
        subprocess.run(["wl-copy", "--", text], check=False)
        subprocess.run(["wtype", "-M", "ctrl", "-P", "v", "-m", "ctrl", "-p", "v"], check=False)


# --- Image Service ---
class ImageService:
    """Screenshot and image encoding service (Wayland native)."""

    @staticmethod
    @safe_execute("Screenshot")
    def take_screenshot() -> Optional[str]:
        """Take screenshot and return base64 encoded string (using grim for Wayland)."""
        # Use grim for Wayland screenshot
        subprocess.run(["grim", CFG.TEMP_SCREEN_PATH], check=True)
        with open(CFG.TEMP_SCREEN_PATH, "rb") as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        os.remove(CFG.TEMP_SCREEN_PATH)
        return encoded


# ============================================================================
# SECTION 7: MANAGERS
# ============================================================================

# --- History Manager ---
class HistoryManager:
    """Manages conversation and answer history."""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars per token)."""
        return len(text) // 4
    
    @staticmethod
    def get_history_tokens() -> int:
        """Calculate total tokens in conversation history."""
        return sum(
            HistoryManager.estimate_tokens(msg["content"])
            for msg in STATE.conversation_history
        )
    
    @staticmethod
    def trim_history() -> None:
        """Remove oldest messages until under token limit."""
        while (HistoryManager.get_history_tokens() > CFG.MAX_TOKENS 
               and STATE.conversation_history):
            STATE.conversation_history.pop(0)
    
    @staticmethod
    def add_message(role: str, content: str) -> None:
        """Add message to conversation history and trim if needed."""
        STATE.conversation_history.append({"role": role, "content": content})
        HistoryManager.trim_history()
    
    @staticmethod
    def add_answer(text: str) -> None:
        """Add answer to tray history."""
        timestamp = time.strftime("%H:%M")
        STATE.answer_history.insert(0, {"text": text, "timestamp": timestamp})
        
        # Trim to limit
        if len(STATE.answer_history) > CFG.ANSWER_HISTORY_LIMIT:
            STATE.answer_history = STATE.answer_history[:CFG.ANSWER_HISTORY_LIMIT]
        
        TrayManager.update_menu()
    
    @staticmethod
    def clear_all() -> None:
        """Clear all history."""
        STATE.answer_history = []
        STATE.conversation_history = []
        STATE.chat_messages = []
        TrayManager.update_menu()
        ChatManager.refresh_overlay()


# --- Chat Manager ---
class ChatManager:
    """Manages chat overlay state and messages."""
    
    @staticmethod
    def add_message(role: str, text: str) -> None:
        """Add message to chat overlay."""
        STATE.chat_messages.append({"role": role, "text": text})
        
        # Trim to limit
        if len(STATE.chat_messages) > CFG.CHAT_MESSAGE_LIMIT:
            STATE.chat_messages = STATE.chat_messages[-CFG.CHAT_MESSAGE_LIMIT:]
        
        ChatManager.refresh_overlay()
    
    @staticmethod
    def toggle_pin() -> None:
        """Toggle chat overlay pin mode."""
        STATE.chat_pinned = not STATE.chat_pinned
        
        if not STATE.chat_pinned and STATE.chat_overlay_window:
            ChatManager._cancel_timer()
            STATE.chat_overlay_window.start_fade_out(callback=ChatManager._destroy)
        else:
            ChatManager.refresh_overlay()
    
    @staticmethod
    @run_on_main_thread
    def refresh_overlay(status_text: Optional[str] = None) -> None:
        """Refresh chat overlay on main thread."""
        ChatManager._show_overlay(status_text)
    
    @staticmethod
    def _show_overlay(status_text: Optional[str] = None) -> None:
        """Show or update chat overlay."""
        ChatManager._cancel_timer()
        
        if not STATE.chat_overlay_window:
            STATE.chat_overlay_window = ChatOverlay()
        elif STATE.chat_overlay_window.fade_out_active:
            STATE.chat_overlay_window.start_fade_in()
        
        STATE.chat_overlay_window.update_content(
            STATE.chat_messages,
            status_text,
            is_pinned=STATE.chat_pinned,
            is_tts=STATE.tts_enabled
        )
        
        if not STATE.chat_pinned:
            STATE.chat_hide_timer = GLib.timeout_add_seconds(
                CFG.CHAT_AUTO_HIDE_SEC,
                ChatManager._auto_hide
            )
    
    @staticmethod
    def _auto_hide() -> bool:
        """Auto-hide callback."""
        STATE.chat_hide_timer = None
        if not STATE.chat_pinned and STATE.chat_overlay_window:
            STATE.chat_overlay_window.start_fade_out(callback=ChatManager._destroy)
        return False
    
    @staticmethod
    def _cancel_timer() -> None:
        """Cancel auto-hide timer if active."""
        if STATE.chat_hide_timer:
            GLib.source_remove(STATE.chat_hide_timer)
            STATE.chat_hide_timer = None
    
    @staticmethod
    def _destroy() -> None:
        """Destroy chat overlay window."""
        if STATE.chat_overlay_window:
            STATE.chat_overlay_window.close()
            STATE.chat_overlay_window = None


# ============================================================================
# SECTION 8: UI COMPONENTS
# ============================================================================

# --- Recording Overlay ---
class GtkOverlay(Gtk.Window):
    """Floating recording overlay with waveform visualization."""

    def __init__(self, mode: str):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        self.mode = mode
        self.config = CFG.MODES.get(mode, CFG.MODES["dictation"])
        self._setup_window()
        self._setup_ui()
        self.show_all()

    def _setup_window(self) -> None:
        """Configure window properties."""
        self.set_app_paintable(True)
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_type_hint(Gdk.WindowTypeHint.NOTIFICATION)

        # Enable transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual and screen.is_composited():
            self.set_visual(visual)
        
        # Position at bottom center
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        geometry = monitor.get_geometry()
        w, h = 220, 60
        x = (geometry.width - w) // 2
        y = geometry.height - h - 80
        self.move(x, y)
        self.set_default_size(w, h)
    
    def _setup_ui(self) -> None:
        """Setup drawing area and animation."""
        self.drawing_area = Gtk.DrawingArea()
        self.drawing_area.connect("draw", self._on_draw)
        self.add(self.drawing_area)
        self.timeout_id = GLib.timeout_add(40, self._animate)
    
    def _on_draw(self, widget: Gtk.DrawingArea, cr: cairo.Context) -> None:
        """Draw overlay content."""
        w, h = widget.get_allocated_width(), widget.get_allocated_height()
        bg_rgb = self._hex_to_rgb(self.config["bg"])
        fg_rgb = self._hex_to_rgb(self.config["fg"])
        
        # Background rounded rect
        self._draw_rounded_rect(cr, w, h, 15)
        cr.set_source_rgba(*bg_rgb, 0.92)
        cr.fill()
        
        # Icon
        cr.set_source_rgb(*fg_rgb)
        cr.select_font_face("Ubuntu", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        cr.set_font_size(20)
        ext = cr.text_extents(self.config["icon"])
        cr.move_to(30 - ext.width / 2, h / 2 + ext.height / 2)
        cr.show_text(self.config["icon"])
        
        # Text
        cr.set_font_size(10)
        cr.select_font_face("Ubuntu", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ext = cr.text_extents(self.config["text"])
        cr.move_to(110 - ext.width / 2, 20)
        cr.show_text(self.config["text"])
        
        # Waveform
        self._draw_waveform(cr, 60, 210, 45, fg_rgb)
    
    def _draw_rounded_rect(self, cr: cairo.Context, w: int, h: int, r: int) -> None:
        """Draw rounded rectangle path."""
        cr.new_sub_path()
        cr.arc(w - r, r, r, -math.pi / 2, 0)
        cr.arc(w - r, h - r, r, 0, math.pi / 2)
        cr.arc(r, h - r, r, math.pi / 2, math.pi)
        cr.arc(r, r, r, math.pi, 3 * math.pi / 2)
        cr.close_path()
    
    def _draw_waveform(self, cr: cairo.Context, x1: int, x2: int, cy: int, color: Tuple[float, ...]) -> None:
        """Draw audio waveform bars."""
        # Get latest audio data
        data = None
        while not STATE.viz_queue.empty():
            try:
                data = STATE.viz_queue.get_nowait()
            except queue.Empty:
                break
        
        cr.set_source_rgb(*color)
        cr.set_line_width(3)
        cr.set_line_cap(cairo.LINE_CAP_ROUND)
        
        if data is not None and len(data) > 0:
            width = x2 - x1
            num_bars = 30
            step = max(1, len(data) // num_bars)
            bar_width = width / num_bars
            max_height = 15
            
            for i in range(num_bars):
                idx = i * step
                if idx >= len(data):
                    break
                chunk = data[idx:idx + step]
                amp = np.max(np.abs(chunk)) if len(chunk) > 0 else 0
                bar_h = max(1, min(max_height, amp * 40 * max_height))
                
                x = x1 + i * bar_width
                cr.move_to(x, cy - bar_h)
                cr.line_to(x, cy + bar_h)
                cr.stroke()
        else:
            # Idle line
            cr.set_line_width(2)
            cr.set_source_rgb(0.33, 0.33, 0.33)
            cr.move_to(x1, cy)
            cr.line_to(x2, cy)
            cr.stroke()
    
    def _animate(self) -> bool:
        """Animation tick."""
        self.drawing_area.queue_draw()
        return True
    
    @staticmethod
    def _hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
        """Convert hex color to RGB tuple (0-1 range)."""
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    
    def close(self) -> None:
        """Clean up and destroy."""
        if self.timeout_id:
            GLib.source_remove(self.timeout_id)
            self.timeout_id = None
        self.destroy()


class OverlayManager:
    """Manages recording overlay visibility."""
    
    @staticmethod
    @run_on_main_thread
    def show(mode: str) -> None:
        """Show overlay for given mode."""
        OverlayManager._show_impl(mode)
    
    @staticmethod
    def _show_impl(mode: str) -> None:
        if STATE.overlay_window:
            try:
                STATE.overlay_window.close()
            except Exception:
                pass
        STATE.overlay_window = GtkOverlay(mode)
    
    @staticmethod
    @run_on_main_thread
    def hide() -> None:
        """Hide overlay."""
        OverlayManager._hide_impl()
    
    @staticmethod
    def _hide_impl() -> None:
        if STATE.overlay_window:
            STATE.overlay_window.close()
            STATE.overlay_window = None


# --- Chat Overlay HTML Template ---
CHAT_CSS = '''
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  height: 100%;
  background: transparent !important;
  font-family: 'Inter', 'Ubuntu', system-ui, -apple-system, sans-serif;
  color: #e2e8f0;
  font-size: 14px;
  line-height: 1.6;
  overflow: hidden; /* Hide native window scrollbar */
}

/* Rounded Glass Window Container */
.chat-window {
  display: flex; 
  flex-direction: column;
  height: 100%;
  background-color: rgba(20, 20, 25, 0.75);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  overflow: hidden;
  margin: 0; position: relative;
}

/* Drag Handle */
.drag-handle {
  position: absolute; top: 0; left: 0; width: 100%; height: 60px;
  z-index: 5; cursor: move; -webkit-app-region: drag;
}

/* Scroll Area */
.chat-scroll-area {
  flex: 1;
  overflow-y: auto;
  scroll-behavior: smooth;
  padding-bottom: 10px;
  z-index: 10; /* Above drag handle */
  position: relative;
}
/* Custom Scrollbar for inner area */
.chat-scroll-area::-webkit-scrollbar { width: 6px; }
.chat-scroll-area::-webkit-scrollbar-track { background: transparent; }
.chat-scroll-area::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 3px; }
.chat-scroll-area::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.25); }

/* HUD / Pin Hint - Static Header */
.pin-hint {
  flex-shrink: 0; /* Keep it fixed height */
  width: fit-content;
  margin: 12px auto 4px auto;
  background: rgba(0, 0, 0, 0.4);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  color: #94a3b8;
  padding: 5px 14px;
  font-size: 11px; font-weight: 600;
  border-radius: 20px;
  z-index: 20; /* Above drag handle */
  display: flex; gap: 10px; align-items: center; justify-content: center;
  transition: opacity 0.3s;
  cursor: default; position: relative;
}
.pin-hint a { color: inherit; text-decoration: none; opacity: 0.8; transition: opacity 0.2s; cursor: pointer; }
.pin-hint a:hover { opacity: 1; color: #fff; }

/* Chat Content */
.chat-container {
  display: flex; flex-direction: column;
  padding: 10px 16px 20px 16px;
}

/* Messages */
.message-wrapper {
  display: flex;
  margin-bottom: 14px;
  animation: slideFadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  opacity: 0;
  transform: translateY(15px);
}
.message-wrapper.user { justify-content: flex-end; }
.message-wrapper.assistant { justify-content: flex-start; }

@keyframes slideFadeIn {
  to { opacity: 1; transform: translateY(0); }
}

.message {
  max-width: 86%;
  padding: 10px 16px;
  border-radius: 14px;
  position: relative;
  word-wrap: break-word;
}

/* User Bubble - Subtler Slate Gradient */
.user .message {
  background: linear-gradient(135deg, #475569 0%, #334155 100%);
  color: #f1f5f9;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Assistant Bubble */
.assistant .message {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #e2e8f0;
}

/* Copy Button */
.copy-btn {
  background: none; border: none; cursor: pointer;
  padding: 6px; margin: 0 4px;
  opacity: 0.6; /* Always visible */
  transition: opacity 0.2s;
  align-self: center;
  color: #64748b;
  z-index: 20; /* Ensure Clickable */
}
.message-wrapper:hover .copy-btn { opacity: 1; }
.copy-btn:hover { opacity: 1; color: #e2e8f0; transform: scale(1.05); }
.copy-btn svg { width: 15px; height: 15px; fill: currentColor; }
.copy-btn.copied { opacity: 1; color: #4ade80; }
.user .copy-btn { order: -1; }

.text code {
  background: rgba(0,0,0,0.3); padding: 2px 5px; border-radius: 4px;
  font-family: 'SF Mono', monospace; font-size: 0.9em; color: #f87171;
}
.text pre {
  background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05);
  color: #cbd5e1; padding: 12px; border-radius: 10px;
  overflow-x: auto; margin: 8px 0; font-family: 'SF Mono', monospace;
  font-size: 0.85em;
}
.text strong { font-weight: 600; color: #fff; }
.status {
  align-self: center; background: rgba(255,255,255,0.05); color: #94a3b8;
  font-size: 11px; padding: 3px 10px; border-radius: 10px;
  margin: 10px 0; border: 1px solid rgba(255,255,255,0.05);
}
'''

CHAT_JS = '''
const copyIcon = '<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>';
const checkIcon = '<svg viewBox="0 0 24 24"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>';

function copyText(btn, index) {
  // Use custom protocol to let Python handle clipboard safely
  window.location.href = "copy://" + index;
  
  // Optimistic UI update
  btn.innerHTML = checkIcon;
  btn.classList.add('copied');
  setTimeout(() => { btn.innerHTML = copyIcon; btn.classList.remove('copied'); }, 1500);
}

function signalDrag() {
  document.title = "Action:Drag:" + Date.now();
}

// Scroll Logic: Only if >= 2 assistant answers
function checkScroll(smooth=true) {
  const scrollArea = document.getElementById('scroll-area');
  const answers = document.querySelectorAll('.message-wrapper.assistant');
  
  if (scrollArea && answers.length >= 2) {
    const opts = smooth ? { top: scrollArea.scrollHeight, behavior: 'smooth' } : { top: scrollArea.scrollHeight };
    scrollArea.scrollTo(opts);
  }
}

// Observe new messages
const chat = document.getElementById('chat');
if (chat) {
  new MutationObserver(() => checkScroll(true)).observe(chat, { childList: true, subtree: true });
}

window.onload = () => checkScroll(false);
'''

CHAT_HTML_TEMPLATE = f'''<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><style>{CHAT_CSS}</style></head>
<body>
<div class="chat-window">
  <div class="drag-handle" onmousedown="signalDrag()"></div>
  {{pin_hint}}
  <div class="chat-scroll-area" id="scroll-area">
    <div id="chat" class="chat-container">{{messages}}</div>
  </div>
</div>
<script>{CHAT_JS}</script>
</body>
</html>'''


class ChatOverlay(Gtk.Window):
    """Chat overlay using WebKit2."""
    
    def __init__(self):
        super().__init__(type=Gtk.WindowType.TOPLEVEL)
        self._setup_window()
        self._setup_webview()
        self._init_animation()
        self.show_all()
    
    def _setup_window(self) -> None:
        """Configure window properties."""
        self.set_decorated(False)
        self.set_keep_above(True)
        self.set_skip_taskbar_hint(True)
        self.set_skip_pager_hint(True)
        self.set_app_paintable(True)
        self.set_type_hint(Gdk.WindowTypeHint.UTILITY)
        
        # Transparency
        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual and screen.is_composited():
            self.set_visual(visual)
        
        # Position at right edge
        display = Gdk.Display.get_default()
        monitor = display.get_primary_monitor() or display.get_monitor(0)
        geometry = monitor.get_geometry()
        w, h = 340, 450
        x = geometry.x + geometry.width - w - 20
        y = geometry.y + (geometry.height - h) // 2
        self.move(x, y)
        self.set_default_size(w, h)
    
    def _setup_webview(self) -> None:
        """Setup WebKit2 webview."""
        self.webview = WebKit2.WebView()
        self.webview.set_background_color(Gdk.RGBA(0, 0, 0, 0))
        settings = self.webview.get_settings()
        settings.set_enable_javascript(True)
        self.webview.connect("decide-policy", self._on_policy_decision)
        self.webview.connect("notify::title", self._on_title_changed)
        self.add(self.webview)
    
    def _on_title_changed(self, webview, pspec) -> None:
        """Handle title changes for drag signals."""
        title = webview.get_title()
        if title and title.startswith("Action:Drag"):
            # Get actual pointer position to prevent jumping
            display = self.get_display()
            seat = display.get_default_seat()
            pointer = seat.get_pointer()
            screen, x, y = pointer.get_position()
            
            self.begin_move_drag(1, x, y, Gtk.get_current_event_time())

    def _init_animation(self) -> None:
        """Initialize fade animation state."""
        self.opacity_value = 0.0
        self.fade_in_active = False
        self.fade_out_active = False
        self.fade_timer = None
        self.fade_callback = None
        self.start_fade_in()
    
    def start_fade_in(self) -> None:
        """Start fade-in animation."""
        self.fade_out_active = False
        self.fade_in_active = True
        self.opacity_value = 0.0
        self._cancel_fade_timer()
        self.fade_timer = GLib.timeout_add(16, self._fade_in_step)
    
    def _fade_in_step(self) -> bool:
        """Fade-in animation step."""
        self.opacity_value = min(1.0, self.opacity_value + 0.1)
        try:
            self.set_opacity(self.opacity_value)
        except Exception:
            pass
        if self.opacity_value >= 1.0:
            self.fade_in_active = False
            self.fade_timer = None
            return False
        return True
    
    def start_fade_out(self, callback: Optional[Callable] = None) -> None:
        """Start fade-out animation."""
        self.fade_in_active = False
        self.fade_out_active = True
        self.fade_callback = callback
        self._cancel_fade_timer()
        self.fade_timer = GLib.timeout_add(16, self._fade_out_step)
    
    def _fade_out_step(self) -> bool:
        """Fade-out animation step."""
        self.opacity_value = max(0.0, self.opacity_value - 0.1)
        try:
            self.set_opacity(self.opacity_value)
        except Exception:
            pass
        if self.opacity_value <= 0.0:
            self.fade_out_active = False
            self.fade_timer = None
            if self.fade_callback:
                self.fade_callback()
            return False
        return True
    
    def _cancel_fade_timer(self) -> None:
        """Cancel active fade timer."""
        if self.fade_timer:
            GLib.source_remove(self.fade_timer)
            self.fade_timer = None
    
    def update_content(self, messages: List[Dict[str, str]], status_text: Optional[str] = None,
                       is_pinned: bool = False, is_tts: bool = False) -> None:
        """Update chat content with markdown rendering."""
        html_messages = []
        svg_icon = '<svg viewBox="0 0 24 24"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>'
        
        for idx, msg in enumerate(messages):
            role = msg["role"]
            rendered = self._render_markdown(msg["text"])
            # Pass index, not ID, for robust handling
            copy_btn = f'<button class="copy-btn" onclick="copyText(this, {idx})">{svg_icon}</button>'
            msg_html = f'<div class="message"><div class="text">{rendered}</div></div>'
            
            html_messages.append(
                f'<div class="message-wrapper {role}">'
                f'{msg_html}'
                f'{copy_btn}'
                f'</div>'
            )
        
        if status_text:
            html_messages.append(f'<div class="message status">{status_text}</div>')
        
        # Build pin hint - simple text with gear icon
        pin_label = CFG.HOTKEY_DEFS["pin"][0]
        tts_label = CFG.HOTKEY_DEFS["tts"][0]
        pin_status = f"{pin_label}: Unpin" if is_pinned else f"{pin_label}: Pin"
        voice_status = f"{tts_label}: Mute" if is_tts else f"{tts_label}: Voice"
        
        pin_hint = (
            f'<div class="pin-hint">'
            f'<span>{pin_status}</span>'
            f'<span style="opacity:0.2; margin:0 4px">|</span>'
            f'<span>{voice_status}</span>'
            f'<span style="opacity:0.2; margin:0 4px">|</span>'
            f'<a href="settings://open" class="settings-link" title="Settings">⚙️</a>'
            f'</div>'
        )
        
        html = CHAT_HTML_TEMPLATE.replace("{messages}", "\n".join(html_messages))
        html = html.replace("{pin_hint}", pin_hint)
        self.webview.load_html(html, None)
    
    def _on_policy_decision(self, webview, decision, decision_type) -> bool:
        """Handle URI navigations (copy://, settings://)."""
        if decision_type == WebKit2.PolicyDecisionType.NAVIGATION_ACTION:
            nav = decision.get_navigation_action()
            uri = nav.get_request().get_uri()
            if not uri:
                return False
                
            if uri.startswith("settings://"):
                GLib.idle_add(SettingsDialog.show)
                decision.ignore()
                return True
                
            if uri.startswith("copy://"):
                try:
                    idx = int(uri.split("copy://")[1])
                    if 0 <= idx < len(STATE.chat_messages):
                        text = STATE.chat_messages[idx]["text"]
                        subprocess.run(["wl-copy", "--", text], check=False)
                except Exception:
                    pass
                decision.ignore()
                return True
                
        return False
    
    @staticmethod
    def _render_markdown(text: str) -> str:
        """Convert simple markdown to HTML."""
        import html as html_lib
        text = html_lib.escape(text)
        
        # Code blocks
        text = re.sub(r'```(?:\w+)?\n?(.*?)```', r'<pre><code>\1</code></pre>', text, flags=re.DOTALL)
        # Inline code
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'<em>\1</em>', text)
        text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<em>\1</em>', text)
        # Line breaks
        text = text.replace('\n', '<br>')
        
        return text
    
    def close(self) -> None:
        """Clean up and destroy."""
        self._cancel_fade_timer()
        self.destroy()


# ============================================================================
# SECTION 9: SETTINGS DIALOG
# ============================================================================
class SettingsDialog:
    """GTK Settings dialog for voice and hotkey configuration."""
    
    _instance: Optional[Gtk.Window] = None
    
    @classmethod
    def show(cls) -> None:
        """Show settings dialog (singleton)."""
        if cls._instance and cls._instance.get_visible():
            cls._instance.present()
            return
        
        cls._instance = cls._create_dialog()
        cls._instance.show_all()
    
    @classmethod
    def _create_dialog(cls) -> Gtk.Window:
        """Create the settings dialog window."""
        dialog = Gtk.Window(title="LinuxWhisper Settings")
        dialog.set_default_size(350, 300)
        dialog.set_resizable(False)
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.set_keep_above(True)
        
        # Main container
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=15)
        vbox.set_margin_top(20)
        vbox.set_margin_bottom(20)
        vbox.set_margin_start(20)
        vbox.set_margin_end(20)
        
        # --- Voice Section ---
        voice_label = Gtk.Label(label="TTS Voice")
        voice_label.set_halign(Gtk.Align.START)
        voice_label.set_markup("<b>TTS Voice</b>")
        vbox.pack_start(voice_label, False, False, 0)
        
        voice_combo = Gtk.ComboBoxText()
        for voice in CFG.TTS_VOICES:
            voice_combo.append_text(voice.title())
        voice_combo.set_active(CFG.TTS_VOICES.index(STATE.tts_voice) if STATE.tts_voice in CFG.TTS_VOICES else 0)
        voice_combo.connect("changed", cls._on_voice_changed)
        vbox.pack_start(voice_combo, False, False, 0)
        
        # --- Hotkeys Section ---
        hotkey_label = Gtk.Label()
        hotkey_label.set_halign(Gtk.Align.START)
        hotkey_label.set_markup("<b>Hotkeys</b>")
        vbox.pack_start(hotkey_label, False, False, 10)
        
        hotkey_grid = Gtk.Grid()
        hotkey_grid.set_column_spacing(15)
        hotkey_grid.set_row_spacing(8)
        
        hotkeys = []
        display_names = {
            "dictation": "Dictation:",
            "ai": "AI Chat:",
            "ai_rewrite": "Rewrite:",
            "vision": "Vision:",
            "pin": "Pin Chat:",
            "tts": "TTS Toggle:",
        }
        
        for mode_id, (label, _, _) in CFG.HOTKEY_DEFS.items():
            name = display_names.get(mode_id, mode_id.replace("_", " ").title() + ":")
            hotkeys.append((name, label))
        
        for i, (name, key) in enumerate(hotkeys):
            name_label = Gtk.Label(label=name)
            name_label.set_halign(Gtk.Align.START)
            key_label = Gtk.Label(label=key)
            key_label.set_halign(Gtk.Align.START)
            key_label.get_style_context().add_class("dim-label")
            hotkey_grid.attach(name_label, 0, i, 1, 1)
            hotkey_grid.attach(key_label, 1, i, 1, 1)
        
        vbox.pack_start(hotkey_grid, False, False, 0)
        
        # Info label
        info_label = Gtk.Label()
        info_label.set_markup("<small><i>(Hotkeys are defined in section 2 of the code.)</i></small>")
        info_label.set_halign(Gtk.Align.START)
        vbox.pack_start(info_label, False, False, 10)
        
        # --- Close Button ---
        close_btn = Gtk.Button(label="Close")
        close_btn.connect("clicked", lambda w: dialog.destroy())
        vbox.pack_end(close_btn, False, False, 0)
        
        dialog.add(vbox)
        dialog.connect("destroy", lambda w: setattr(cls, '_instance', None))
        
        return dialog
    
    @staticmethod
    def _on_voice_changed(combo: Gtk.ComboBoxText) -> None:
        """Handle voice selection change."""
        active = combo.get_active()
        if 0 <= active < len(CFG.TTS_VOICES):
            STATE.tts_voice = CFG.TTS_VOICES[active]
            ChatManager.refresh_overlay()


# ============================================================================
# SECTION 10: SYSTEM TRAY
# ============================================================================
class TrayManager:
    """System tray (AppIndicator) management."""
    
    @staticmethod
    def start() -> None:
        """Initialize and start system tray."""
        STATE.indicator = AppIndicator.Indicator.new(
            "linuxwhisper",
            "emblem-favorite",
            AppIndicator.IndicatorCategory.APPLICATION_STATUS
        )
        STATE.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        STATE.indicator.set_title("LinuxWhisper")
        TrayManager.update_menu()
        Gtk.main()
    
    @staticmethod
    @run_on_main_thread
    def update_menu() -> None:
        """Rebuild and update tray menu."""
        if not STATE.indicator:
            return
        STATE.gtk_menu = TrayManager._build_menu()
        STATE.indicator.set_menu(STATE.gtk_menu)
    
    @staticmethod
    def _build_menu() -> Gtk.Menu:
        """Build GTK menu for tray."""
        menu = Gtk.Menu()
        
        # History items
        if STATE.answer_history:
            for item in STATE.answer_history[:CFG.ANSWER_HISTORY_LIMIT]:
                preview = item["text"][:50].replace("\n", " ")
                if len(item["text"]) > 50:
                    preview += "..."
                label = f"[{item['timestamp']}] {preview}"
                menu_item = Gtk.MenuItem(label=label)
                menu_item.connect("activate", TrayManager._make_history_callback(item))
                menu.append(menu_item)
            menu.append(Gtk.SeparatorMenuItem())
        else:
            empty = Gtk.MenuItem(label="(No History)")
            empty.set_sensitive(False)
            menu.append(empty)
            menu.append(Gtk.SeparatorMenuItem())
        
        # Clear history
        clear = Gtk.MenuItem(label="Clear History")
        clear.connect("activate", lambda w: HistoryManager.clear_all())
        menu.append(clear)
        
        # Settings
        settings_item = Gtk.MenuItem(label="Settings")
        settings_item.connect("activate", lambda w: SettingsDialog.show())
        menu.append(settings_item)
        
        menu.append(Gtk.SeparatorMenuItem())
        
        # Quit
        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", TrayManager._quit)
        menu.append(quit_item)
        
        menu.show_all()
        return menu
    
    @staticmethod
    def _make_history_callback(item: Dict[str, str]) -> Callable:
        """Create callback for history item click."""
        def callback(widget):
            # Remove prefix labels like [Dictation]
            clean = re.sub(r"^\[.*?\]\s*", "", item["text"])
            ClipboardService.paste_text(clean)
        return callback
    
    @staticmethod
    def _quit(widget) -> None:
        """Quit application."""
        Gtk.main_quit()
        os._exit(0)


# ============================================================================
# SECTION 10: MODE HANDLER
# ============================================================================
class ModeHandler:
    """Unified handler for all recording modes."""
    
    @staticmethod
    def process(mode: str, transcribed_text: str) -> None:
        """Route to appropriate handler based on mode."""
        handlers = {
            "dictation": ModeHandler._handle_dictation,
            "ai": ModeHandler._handle_ai,
            "ai_rewrite": ModeHandler._handle_ai_rewrite,
            "vision": ModeHandler._handle_vision,
        }
        handler = handlers.get(mode)
        if handler and transcribed_text:
            handler(transcribed_text)
    
    @staticmethod
    def _handle_dictation(text: str) -> None:
        """Handle dictation mode: transcribe and type."""
        log.info("dictation_complete", text_length=len(text))
        HistoryManager.add_answer(f"[Dictation] {text}")
        ChatManager.add_message("user", f"🎤 {text}")
        ClipboardService.type_text(text)
        log.info("dictation_typed")
    
    @staticmethod
    def _handle_ai(text: str) -> None:
        """Handle AI chat mode: get response and type."""
        log.info("ai_chat_request", prompt_length=len(text))
        response = AIService.chat(text)
        if not response:
            log.warning("ai_chat_no_response")
            return

        # Update histories
        HistoryManager.add_message("user", text)
        HistoryManager.add_message("assistant", response)
        HistoryManager.add_answer(response)

        # Update chat overlay
        ChatManager.add_message("user", text)
        ChatManager.add_message("assistant", response)

        ClipboardService.type_text(response)
        log.info("ai_chat_complete", response_length=len(response))
        TTSService.speak(response)
    
    @staticmethod
    def _handle_ai_rewrite(text: str) -> None:
        """Handle AI rewrite mode: rewrite selected text based on instruction."""
        result = subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True)
        original = result.stdout.strip() if result.returncode == 0 else ""
        log.info("ai_rewrite_request", instruction_length=len(text), original_length=len(original))
        prompt = (
            f"INSTRUCTION:\n{text}\n\n"
            f"ORIGINAL TEXT:\n{original}\n\n"
            "Rewrite the original text based on the instruction. "
            "Output ONLY the finished text, without introduction or formatting."
        )

        response = AIService.chat(prompt)
        if not response:
            log.warning("ai_rewrite_no_response")
            return

        # Update histories
        HistoryManager.add_message("user", f"[Rewrite] {text}\nOriginal: {original[:200]}...")
        HistoryManager.add_message("assistant", response)
        HistoryManager.add_answer(response)

        # Update chat overlay
        ChatManager.add_message("user", f"✍️ {text}")
        ChatManager.add_message("assistant", response)

        ClipboardService.paste_text(response)
        log.info("ai_rewrite_complete", response_length=len(response))
        TTSService.speak(response)
    
    @staticmethod
    def _handle_vision(text: str) -> None:
        """Handle vision mode: screenshot + AI analysis."""
        log.info("vision_request", prompt_length=len(text))
        image_b64 = ImageService.take_screenshot()
        if not image_b64:
            log.warning("vision_screenshot_failed")
            return

        response = AIService.vision(text, image_b64)
        if not response:
            log.warning("vision_no_response")
            return

        # Update histories
        HistoryManager.add_message("user", f"[Screenshot] {text}")
        HistoryManager.add_message("assistant", response)
        HistoryManager.add_answer(response)

        # Update chat overlay
        ChatManager.add_message("user", f"📸 {text}")
        ChatManager.add_message("assistant", response)

        ClipboardService.type_text(response)
        log.info("vision_complete", response_length=len(response))
        TTSService.speak(response)


# ============================================================================
# SECTION 11: KEYBOARD HANDLER (D-Bus GlobalShortcuts Portal)
# ============================================================================
class GlobalShortcutsHandler:
    """
    Global keyboard handler using XDG Desktop Portal GlobalShortcuts.

    Works natively on Wayland (Hyprland, GNOME, KDE) without X11 dependency.
    Uses Activated/Deactivated signals for push-to-talk functionality.
    """

    PORTAL_BUS = "org.freedesktop.portal.Desktop"
    PORTAL_PATH = "/org/freedesktop/portal/desktop"
    SHORTCUTS_IFACE = "org.freedesktop.portal.GlobalShortcuts"
    REQUEST_IFACE = "org.freedesktop.portal.Request"

    def __init__(self):
        self.session_handle: Optional[str] = None
        self.bus: Optional[Gio.DBusConnection] = None
        self.signal_ids: List[int] = []

    def start(self) -> bool:
        """Initialize D-Bus connection and register shortcuts."""
        try:
            self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            self._create_session()
            return True
        except Exception as e:
            log.error("globalshortcuts_init_failed", error=str(e), hint="Make sure xdg-desktop-portal-hyprland is running")
            return False

    def _create_session(self) -> None:
        """Create a GlobalShortcuts session (step 1 of 2)."""
        # Generate unique token
        token = f"linuxwhisper_{os.getpid()}"

        # CreateSession without shortcuts first
        options = {
            "handle_token": GLib.Variant("s", token),
            "session_handle_token": GLib.Variant("s", f"session_{token}"),
        }

        # Subscribe to Response signals BEFORE making the call
        self.signal_ids.append(
            self.bus.signal_subscribe(
                self.PORTAL_BUS,
                None,  # any interface
                "Response",
                None,  # any path
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_response
            )
        )

        # Call CreateSession
        result = self.bus.call_sync(
            self.PORTAL_BUS,
            self.PORTAL_PATH,
            self.SHORTCUTS_IFACE,
            "CreateSession",
            GLib.Variant("(a{sv})", (options,)),
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        log.debug("dbus_create_session_request", request_path=result.unpack()[0])

    def _on_response(self, connection, sender, path, iface, signal, params) -> None:
        """Handle all Response signals."""
        response, results = params.unpack()

        # Check if this is CreateSession response (has session_handle)
        if "session_handle" in results and not self.session_handle:
            self._on_create_session_response(response, results)
        # Check if this is BindShortcuts response (has shortcuts)
        elif "shortcuts" in results:
            self._on_bind_shortcuts_response(response, results)

    def _bind_shortcuts(self) -> None:
        """Bind shortcuts to the session (step 2 of 2)."""
        shortcuts = []
        for mode_id, (label, trigger, description) in CFG.HOTKEY_DEFS.items():
            shortcuts.append((
                mode_id,
                {
                    "description": GLib.Variant("s", description),
                    "preferred-trigger": GLib.Variant("s", trigger),
                }
            ))

        result = self.bus.call_sync(
            self.PORTAL_BUS,
            self.PORTAL_PATH,
            self.SHORTCUTS_IFACE,
            "BindShortcuts",
            GLib.Variant("(oa(sa{sv})sa{sv})", (self.session_handle, shortcuts, "", {})),
            GLib.VariantType("(o)"),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        log.debug("dbus_bind_shortcuts_request", request_path=result.unpack()[0])

    def _on_bind_shortcuts_response(self, response: int, results: dict) -> None:
        """Handle BindShortcuts response."""
        if response != 0:
            log.error("bind_shortcuts_failed", response=response)
            return

        shortcuts = results.get("shortcuts", [])
        log.info("shortcuts_bound", count=len(shortcuts))

    def _on_create_session_response(self, response: int, results: dict) -> None:
        """Handle CreateSession response."""
        if response != 0:
            log.error("globalshortcuts_session_failed", response=response)
            return

        self.session_handle = results.get("session_handle", None)
        if not self.session_handle:
            log.error("globalshortcuts_no_session_handle")
            return

        log.info("globalshortcuts_session_created", session_handle=self.session_handle)

        # Subscribe to Activated/Deactivated signals
        self.signal_ids.append(
            self.bus.signal_subscribe(
                self.PORTAL_BUS,
                self.SHORTCUTS_IFACE,
                "Activated",
                self.PORTAL_PATH,
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_activated
            )
        )

        self.signal_ids.append(
            self.bus.signal_subscribe(
                self.PORTAL_BUS,
                self.SHORTCUTS_IFACE,
                "Deactivated",
                self.PORTAL_PATH,
                None,
                Gio.DBusSignalFlags.NONE,
                self._on_deactivated
            )
        )

        # Now bind the shortcuts (step 2)
        self._bind_shortcuts()

    def _on_activated(self, connection, sender, path, iface, signal, params) -> None:
        """Handle shortcut activation (key press)."""
        session_handle, shortcut_id, timestamp, options = params.unpack()

        # Verify it's our session
        if session_handle != self.session_handle:
            return

        GLib.idle_add(self._handle_press, shortcut_id)

    def _on_deactivated(self, connection, sender, path, iface, signal, params) -> None:
        """Handle shortcut deactivation (key release)."""
        session_handle, shortcut_id, timestamp, options = params.unpack()

        # Verify it's our session
        if session_handle != self.session_handle:
            return

        GLib.idle_add(self._handle_release, shortcut_id)

    def _handle_press(self, shortcut_id: str) -> bool:
        """Handle key press on main thread."""
        if STATE.recording:
            return False

        # Extract mode from shortcut_id (e.g., "linuxwhisper-dictation" -> "dictation")
        mode = shortcut_id.replace("linuxwhisper-", "") if shortcut_id.startswith("linuxwhisper-") else shortcut_id
        log.debug("hotkey_pressed", shortcut_id=shortcut_id, mode=mode)

        # Pin toggle (non-recording action)
        if mode == "pin":
            ChatManager.toggle_pin()
            log.info("chat_pin_toggled", pinned=STATE.chat_pinned)
            return False

        # TTS toggle (non-recording action)
        if mode == "tts":
            TTSService.toggle()
            log.info("tts_toggled", enabled=STATE.tts_enabled)
            return False

        # Check for recording mode
        if mode in CFG.MODES:
            STATE.current_mode = mode
            log.info("recording_start", mode=mode)

            # For rewrite mode, copy selected text first
            if mode == "ai_rewrite":
                subprocess.run(["wl-copy", "--primary"], capture_output=True)  # Wayland clipboard
                subprocess.run(["wl-paste", "--primary"], capture_output=True)
                time.sleep(0.1)

            OverlayManager.show(mode)
            AudioService.start_recording()

        return False

    def _handle_release(self, shortcut_id: str) -> bool:
        """Handle key release on main thread."""
        if not STATE.recording:
            return False

        # Extract mode from shortcut_id
        mode = shortcut_id.replace("linuxwhisper-", "") if shortcut_id.startswith("linuxwhisper-") else shortcut_id
        log.debug("hotkey_released", shortcut_id=shortcut_id, mode=mode)

        # Check if released key matches current mode
        if mode == STATE.current_mode:
            OverlayManager.hide()
            audio_data = AudioService.stop_recording()
            log.info("recording_stop", mode=mode, has_audio=audio_data is not None)

            if audio_data is not None:
                # Process in background thread
                threading.Thread(
                    target=self._process_audio,
                    args=(audio_data, mode),
                    daemon=True
                ).start()

        return False

    def _process_audio(self, audio_data: np.ndarray, mode: str) -> None:
        """Process recorded audio in background thread."""
        log.info("processing_audio", mode=mode)
        transcribed = AudioService.transcribe(audio_data)
        if transcribed:
            log.info("dispatching_to_handler", mode=mode, text_preview=transcribed[:50])
            GLib.idle_add(lambda: ModeHandler.process(mode, transcribed))
        else:
            log.warning("transcription_empty", mode=mode)

    def stop(self) -> None:
        """Clean up D-Bus subscriptions."""
        if self.bus:
            for signal_id in self.signal_ids:
                self.bus.signal_unsubscribe(signal_id)
            self.signal_ids = []


# Global handler instance
SHORTCUTS_HANDLER: Optional[GlobalShortcutsHandler] = None


# ============================================================================
# SECTION 12: MAIN ENTRY POINT
# ============================================================================
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LinuxWhisper - Voice Assistant for Linux",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  -d, --dictation    Start in dictation mode (speech-to-text)
  -a, --ai           Start in AI chat mode
  -r, --rewrite      Start in AI rewrite mode
  -v, --vision       Start in vision mode (screenshot + AI)

Without arguments, runs normally with hotkey support.
Press the corresponding hotkey (F3/F4/F7/F8) to stop recording.
        """
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-d", "--dictation", action="store_true",
                           help="Start recording in dictation mode immediately")
    mode_group.add_argument("-a", "--ai", action="store_true",
                           help="Start recording in AI chat mode immediately")
    mode_group.add_argument("-r", "--rewrite", action="store_true",
                           help="Start recording in AI rewrite mode immediately")
    mode_group.add_argument("-v", "--vision", action="store_true",
                           help="Start recording in vision mode immediately")
    return parser.parse_args()


def start_immediate_mode(mode: str) -> bool:
    """Start recording in specified mode immediately. Returns False to remove from idle."""
    STATE.current_mode = mode

    # For rewrite mode, ensure clipboard is ready
    if mode == "ai_rewrite":
        subprocess.run(["wl-paste", "--primary"], capture_output=True)
        time.sleep(0.1)

    OverlayManager.show(mode)
    AudioService.start_recording()
    hotkey = CFG.HOTKEY_DEFS.get(f'linuxwhisper-{mode}', ('F3',))[0]
    log.info("recording_started", mode=mode, stop_key=hotkey)
    return False  # Remove from GLib.idle_add


def main() -> None:
    """Application entry point."""
    global SHORTCUTS_HANDLER

    args = parse_args()

    # Determine if immediate mode requested
    immediate_mode = None
    if args.dictation:
        immediate_mode = "dictation"
    elif args.ai:
        immediate_mode = "ai"
    elif args.rewrite:
        immediate_mode = "ai_rewrite"
    elif args.vision:
        immediate_mode = "vision"

    log.info("app_starting", transport="D-Bus GlobalShortcuts (Wayland native)")

    # Log configured shortcuts
    for mode_id, (label, trigger, description) in CFG.HOTKEY_DEFS.items():
        log.info("hotkey_registered", key=label, description=description)

    log.info("system_tray_active")

    # Initialize GlobalShortcuts handler
    SHORTCUTS_HANDLER = GlobalShortcutsHandler()
    if not SHORTCUTS_HANDLER.start():
        log.error("globalshortcuts_unavailable", hint="Ensure xdg-desktop-portal-hyprland is running: systemctl --user status xdg-desktop-portal-hyprland")
        sys.exit(1)

    # Schedule immediate mode start after GTK is ready
    if immediate_mode:
        GLib.timeout_add(500, lambda: start_immediate_mode(immediate_mode))

    # Run GTK main loop (blocks)
    TrayManager.start()


if __name__ == "__main__":
    main()
