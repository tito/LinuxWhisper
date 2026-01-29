LinuxWhisper
============

Voice-to-text and AI assistant for Linux desktops.
Uses OpenAI-compatible APIs (LocalAI, Groq, OpenAI) for transcription, chat, vision, and TTS.

Features
--------

- **Dictation (F3)**: Speech-to-text using Whisper.
- **AI Chat (F4)**: Context-aware Q&A.
- **Smart Rewrite (F7)**: Highlight text, speak to rewrite.
- **Vision (F8)**: Screenshot analysis.
- **Pin Chat (F9)**: Toggle chat overlay pin mode.
- **TTS (F10)**: Toggle text-to-speech for AI responses.

Prerequisites
-------------

- Linux with Wayland (Hyprland, GNOME, KDE, Sway)
- Python 3.10+
- Wayland tools: `wl-clipboard`, `wtype`, `grim`
- XDG Desktop Portal with GlobalShortcuts support

Installation
------------

1. Clone:

       git clone https://github.com/Dianjeol/LinuxWhisper.git
       cd LinuxWhisper

2. Install dependencies:

       # Arch
       sudo pacman -S wl-clipboard wtype grim xdg-desktop-portal-hyprland

       # Debian/Ubuntu
       sudo apt install wl-clipboard wtype grim

3. Run setup:

       ./setup.sh

Configuration
-------------

Create `.env` file:

    OPENAI_API_BASE=http://localhost:8080/v1   # LocalAI, or your API endpoint
    OPENAI_API_KEY=your_key                    # or "unused" for LocalAI

    # Optional model overrides
    MODEL_CHAT=gpt-4
    MODEL_VISION=gpt-4-vision-preview
    MODEL_WHISPER=whisper-1
    MODEL_TTS=tts-1

**LocalAI setup**: See https://localai.io for running models locally.

**Groq**: Get free key at https://console.groq.com, set `OPENAI_API_BASE=https://api.groq.com/openai/v1`

Hyprland Setup
--------------

LinuxWhisper uses **D-Bus GlobalShortcuts portal** for Wayland-native hotkeys.

1. Ensure the portal is running:

       systemctl --user status xdg-desktop-portal-hyprland

2. Add keybindings to `~/.config/hypr/hyprland.conf`:

       # LinuxWhisper shortcuts
       bind = , F3, global, :linuxwhisper-dictation
       bind = , F4, global, :linuxwhisper-ai
       bind = , F7, global, :linuxwhisper-ai_rewrite
       bind = , F8, global, :linuxwhisper-vision
       bind = , F9, global, :linuxwhisper-pin
       bind = , F10, global, :linuxwhisper-tts

3. Reload config:

       hyprctl reload

Usage
-----

    uv run linuxwhisper

Hotkeys (configured via Hyprland `global` dispatcher):

| Key | Action |
|-----|--------|
| F3 | Dictation (text at cursor) |
| F4 | AI Chat (response at cursor) |
| F7 | Rewrite (select -> hold -> speak -> release) |
| F8 | Vision (screenshot + AI) |
| F9 | Pin/Unpin chat overlay |
| F10 | Toggle TTS / Mute |

**Settings:**
Click the gear icon in the chat or use "Settings" in the System Tray.

Architecture
------------

- **D-Bus GlobalShortcuts**: Wayland-native hotkey registration via XDG portal
- **Clipboard**: `wl-copy` / `wl-paste` for Wayland
- **Keyboard simulation**: `wtype` for Wayland
- **Screenshots**: `grim` for Wayland
- **UI**: GTK3 + WebKit2 for overlays
