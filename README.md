# autoedit

**Automatic TikTok/CapCut-style montage.** Feed it a folder of photos/videos and one
song; it detects the song's real beat and cuts a vertical reel synced to it — Ken
Burns motion, color grading, glitch/flash accents on the strong beats. Everything
runs locally.

🇮🇹 Documentazione in italiano: **[LEGGIMI.md](LEGGIMI.md)**

---

## Requirements

- **Python 3.10+**
- **ffmpeg** on your `PATH`
  - macOS: `brew install ffmpeg`
  - Windows: `winget install Gyan.FFmpeg` (then open a new terminal)
  - Linux: your package manager (`apt install ffmpeg`, …)

## Install & run

### Easiest — no terminal

- **macOS**: double-click `avvia.command`
- **Windows**: double-click `avvia.bat`

First launch builds a virtualenv and installs dependencies (a few minutes, once),
then opens `http://127.0.0.1:7860` in your browser.

### From source

```sh
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-gui.txt
python gui.py                   # GUI
# or
python run.py --help           # CLI, no install needed
```

### As a package

```sh
pip install -e ".[gui,ai]"     # core + Gradio GUI + LLM brief
autoedit --help
```

Extras: `gui` (Gradio), `ai` (OpenAI/Anthropic clients for the AI brief), `all`.

## Usage

1. **Analyze** — upload photos/videos + the song. You get thumbnails, the waveform
   with beats marked, and the track's tempo. For videos it also estimates the most
   interesting in-point.
   *Mixing photos and short video clips (2–3s each) keeps the montage from feeling
   monotonous.*
2. Reorder or exclude media in the table.
3. Pick an **edit style** (Clean / Hype / Smooth / Glitch / Retro — coherent
   TikTok/CapCut-like recipes), or *Custom* to tune Effects / Rhythm / Text / Logo
   by hand. You can also load a **preset**.
4. **Generate** (or **Variant** for another cut on the same music).
5. Download the video from the right-hand panel.

### CLI examples

```sh
python run.py --media-dir ./photos --audio song.mp3 --out reel.mp4
python run.py --media-dir ./photos --audio song.mp3 --edit-style hype --seed 7
python run.py --help      # every option
```

### AI brief (optional)

In the *🤖 Brief AI* panel, describe the video in words and an LLM fills in style,
motion, transitions, rhythm, effects, title and format.

- **OpenAI-compatible** — including a **local** model: Ollama
  (`http://localhost:11434/v1`), LM Studio (`http://localhost:1234/v1`),
  llama.cpp server, vLLM… Base URL + model name; the API key is often not needed.
- **Anthropic** — model `claude-sonnet-5` (or `claude-opus-5` /
  `claude-haiku-4-5`) + your API key.

Copy `llm_config.example.json` to `llm_config.json` and edit it, or use the panel's
*Save connection* button. **`llm_config.json` holds your API key and is git-ignored** —
don't commit it.

## Notes

- Presets are plain JSON in `presets/`. The three bundled ones are versioned; any
  you save are ignored by git.
- Rendering is CPU-bound ffmpeg; a montage of a few dozen clips takes a minute or so.
- If you publish monetized content, mind the licensing of the music, fonts and any
  generated assets you add — the tool doesn't check that for you.

## License

MIT — see [LICENSE](LICENSE).
