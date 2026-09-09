# autoedit — montaggio automatico stile TikTok/CapCut

Monta un reel verticale da foto/video + una canzone, con tagli
sincronizzati sul beat reale del brano. Tutto in locale.

## Serve

- **Python 3** (da https://www.python.org se non ce l'hai)
- **ffmpeg** nel PATH — su macOS: `brew install ffmpeg`

## Avvio senza terminale

- **macOS**: doppio click su **`avvia.command`**
- **Windows**: doppio click su **`avvia.bat`**

Al primo avvio prepara da solo l'ambiente e installa le dipendenze
(qualche minuto, una volta sola). Poi si apre nel browser su
`http://127.0.0.1:7860`.

## Come si usa

1. **Analizza** — carica foto/video + la canzone. Vedi le miniature, la
   forma d'onda con i beat segnati e i dati del brano. Per i video viene
   anche stimato il punto d'attacco più interessante.

   *Mescolare foto e spezzoni di video rende il montaggio molto meno
   monotono. Usa clip da almeno 2-3 secondi l'una.*
2. Nella tabella puoi **riordinare** (colonna *ordine*) o **escludere**
   (colonna *includi*) i media.
3. Scegli lo **Stile di montaggio** (Pulito / Hype / Morbido / Glitch /
   Retro — ricette coerenti stile TikTok/CapCut), oppure «Personalizzato»
   per regolare a mano **Effetti / Ritmo / Testo / Logo**. Puoi anche
   caricare un **Preset**.
4. **Genera** (o **Variante** per un altro montaggio con la stessa musica).
5. Scarica il video dal pannello a destra.

### Preset

Nel pannello *💾 Preset*: scegli uno dei preset inclusi e premi *Carica*,
oppure scrivi un nome e premi *Salva* per conservare le impostazioni
attuali. I preset sono file JSON nella cartella `presets/`.

### Brief AI (opzionale)

Nel pannello *🤖 Brief AI* descrivi a parole il video che vuoi e un LLM
compila stile, movimento, transizioni, ritmo, effetti, titolo, formato.

Provider a scelta:
- **Compatibile OpenAI** — anche un modello **locale**: Ollama
  (`http://localhost:11434/v1`), LM Studio (`http://localhost:1234/v1`),
  llama.cpp server, vLLM... Base URL + nome modello; la API key spesso
  non serve (metti un valore qualsiasi).
- **Anthropic** — modello `claude-opus-5` (o `claude-sonnet-5` /
  `claude-haiku-4-5`) + la tua API key.

Premi *Salva connessione* (finisce in `llm_config.json`, **contiene la
API key**), scrivi il brief, premi *✨ Compila impostazioni*. Poi puoi
ritoccare a mano e premere *Genera*.

Richiede il pacchetto del provider scelto:
`pip install openai` oppure `pip install anthropic` (già in
`requirements-gui.txt`).

## Da riga di comando (avanzato)

```
.venv/bin/python run.py --media-dir ./foto --audio brano.mp3 --out reel.mp4
.venv/bin/python run.py --help      # tutte le opzioni
```
