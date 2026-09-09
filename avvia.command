#!/bin/bash
# Doppio click su questo file per avviare autoedit senza usare il terminale.
cd "$(dirname "$0")" || exit 1

PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "Primo avvio: preparo l'ambiente e installo le dipendenze."
  echo "Puo' richiedere qualche minuto, solo la prima volta..."
  echo
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERRORE: manca Python 3. Installalo da https://www.python.org/downloads/"
    read -r -p "Premi invio per chiudere. "
    exit 1
  fi
  python3 -m venv .venv || { echo "Creazione ambiente fallita."; read -r -p "Premi invio. "; exit 1; }
  "$PY" -m pip install --upgrade pip -q
  if ! "$PY" -m pip install -r requirements-gui.txt; then
    echo "Installazione delle dipendenze fallita."
    read -r -p "Premi invio per chiudere. "
    exit 1
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ATTENZIONE: 'ffmpeg' non trovato."
  echo "Installalo con Homebrew:  brew install ffmpeg"
  echo "(senza ffmpeg il montaggio non funziona)"
  echo
fi

echo "Avvio autoedit... si apre nel browser. Chiudi questa finestra per fermarlo."
exec "$PY" gui.py
