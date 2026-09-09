@echo off
REM Doppio click per avviare autoedit su Windows.
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo Primo avvio: preparo l'ambiente e installo le dipendenze.
  echo Puo' richiedere qualche minuto, solo la prima volta...
  echo.
  py -3 -m venv .venv || python -m venv .venv
  if not exist "%PY%" (
    echo ERRORE: manca Python 3. Installalo da https://www.python.org/downloads/
    pause
    exit /b 1
  )
  "%PY%" -m pip install --upgrade pip -q
  "%PY%" -m pip install -r requirements-gui.txt
  if errorlevel 1 (
    echo Installazione delle dipendenze fallita.
    pause
    exit /b 1
  )
)

where ffmpeg >nul 2>nul || echo ATTENZIONE: 'ffmpeg' non trovato nel PATH. Il montaggio non funzionera' senza.

echo Avvio autoedit... si apre nel browser. Chiudi questa finestra per fermarlo.
"%PY%" gui.py
pause
