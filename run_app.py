import os
import sys
import time
import webbrowser
import subprocess

# Base folder = this file's folder (works for script AND for exe)
if getattr(sys, "frozen", False):
    BASE = os.path.dirname(sys.executable)
else:
    BASE = os.path.dirname(os.path.abspath(__file__))

BASE = os.path.abspath(BASE)

def get_python_exe():
    """Use the bundled venv Python if present."""
    venv_python = os.path.join(BASE, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return "python"  # fallback for your dev machine

def main():
    os.chdir(BASE)
    python_exe = get_python_exe()

    # Start Streamlit server in background
    subprocess.Popen([
        python_exe,
        "-m", "streamlit",
        "run", "app.py",
        "--server.headless", "false",
        "--server.fileWatcherType", "none",
    ])

    # Wait a bit then open browser
    time.sleep(5)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    main()
