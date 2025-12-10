import os
import subprocess
import time
import webbrowser

BASE = r"D:\ultralytics-main"

def get_python_exe():
    # Prefer venv Python if it exists
    venv_python = os.path.join(BASE, "venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return "python"

def main():
    os.chdir(BASE)
    python_exe = get_python_exe()

    # Start Streamlit server
    subprocess.Popen([
        python_exe,
        "-m", "streamlit",
        "run", "app.py",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
    ])

    # Give server a few seconds to start
    time.sleep(5)

    # Open browser
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    main()
