import os
import sys
import subprocess
import time
import signal
import threading
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ProcessManager:
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.running = True
        
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        print("\nReceived shutdown signal, stopping all processes...")
        self.stop_all()
        sys.exit(0)
    
    def start_process(self, command: List[str], name: str, cwd: str = None) -> subprocess.Popen:
        print(f"Starting {name}...")
        env = os.environ.copy()
        env['PYTHONPATH'] = BASE_DIR
        
        process = subprocess.Popen(
            command,
            cwd=cwd or BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        self.processes.append(process)
        
        threading.Thread(
            target=self._monitor_output,
            args=(process, name, 'stdout'),
            daemon=True
        ).start()
        
        threading.Thread(
            target=self._monitor_output,
            args=(process, name, 'stderr'),
            daemon=True
        ).start()
        
        print(f"Started {name} (PID: {process.pid})")
        return process
    
    def _monitor_output(self, process: subprocess.Popen, name: str, stream_type: str):
        stream = process.stdout if stream_type == 'stdout' else process.stderr
        for line in iter(stream.readline, ''):
            if line:
                print(f"[{name}] {line.strip()}")
    
    def stop_all(self):
        self.running = False
        print(f"Stopping {len(self.processes)} processes...")
        
        for process in reversed(self.processes):
            try:
                if process.poll() is None:
                    print(f"Stopping process PID {process.pid}...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Force killing process PID {process.pid}")
                        process.kill()
            except Exception as e:
                print(f"Error stopping process: {e}")
        
        print("All processes stopped")
    
    def wait_all(self):
        try:
            while self.running:
                all_alive = True
                for process in self.processes:
                    if process.poll() is not None:
                        all_alive = False
                        print(f"Process {process.pid} exited with code {process.returncode}")
                        break
                
                if not all_alive:
                    break
                
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all()


def main():
    python_exec = sys.executable
    manager = ProcessManager()
    
    print("="*60)
    print("Financial News Sentiment Trading System")
    print("="*60)
    print()
    
    print("Checking dependencies...")
    requirements_file = os.path.join(BASE_DIR, "requirements.txt")
    if os.path.exists(requirements_file):
        result = subprocess.run(
            [python_exec, "-m", "pip", "install", "-q", "-r", requirements_file],
            capture_output=True
        )
        if result.returncode == 0:
            print("Dependencies installed successfully")
        else:
            print(f"Warning: Some dependencies may need attention: {result.stderr.decode()[:200]}")
    print()
    
    print("Creating necessary directories...")
    from config.settings import settings
    os.makedirs(settings.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(settings.HISTORICAL_DATA_DIR, exist_ok=True)
    os.makedirs(settings.FINBERT_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    print()
    
    print("Starting system components...")
    print()
    
    manager.start_process(
        [python_exec, os.path.join(BASE_DIR, "data_source", "websocket_server.py")],
        "WebSocket Server"
    )
    
    print("\nWaiting for WebSocket server to initialize...")
    time.sleep(3)
    print()
    
    streamlit_script = os.path.join(BASE_DIR, "visualization", "streamlit_app.py")
    manager.start_process(
        [
            python_exec, "-m", "streamlit", "run", streamlit_script,
            "--server.port", str(settings.STREAMLIT_PORT),
            "--server.headless", "true",
            "--server.runOnSave", "false"
        ],
        "Streamlit Frontend"
    )
    
    print()
    print("="*60)
    print("System started successfully!")
    print("="*60)
    print()
    print(f"WebSocket Server: ws://{settings.WEBSOCKET_HOST}:{settings.WEBSOCKET_PORT}")
    print(f"Streamlit Dashboard: http://localhost:{settings.STREAMLIT_PORT}")
    print()
    print(f"News generation rate: {settings.NEWS_GENERATION_RATE} articles/second")
    print(f"Window: {settings.WINDOW_DURATION}s sliding, slide every {settings.SLIDE_DURATION}s")
    print(f"Symbols: {', '.join(settings.SYMBOLS)}")
    print()
    print("Press Ctrl+C to stop all components")
    print("="*60)
    print()
    
    manager.wait_all()


if __name__ == "__main__":
    main()
