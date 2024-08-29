import subprocess
import sys
import os
import time

def install_requirements():
    requirements_file = 'requirements.txt'
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("All dependencies are installed.")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies.")
            sys.exit(1)
    else:
        print(f"{requirements_file} does not exist.")
        sys.exit(1)

def run_backend():
    backend_command = ["python", "./backend/main.py", "--model", "./smaller.gguf"]
    backend_process = subprocess.Popen(backend_command, stdout=sys.stdout, stderr=sys.stderr)
    return backend_process

def run_frontend():
    frontend_command = ["streamlit", "run", "./frontend/app.py"]
    frontend_process = subprocess.Popen(frontend_command, stdout=sys.stdout, stderr=sys.stderr)
    return frontend_process

def wait_for_backend(backend_url, timeout=120):
    import requests
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(backend_url+"/docs")
            if response.status_code == 200:
                print("Backend is responding.")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    print("Timeout: Backend did not respond in time.")
    return False

if __name__ == "__main__":
    install_requirements()
    
    backend_process = run_backend()
    backend_url = "http://127.0.0.1:8000"  # Adjust to the correct URL for your FastAPI app
    
    if wait_for_backend(backend_url):
        frontend_process = run_frontend()
        
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            backend_process.terminate()
            frontend_process.terminate()
            backend_process.wait()
            frontend_process.wait()
    else:
        backend_process.terminate()
        sys.exit(1)
