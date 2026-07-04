import subprocess
import sys
import time
import os

def install_ollama():
    try:
        # 检查是否已经安装了Ollama
        if os.system("where ollama >nul 2>nul") != 0:
            print("Installing Ollama...")
            # 使用Windows MSI安装程序
            subprocess.run(['msiexec', '/i', 'ollama-windows-x86_64.msi', '/quiet'], check=True)
            print("Ollama installed successfully.")
            # 等待Ollama服务启动
            time.sleep(10)
        else:
            print("Ollama is already installed.")
    except Exception as e:
        print(f"Failed to install Ollama: {str(e)}")
        return False
    return True

def download_models():
    models = []
    try:
        with open('ollama_models.txt', 'r') as f:
            models = [line.strip() for line in f if line.strip()]
    except:
        models = ['llama2']  # 默认模型
    
    for model in models:
        try:
            subprocess.run(['ollama', 'pull', model], check=True)
            print(f"Successfully downloaded {model}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {model}: {str(e)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if install_ollama():
        download_models()
    sys.exit(0)
