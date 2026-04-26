import subprocess
import sys

def setup():
    print("Setting up HybridRec++ environment...")
    try:
        # Install dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        # Download data
        subprocess.check_call([sys.executable, "src/download_data.py"])
        print("\nSetup complete. Run 'streamlit run app/streamlit_app.py' to launch.")
    except Exception as e:
        print(f"Setup failed: {e}")

if __name__ == "__main__":
    setup()
