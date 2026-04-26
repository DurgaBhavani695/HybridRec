import subprocess
import sys

def setup():
    print("Setting up HybridRec++ environment...")
    try:
        # Use uv to sync dependencies
        subprocess.check_call(["uv", "sync", "--all-extras"])
        # Download data
        subprocess.check_call(["uv", "run", "python", "src/download_data.py"])
        print("\nSetup complete. Run 'uv run streamlit run app/streamlit_app.py' to launch.")
    except Exception as e:
        print(f"Setup failed: {e}")

if __name__ == "__main__":
    setup()
