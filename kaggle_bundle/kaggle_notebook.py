# Kaggle cell -- clear the cell completely (Ctrl+A, Delete) before pasting.
import glob, os, subprocess, sys
os.environ["WTOK_EPOCHS"] = "150"
os.environ["WTOK_MAX_HOURS"] = "8.5"
os.environ["WTOK_RESUME_DIR"] = ""   # set to a previous Output dataset to continue
LAUNCH = os.environ.get("WTOK_LAUNCHER", "kaggle_frame.py")
hits = glob.glob(f"/kaggle/input/**/{LAUNCH}", recursive=True)
if not hits:
    print("kaggle_run.py NOT FOUND. /kaggle/input currently holds:")
    for p in sorted(glob.glob("/kaggle/input/*")):
        print("  ", p)
        for q in sorted(glob.glob(p + "/*"))[:6]:
            print("      ", q)
    raise SystemExit("Attach the wtok-code dataset, or refresh it to the newest "
                     "version (Kaggle pins the version an input was added at).")
print("launcher:", hits[0], flush=True)
subprocess.run([sys.executable, hits[0]], check=True)
