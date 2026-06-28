import subprocess, csv, os, sys, time, glob

HARMONIX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "harmonixset")
URL_FILE = os.path.join(HARMONIX_DIR, "dataset", "youtube_urls.csv")
AUDIO_DIR = os.path.join(HARMONIX_DIR, "audio")

os.makedirs(AUDIO_DIR, exist_ok=True)

existing = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(AUDIO_DIR, "*.wav"))}

entries = []
with open(URL_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        fid = row["File"]
        if fid in existing:
            continue
        entries.append(row)

print(f"Already have {len(existing)} files, need to download {len(entries)} files")
sys.stdout.flush()

failed = []
for i, row in enumerate(entries):
    fid = row["File"]
    url = row["URL"]
    out_path = os.path.join(AUDIO_DIR, f"{fid}.%(ext)s")
    print(f"[{i+1}/{len(entries)}] {fid}...", end=" ", flush=True)
    try:
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
            "-o", out_path,
            "--no-playlist",
            "--quiet",
            "--no-warnings",
            url,
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        print("OK", flush=True)
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        failed.append(fid)
    time.sleep(0.5)

if failed:
    print(f"\nFailed downloads ({len(failed)}): {failed}")
else:
    print(f"\nAll downloads complete!")
