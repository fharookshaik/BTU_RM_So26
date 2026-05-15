import os
import subprocess
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import yt_dlp
import json
from pathlib import Path

class HarmonixLoader:
    """
    Minified Harmonix Set loader.
    Downloads repo + YouTube audio + pre-processes to mel-spectrograms + annotations.
    """
    def __init__(self, root_dir: str = "harmonix_data", sr: int = 22050, n_mels: int = 128, hop_length: int = 512):
        self.root = Path(root_dir)
        self.repo_dir = self.root / "harmonixset"
        self.tracks_dir = self.root / "tracks"
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

    def setup(self):
        """Clone repo if needed and load metadata + YouTube URLs."""
        if not self.repo_dir.exists():
            print("Cloning harmonixset repository...")
            subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/urinieto/harmonixset.git",
                str(self.repo_dir)
            ], check=True)

        # Load files
        self.metadata = pd.read_csv(self.repo_dir / "dataset" / "metadata.csv")
        self.youtube_df = pd.read_csv(self.repo_dir / "dataset" / "youtube_urls.csv")
        self.alignment_scores = pd.read_csv(self.repo_dir / "dataset" / "youtube_alignment_scores.csv")

        print(f"Loaded {len(self.metadata)} tracks")

    def _get_youtube_url(self, track_id: str) -> str:
        row = self.youtube_df[self.youtube_df["File"] == track_id]
        return row["URL"].values[0] if len(row) > 0 else None

    def download_audio(self, track_id: str, force: bool = False) -> Path | None:
        """Download audio from YouTube using yt-dlp."""
        url = self._get_youtube_url(track_id)
        if not url:
            print(f"No YouTube URL for {track_id}")
            return None

        out_dir = self.tracks_dir / track_id
        out_dir.mkdir(exist_ok=True)
        audio_path = out_dir / "audio.wav"

        if audio_path.exists() and not force:
            return audio_path

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out_dir / "temp.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }],
            "postprocessor_args": ["-ar", str(self.sr)],
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            # Rename to audio.wav
            temp_files = list(out_dir.glob("temp.*"))
            if temp_files:
                temp_files[0].rename(audio_path)
            return audio_path
        except Exception as e:
            print(f"Failed to download {track_id}: {e}")
            return None

    def preprocess_track(self, track_id: str, force: bool = False) -> dict | None:
        """Download + extract mel-spectrogram + save annotations."""
        out_dir = self.tracks_dir / track_id
        mel_path = out_dir / "mel.npy"
        beats_path = out_dir / "beats.npy"
        segments_path = out_dir / "segments.csv"

        if mel_path.exists() and not force:
            return {"mel": np.load(mel_path), "track_id": track_id}

        # Download audio
        audio_path = self.download_audio(track_id)
        if not audio_path or not audio_path.exists():
            return None

        # Load audio
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)

        # Extract mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            hop_length=self.hop_length, power=2.0
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Load original annotations
        beats_file = self.repo_dir / "dataset" / "beats_and_downbeats" / f"{track_id}.txt"
        segments_file = self.repo_dir / "dataset" / "segments" / f"{track_id}.txt"

        beats = pd.read_csv(beats_file, sep="\t", names=["time", "pos_in_bar", "bar"])
        segments = pd.read_csv(segments_file, sep="\t", names=["time", "label"])

        # Save everything
        np.save(mel_path, mel_db)
        np.save(beats_path, beats[["time", "pos_in_bar"]].values)
        segments.to_csv(segments_path, index=False)

        # Save metadata
        meta = {
            "track_id": track_id,
            "duration": len(y) / self.sr,
            "sr": self.sr,
            "n_mels": self.n_mels,
            "hop_length": self.hop_length,
        }
        with open(out_dir / "info.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {
            "mel": mel_db,
            "beats": beats,
            "segments": segments,
            "track_id": track_id
        }

    def prepare_subset(self, n_tracks: int = 50, min_alignment: float = 0.85, force: bool = False):
        """Prepare a subset of tracks (recommended for testing)."""
        self.setup()

        # Filter high-quality tracks
        good_tracks = self.alignment_scores[self.alignment_scores["score"] >= min_alignment]["File"].tolist()
        selected = good_tracks[:n_tracks]

        print(f"Preparing {len(selected)} tracks (min alignment score: {min_alignment})...")

        results = []
        for track_id in tqdm(selected, desc="Processing tracks"):
            item = self.preprocess_track(track_id, force=force)
            if item:
                results.append(item)

        print(f"Successfully processed {len(results)} tracks")
        return results

    def get_track(self, track_id: str):
        """Load a single pre-processed track."""
        out_dir = self.tracks_dir / track_id
        if not out_dir.exists():
            return self.preprocess_track(track_id)

        return {
            "mel": np.load(out_dir / "mel.npy"),
            "beats": np.load(out_dir / "beats.npy"),
            "segments": pd.read_csv(out_dir / "segments.csv"),
            "info": json.load(open(out_dir / "info.json"))
        }


# ====================== USAGE EXAMPLE ======================
if __name__ == "__main__":
    loader = MinifiedHarmonixLoader(root_dir="/Users/fharook/Documents/BTU/Research_Module/data")

    # Prepare first 30 high-quality tracks (fast test)
    data = loader.prepare_subset(n_tracks=30, min_alignment=0.88)

    # Or load a specific track later
    # track = loader.get_track("some_track_id")