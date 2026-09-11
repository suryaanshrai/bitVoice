import argparse
import shutil
import subprocess
from pathlib import Path


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def compress_wav_to_mp3(
    wav_path: Path,
    mp3_path: Path,
    *,
    vbr_quality: int,
    sample_rate_hz: int,
    ffmpeg_binary: str,
    overwrite: bool,
) -> bool:
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        local_app_data = Path.home() / "AppData" / "Local"
        winget_ffmpeg = local_app_data / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_ffmpeg.exists():
            ffmpeg = str(winget_ffmpeg)
    ffmpeg = ffmpeg or ffmpeg_binary

    if mp3_path.exists() and not overwrite:
        return is_nonempty_file(mp3_path)

    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "libmp3lame",
        "-q:a",
        str(int(vbr_quality)),
        str(mp3_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it and ensure it's on PATH (e.g. `winget install Gyan.FFmpeg`)."
        )

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"ffmpeg exit code {result.returncode}")

    return is_nonempty_file(mp3_path)


def iter_wavs(root: Path):
    yield from root.rglob("*.wav")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compress existing WAV files under audios/ to VBR MP3 using ffmpeg, optionally deleting the WAVs after successful compression."
    )
    parser.add_argument("--root", default="audios", help="Root folder to scan (default: audios)")
    parser.add_argument("--quality", type=int, default=6, help="MP3 VBR quality (0 best .. 9 worst). Default: 6")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Output sample rate Hz (default: 24000). Use 44100 for max compatibility.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary name/path (default: ffmpeg)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mp3 files")
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="Keep source wav files (default behavior deletes after successful mp3 creation)",
    )

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 2

    total = 0
    compressed = 0
    skipped = 0
    failed = 0

    for wav_path in iter_wavs(root):
        total += 1
        mp3_path = wav_path.with_suffix(".mp3")

        if mp3_path.exists() and not args.overwrite:
            if is_nonempty_file(mp3_path):
                skipped += 1
                continue

        try:
            ok = compress_wav_to_mp3(
                wav_path,
                mp3_path,
                vbr_quality=args.quality,
                sample_rate_hz=args.sample_rate,
                ffmpeg_binary=args.ffmpeg,
                overwrite=args.overwrite,
            )
        except Exception as e:
            failed += 1
            print(f"FAIL: {wav_path} -> {mp3_path} | {e}")
            continue

        if ok:
            compressed += 1
            print(f"OK: {wav_path} -> {mp3_path}")
            if not args.keep_wav:
                try:
                    wav_path.unlink(missing_ok=True)
                except OSError as e:
                    print(f"WARN: could not delete {wav_path}: {e}")
        else:
            failed += 1
            print(f"FAIL: produced empty mp3 for {wav_path}")

    print(
        f"Done. total={total} compressed={compressed} skipped={skipped} failed={failed} root={root}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
