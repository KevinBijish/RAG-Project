import os
import re
from pathlib import Path

import fitz
import requests
import whisper
from bs4 import BeautifulSoup
from yt_dlp import YoutubeDL


def extract_pdf_text(pdf_path: Path) -> str:
    text_parts = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text_parts.append(f"\n--- Page {page_number} ---\n")
            text_parts.append(page.get_text())

    return "".join(text_parts).strip()


def clean_text(text: str) -> str:
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_website_text(url: str) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    article = soup.find("article")
    content = article if article else soup.body

    if content is None:
        return ""

    text_items = content.find_all(["h1", "h2", "h3", "p", "li"])
    text = "\n".join(item.get_text(" ", strip=True) for item in text_items)
    return clean_text(text)


def download_youtube_audio(youtube_url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "format": "bestaudio",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        original_file = Path(ydl.prepare_filename(info))
        audio_file = original_file.with_suffix(".mp3")

    if not audio_file.exists():
        raise FileNotFoundError("Could not find downloaded MP3 audio.")

    return audio_file


def transcribe_youtube(youtube_url: str, output_dir: Path) -> str:
    audio_path = download_youtube_audio(youtube_url, output_dir)
    model_name = os.getenv("WHISPER_MODEL", "base")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(audio_path))
    return result["text"].strip()
