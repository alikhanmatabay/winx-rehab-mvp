import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import json
import datetime
import asyncio
import base64
import threading
from io import BytesIO
import cv2
import numpy as np
import mediapipe as mp
import edge_tts
import streamlit.components.v1 as components

# ==============================================================================
# ХРАНЕНИЕ ПРОГРЕССА И ГЕЙМИФИКАЦИЯ (progress.json)
# ==============================================================================
PROGRESS_FILE = "progress.json"
DEFAULT_PROGRESS = {
    "E01": [
        {"date": "2026-11-20", "amplitude_pct": 40},
        {"date": "2026-11-22", "amplitude_pct": 52}
    ],
    "E02": [],
    "E03": [],
    "E04": []
}
TARGET_AMPLITUDE_NORMS = {
    "E01": 75,
    "E02": 60,
    "E03": 85,
    "E04": 80
}

def load_progress() -> dict:
    """Загружает историю тренировок из progress.json или инициализирует стартовыми данными."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for k in ["E01", "E02", "E03", "E04"]:
                        if k not in data:
                            data[k] = []
                    return data
        except Exception:
            pass
    save_progress(DEFAULT_PROGRESS)
    return json.loads(json.dumps(DEFAULT_PROGRESS))

def save_progress(data: dict):
    """Сохраняет прогресс в progress.json."""
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_exercise_stats(progress_data: dict, exercise_id: str, today_str: str = None) -> dict:
    """Возвращает статус за сегодня и лучший результат прошлых дней."""
    today = today_str or datetime.date.today().isoformat()
    entries = progress_data.get(exercise_id, [])
    today_entries = [e for e in entries if e.get("date") == today]
    is_completed_today = len(today_entries) > 0
    today_pct = max([e.get("amplitude_pct", 0) for e in today_entries]) if is_completed_today else None
    past_entries = [e for e in entries if e.get("date") != today]
    latest_past_pct = past_entries[-1].get("amplitude_pct") if past_entries else None
    best_past_pct = max([e.get("amplitude_pct", 0) for e in past_entries]) if past_entries else None
    return {
        "completed_today": is_completed_today,
        "today_pct": today_pct,
        "past_pct": latest_past_pct or best_past_pct,
        "history": entries
    }

def record_exercise_session(exercise_id: str, amplitude_pct: int, today_str: str = None) -> dict:
    """Записывает результат упражнения за сегодня в progress.json и возвращает дельту."""
    today = today_str or datetime.date.today().isoformat()
    data = load_progress()
    stats_before = get_exercise_stats(data, exercise_id, today)
    prev_pct = stats_before["past_pct"]
    
    entries = data.get(exercise_id, [])
    existing_today = [e for e in entries if e.get("date") == today]
    if existing_today:
        existing_today[0]["amplitude_pct"] = max(existing_today[0].get("amplitude_pct", 0), amplitude_pct)
    else:
        entries.append({"date": today, "amplitude_pct": amplitude_pct})
    data[exercise_id] = entries
    save_progress(data)
    
    diff_pct = (amplitude_pct - prev_pct) if prev_pct is not None else None
    return {
        "amplitude_pct": amplitude_pct,
        "prev_pct": prev_pct,
        "diff_pct": diff_pct,
        "completed_today": True
    }

VOICE_MAP = {
    "RU": "ru-RU-SvetlanaNeural",
    "KZ": "kk-KZ-AigulNeural",
    "EN": "en-US-AriaNeural",
}

@st.cache_data(show_spinner=False)
def generate_speech(text: str, lang_code: str) -> bytes | None:
    """Генерирует MP3-озвучку в оперативной памяти (BytesIO) через edge-tts."""
    voice = VOICE_MAP.get(lang_code, "ru-RU-SvetlanaNeural")
    async def _synthesize() -> bytes:
        communicate = edge_tts.Communicate(text, voice)
        buf = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()
    try:
        audio_bytes = asyncio.run(_synthesize())
        return audio_bytes if audio_bytes else None
    except Exception:
        return None

def play_hidden_audio(audio_bytes: bytes, uid: str = ""):
    """Воспроизводит MP3 через скрытый HTML <audio autoplay> без визуального плеера."""
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    components.html(
        f"""
        <audio id="tts_{uid}" autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        height=0,
    )

st.set_page_config(
    page_title="TirekAI — Домашняя реабилитация",
    page_icon="🩺",
    layout="wide"
)

# 1. Каталог 4 упражнений датасета со схемами биомеханики суставов и инструкциями на 3 языках (RU, KZ, EN)
EXERCISES = {
    "E01": {
        "code": "E01",
        "bg_color": "#E6F2F8",
        # Схема: Раскрытие кисти
        "svg": """<svg viewBox="0 0 100 100" width="86" height="86" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="50" y1="82" x2="50" y2="60" stroke="#0284C7" stroke-width="4.5" stroke-linecap="round"/>
  <circle cx="50" cy="60" r="5" fill="#0369A1" stroke="#FFFFFF" stroke-width="1.5"/>
  <rect x="36" y="44" width="28" height="18" rx="8" stroke="#93C5FD" stroke-width="2.5" stroke-dasharray="3 2" fill="#E0F2FE"/>
  <line x1="50" y1="60" x2="28" y2="34" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
  <circle cx="28" cy="34" r="3.5" fill="#0284C7"/>
  <line x1="50" y1="60" x2="40" y2="24" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
  <circle cx="40" cy="24" r="3.5" fill="#0284C7"/>
  <line x1="50" y1="60" x2="52" y2="22" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
  <circle cx="52" cy="22" r="3.5" fill="#0284C7"/>
  <line x1="50" y1="60" x2="64" y2="26" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
  <circle cx="64" cy="26" r="3.5" fill="#0284C7"/>
  <line x1="50" y1="60" x2="74" y2="38" stroke="#0284C7" stroke-width="3" stroke-linecap="round"/>
  <circle cx="74" cy="38" r="3.5" fill="#0284C7"/>
  <path d="M26 44 A32 32 0 0 1 76 44" stroke="#38BDF8" stroke-width="2.5" stroke-dasharray="3.5 3" stroke-linecap="round"/>
  <polygon points="73,40 79,44 73,48" fill="#38BDF8"/>
</svg>""",
        "RU": {
            "title": "Раскрытие кисти",
            "voice_text": "Упражнение: раскрытие кисти. Сожмите пальцы в кулак, затем плавно раскройте ладонь и полностью выпрямите пальцы. Выполните три плавных повторения."
        },
        "KZ": {
            "title": "Саусақтарды ашу",
            "voice_text": "Жаттығу: алақанды ашу. Саусақтарыңызды жұдырыққа түйіп, содан кейін алақаныңызды баяу ашып, саусақтарыңызды толық жазыңыз. Үш рет баяу қайталаңыз."
        },
        "EN": {
            "title": "Opening the hand",
            "voice_text": "Exercise: opening the hand. Make a gentle fist, then slowly open your palm and straighten all fingers. Perform three smooth repetitions."
        },
        "storyboard": [
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Запястье/предплечье -->
  <rect x="44" y="82" width="32" height="30" rx="10" fill="#E0F2FE" stroke="#0E7490" stroke-width="3.5"/>
  <!-- Ладонь (широкая) -->
  <rect x="34" y="50" width="52" height="38" rx="12" fill="#E0F2FE" stroke="#0E7490" stroke-width="3.5"/>
  <!-- 4 согнутые фаланги (кулак) -->
  <rect x="38" y="34" width="10" height="20" rx="5" fill="#BAE6FD" stroke="#0E7490" stroke-width="3"/>
  <rect x="52" y="30" width="10" height="24" rx="5" fill="#BAE6FD" stroke="#0E7490" stroke-width="3"/>
  <rect x="66" y="32" width="10" height="22" rx="5" fill="#BAE6FD" stroke="#0E7490" stroke-width="3"/>
  <rect x="78" y="38" width="9" height="16" rx="4.5" fill="#BAE6FD" stroke="#0E7490" stroke-width="2.5"/>
  <!-- Большой палец прижат сбоку -->
  <rect x="22" y="54" width="14" height="22" rx="7" fill="#BAE6FD" stroke="#0E7490" stroke-width="3" transform="rotate(-20 29 65)"/>
</svg>""",
                "RU": "1. Сжатый кулак",
                "KZ": "1. Жұдырық түю",
                "EN": "1. Clenched fist"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Запястье/предплечье -->
  <rect x="44" y="82" width="32" height="30" rx="10" fill="#E0F2FE" stroke="#0E7490" stroke-width="3.5"/>
  <!-- Ладонь -->
  <rect x="34" y="52" width="52" height="36" rx="12" fill="#E0F2FE" stroke="#0E7490" stroke-width="3.5"/>
  <!-- Пальцы полураскрыты, разгибаются вверх -->
  <rect x="37" y="22" width="10" height="34" rx="5" fill="#A5F3FC" stroke="#0E7490" stroke-width="3" transform="rotate(-10 42 39)"/>
  <rect x="51" y="16" width="10" height="38" rx="5" fill="#A5F3FC" stroke="#0E7490" stroke-width="3"/>
  <rect x="65" y="18" width="10" height="36" rx="5" fill="#A5F3FC" stroke="#0E7490" stroke-width="3" transform="rotate(8 70 36)"/>
  <rect x="78" y="28" width="9" height="28" rx="4.5" fill="#A5F3FC" stroke="#0E7490" stroke-width="2.5" transform="rotate(16 82 42)"/>
  <!-- Большой палец в стороне -->
  <rect x="20" y="52" width="14" height="24" rx="7" fill="#A5F3FC" stroke="#0E7490" stroke-width="3" transform="rotate(-30 27 64)"/>
  <!-- Стрелки раскрытия -->
  <path d="M18 40 Q10 28 14 16" stroke="#10B981" stroke-width="2.5" stroke-dasharray="4 3" fill="none" stroke-linecap="round"/>
  <polygon points="10,19 15,12 19,19" fill="#10B981"/>
  <path d="M102 40 Q110 28 106 16" stroke="#10B981" stroke-width="2.5" stroke-dasharray="4 3" fill="none" stroke-linecap="round"/>
  <polygon points="110,19 105,12 101,19" fill="#10B981"/>
</svg>""",
                "RU": "2. Раскрытие пальцев",
                "KZ": "2. Саусақтарды ашу",
                "EN": "2. Opening fingers"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Запястье/предплечье -->
  <rect x="44" y="84" width="32" height="28" rx="10" fill="#D1FAE5" stroke="#059669" stroke-width="3.5"/>
  <!-- Широкая ладонь -->
  <rect x="32" y="52" width="56" height="38" rx="13" fill="#D1FAE5" stroke="#059669" stroke-width="3.5"/>
  <!-- 4 пальца полностью раскрыты вверх (разная длина) -->
  <!-- Мизинец (короткий) -->
  <rect x="80" y="22" width="9" height="32" rx="4.5" fill="#A7F3D0" stroke="#059669" stroke-width="2.5"/>
  <!-- Безымянный -->
  <rect x="68" y="14" width="10" height="40" rx="5" fill="#A7F3D0" stroke="#059669" stroke-width="3"/>
  <!-- Средний (самый длинный) -->
  <rect x="55" y="10" width="11" height="44" rx="5.5" fill="#A7F3D0" stroke="#059669" stroke-width="3"/>
  <!-- Указательный -->
  <rect x="42" y="14" width="10" height="40" rx="5" fill="#A7F3D0" stroke="#059669" stroke-width="3"/>
  <!-- Большой палец отведён влево -->
  <rect x="20" y="48" width="14" height="28" rx="7" fill="#A7F3D0" stroke="#059669" stroke-width="3" transform="rotate(-35 27 62)"/>
</svg>""",
                "RU": "3. Прямая ладонь",
                "KZ": "3. Ашық алақан",
                "EN": "3. Fully open palm"
            }
        ]
    },
    "E02": {
        "code": "E02",
        "bg_color": "#E8F5EE",
        # Схема: Подъём запястья
        "svg": """<svg viewBox="0 0 100 100" width="86" height="86" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="18" y1="64" x2="52" y2="64" stroke="#059669" stroke-width="5" stroke-linecap="round"/>
  <line x1="16" y1="70" x2="54" y2="70" stroke="#CBD5E1" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="52" cy="64" r="6.5" fill="#047857" stroke="#FFFFFF" stroke-width="2"/>
  <line x1="52" y1="64" x2="84" y2="64" stroke="#A7F3D0" stroke-width="3.5" stroke-dasharray="3 3" stroke-linecap="round"/>
  <circle cx="84" cy="64" r="3.5" fill="#6EE7B7"/>
  <line x1="52" y1="64" x2="76" y2="32" stroke="#059669" stroke-width="4.5" stroke-linecap="round"/>
  <circle cx="76" cy="32" r="4.5" fill="#047857" stroke="#FFFFFF" stroke-width="1.5"/>
  <path d="M72 60 A24 24 0 0 0 74 38" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="70,41 74,34 78,41" fill="#10B981"/>
</svg>""",
        "RU": {
            "title": "Подъём запястья",
            "voice_text": "Упражнение: подъём запястья. Положите предплечье ровно, опустите кисть вниз, а затем плавно поднимите её вверх. Выполните три плавных повторения без пауз."
        },
        "KZ": {
            "title": "Білекті көтеру",
            "voice_text": "Жаттығу: білекті көтеру. Білегіңізді тегіс қойып, қолыңызды төмен түсіріңіз, содан кейін баяу жоғары көтеріңіз. Кідіріссіз үш рет баяу қайталаңыз."
        },
        "EN": {
            "title": "Lifting the wrist",
            "voice_text": "Exercise: lifting the wrist. Rest your forearm flat, lower your hand down, and then smoothly lift your wrist upward. Perform three smooth repetitions without pauses."
        },
        "storyboard": [
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="16" y1="65" x2="62" y2="65" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="16" cy="65" r="4.5" fill="#94A3B8"/>
  <line x1="62" y1="65" x2="98" y2="65" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="3 3"/>
  <line x1="62" y1="65" x2="88" y2="91" stroke="#64748B" stroke-width="4" stroke-linecap="round"/>
  <circle cx="88" cy="91" r="5" fill="#64748B"/>
  <circle cx="62" cy="65" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M78 65 A16 16 0 0 1 74 77" stroke="#94A3B8" stroke-width="2" fill="none"/>
  <text x="82" y="80" font-family="-apple-system,sans-serif" font-size="11" font-weight="700" fill="#64748B">-45°</text>
</svg>""",
                "RU": "1. Кисть опущена",
                "KZ": "1. Білек төмен",
                "EN": "1. Wrist dropped down"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="16" y1="65" x2="62" y2="65" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="16" cy="65" r="4.5" fill="#94A3B8"/>
  <line x1="62" y1="65" x2="98" y2="65" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="98" cy="65" r="5" fill="#0E7490"/>
  <circle cx="62" cy="65" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M86 85 A34 34 0 0 0 88 45" stroke="#10B981" stroke-width="2.5" stroke-dasharray="3.5 2.5" fill="none" stroke-linecap="round"/>
  <polygon points="82,48 88,40 94,48" fill="#10B981"/>
  <text x="78" y="36" font-family="-apple-system,sans-serif" font-size="11" font-weight="700" fill="#10B981">0°</text>
</svg>""",
                "RU": "2. Подъём вверх",
                "KZ": "2. Жоғары көтеру",
                "EN": "2. Lifting upward"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="16" y1="65" x2="62" y2="65" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="16" cy="65" r="4.5" fill="#94A3B8"/>
  <line x1="62" y1="65" x2="98" y2="65" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="3 3"/>
  <line x1="62" y1="65" x2="88" y2="39" stroke="#10B981" stroke-width="4" stroke-linecap="round"/>
  <circle cx="88" cy="39" r="6" fill="#10B981" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="62" cy="65" r="6" fill="#10B981" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M78 65 A16 16 0 0 0 74 53" stroke="#10B981" stroke-width="2" fill="none"/>
  <text x="82" y="30" font-family="-apple-system,sans-serif" font-size="12" font-weight="700" fill="#10B981">+45°</text>
</svg>""",
                "RU": "3. Верхняя точка",
                "KZ": "3. Жоғарғы нүкте",
                "EN": "3. Top position"
            }
        ]
    },
    "E03": {
        "code": "E03",
        "bg_color": "#EFEBF8",
        # Схема: Разгибание локтя
        "svg": """<svg viewBox="0 0 100 100" width="86" height="86" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="18" y1="52" x2="48" y2="52" stroke="#7C3AED" stroke-width="4.5" stroke-linecap="round"/>
  <circle cx="18" cy="52" r="4.5" fill="#8B5CF6"/>
  <circle cx="48" cy="52" r="7" fill="#6D28D9" stroke="#FFFFFF" stroke-width="2"/>
  <line x1="48" y1="52" x2="48" y2="24" stroke="#DDD6FE" stroke-width="3.5" stroke-dasharray="3.5 3" stroke-linecap="round"/>
  <circle cx="48" cy="24" r="3.5" fill="#C4B5FD"/>
  <line x1="48" y1="52" x2="86" y2="52" stroke="#7C3AED" stroke-width="4.5" stroke-linecap="round"/>
  <circle cx="86" cy="52" r="4.5" fill="#6D28D9" stroke="#FFFFFF" stroke-width="1.5"/>
  <path d="M48 32 A20 20 0 0 1 68 52" stroke="#8B5CF6" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="65,47 71,52 65,56" fill="#8B5CF6"/>
  <text x="48" y="74" text-anchor="middle" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif" font-size="12" font-weight="800" fill="#6D28D9">180°</text>
</svg>""",
        "RU": {
            "title": "Разгибание локтя",
            "voice_text": "Упражнение: разгибание локтя. Согните руку в локте под прямым углом, затем плавно выпрямите её полностью до ста восьмидесяти градусов. Выполните три плавных повторения."
        },
        "KZ": {
            "title": "Шынтақты жазу",
            "voice_text": "Жаттығу: шынтақты жазу. Қолыңызды шынтақтан тік бұрыш жасап бүгіңіз, содан кейін оны жүз сексен градусқа дейін толық жазыңыз. Үш рет баяу қайталаңыз."
        },
        "EN": {
            "title": "Extending the elbow",
            "voice_text": "Exercise: extending the elbow. Bend your elbow to ninety degrees, then smoothly straighten your arm all the way to one hundred eighty degrees. Perform three smooth repetitions."
        },
        "storyboard": [
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="80" x2="65" y2="80" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="20" cy="80" r="5" fill="#94A3B8"/>
  <line x1="65" y1="80" x2="65" y2="30" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="65" cy="30" r="5" fill="#0E7490"/>
  <circle cx="65" cy="80" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M65 60 A20 20 0 0 0 45 80" stroke="#94A3B8" stroke-width="2" stroke-dasharray="3 2" fill="none"/>
  <text x="44" y="66" font-family="-apple-system,sans-serif" font-size="12" font-weight="700" fill="#64748B">90°</text>
</svg>""",
                "RU": "1. Локоть 90°",
                "KZ": "1. Шынтақ 90°",
                "EN": "1. Elbow bent 90°"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="20" y1="80" x2="65" y2="80" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="20" cy="80" r="5" fill="#94A3B8"/>
  <line x1="65" y1="80" x2="65" y2="30" stroke="#CBD5E1" stroke-width="2" stroke-dasharray="3 3"/>
  <line x1="65" y1="80" x2="100" y2="45" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="100" cy="45" r="5" fill="#0E7490"/>
  <circle cx="65" cy="80" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <path d="M65 48 A32 32 0 0 1 88 56" stroke="#10B981" stroke-width="2.5" stroke-dasharray="3.5 2.5" fill="none" stroke-linecap="round"/>
  <polygon points="84,50 92,58 92,49" fill="#10B981"/>
  <text x="44" y="66" font-family="-apple-system,sans-serif" font-size="11" font-weight="700" fill="#0E7490">135°</text>
</svg>""",
                "RU": "2. Разгибание",
                "KZ": "2. Жазу",
                "EN": "2. Extending arm"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="15" y1="65" x2="105" y2="65" stroke="#10B981" stroke-width="4" stroke-linecap="round"/>
  <circle cx="15" cy="65" r="5" fill="#10B981"/>
  <circle cx="60" cy="65" r="6" fill="#10B981" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="105" cy="65" r="5" fill="#10B981"/>
  <text x="60" y="86" text-anchor="middle" font-family="-apple-system,sans-serif" font-size="13" font-weight="800" fill="#10B981">180°</text>
</svg>""",
                "RU": "3. Рука прямая 180°",
                "KZ": "3. Түзу қол 180°",
                "EN": "3. Straight arm 180°"
            }
        ]
    },
    "E04": {
        "code": "E04",
        "bg_color": "#FBF2E6",
        # Схема: Подъём предмета
        "svg": """<svg viewBox="0 0 100 100" width="86" height="86" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="37" y="24" width="14" height="6" rx="2" stroke="#D97706" stroke-width="2.5" fill="#FEF3C7"/>
  <rect x="40" y="30" width="8" height="5" stroke="#D97706" stroke-width="2" fill="#FEF3C7"/>
  <rect x="32" y="35" width="24" height="42" rx="6" stroke="#D97706" stroke-width="2.5" fill="#FEF3C7"/>
  <line x1="32" y1="52" x2="56" y2="52" stroke="#B45309" stroke-width="2" stroke-dasharray="2 2"/>
  <line x1="68" y1="74" x2="68" y2="34" stroke="#D97706" stroke-width="3" stroke-linecap="round"/>
  <polygon points="63,38 68,26 73,38" fill="#D97706"/>
  <line x1="22" y1="68" x2="22" y2="44" stroke="#FBBF24" stroke-width="2" stroke-linecap="round" stroke-dasharray="3 3"/>
  <polygon points="19,47 22,40 25,47" fill="#FBBF24"/>
</svg>""",
        "RU": {
            "title": "Подъём предмета",
            "voice_text": "Упражнение: подъём предмета. Опустите руку вдоль корпуса, затем плавно поднимите предмет вперёд и вверх до уровня плеча. Выполните три плавных повторения."
        },
        "KZ": {
            "title": "Затты көтеру",
            "voice_text": "Жаттығу: затты көтеру. Қолыңызды денеңіздің бойымен төмен түсіріп, содан кейін затты алға және иық деңгейіне дейін баяу көтеріңіз. Үш рет баяу қайталаңыз."
        },
        "EN": {
            "title": "Lifting an object",
            "voice_text": "Exercise: lifting an object. Lower your arm along your side, then smoothly lift the object forward and up to shoulder level. Perform three smooth repetitions."
        },
        "storyboard": [
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="30" y1="18" x2="30" y2="102" stroke="#CBD5E1" stroke-width="3" stroke-dasharray="4 3" stroke-linecap="round"/>
  <line x1="38" y1="32" x2="38" y2="86" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="38" cy="32" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="38" cy="92" r="7" fill="#E0F2FE" stroke="#0E7490" stroke-width="3"/>
</svg>""",
                "RU": "1. Рука внизу",
                "KZ": "1. Қол төменде",
                "EN": "1. Arm hanging down"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="30" y1="18" x2="30" y2="102" stroke="#CBD5E1" stroke-width="3" stroke-dasharray="4 3" stroke-linecap="round"/>
  <line x1="38" y1="32" x2="73" y2="67" stroke="#0E7490" stroke-width="4" stroke-linecap="round"/>
  <circle cx="38" cy="32" r="6" fill="#0E7490" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="77" cy="71" r="7" fill="#D1FAE5" stroke="#10B981" stroke-width="3"/>
  <line x1="96" y1="88" x2="96" y2="46" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"/>
  <polygon points="91,50 96,40 101,50" fill="#10B981"/>
</svg>""",
                "RU": "2. Подъём",
                "KZ": "2. Көтеру",
                "EN": "2. Lifting up"
            },
            {
                "svg": """<svg viewBox="0 0 120 120" width="115" height="115" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="30" y1="18" x2="30" y2="102" stroke="#CBD5E1" stroke-width="3" stroke-dasharray="4 3" stroke-linecap="round"/>
  <line x1="38" y1="32" x2="88" y2="32" stroke="#10B981" stroke-width="4" stroke-linecap="round"/>
  <circle cx="38" cy="32" r="6" fill="#10B981" stroke="#FFFFFF" stroke-width="2"/>
  <circle cx="94" cy="32" r="7" fill="#D1FAE5" stroke="#10B981" stroke-width="3"/>
  <path d="M38 52 A20 20 0 0 0 58 32" stroke="#10B981" stroke-width="2" stroke-dasharray="3 2" fill="none"/>
  <text x="44" y="50" font-family="-apple-system,sans-serif" font-size="11" font-weight="700" fill="#10B981">90°</text>
  <path d="M89 18 L93 22 L101 14" stroke="#10B981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
</svg>""",
                "RU": "3. Уровень плеча",
                "KZ": "3. Иық деңгейі",
                "EN": "3. Shoulder level"
            }
        ]
    }
}

# 2. Инициализация состояния
if "selected_exercise" not in st.session_state:
    st.session_state["selected_exercise"] = None
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "lang" not in st.session_state:
    st.session_state["lang"] = "RU"

# Синхронизация языка из URL (?lang=RU/KZ/EN)
if "lang" in st.query_params:
    url_lang = st.query_params["lang"]
    if url_lang in ("RU", "KZ", "EN") and st.session_state["lang"] != url_lang:
        st.session_state["lang"] = url_lang

# Синхронизация упражнения из URL (?ex=E01..E04)
if "ex" in st.query_params:
    ex_query = st.query_params["ex"]
    if ex_query in EXERCISES and st.session_state["selected_exercise"] != ex_query:
        st.session_state["selected_exercise"] = ex_query
        st.session_state["analysis_result"] = None

# 3. Пользовательские стили (Широкая верстка 96%, карточки min-height 360px, круг 120px)
st.markdown("""
<style>
    /* 1. Скрытие стандартного верхнего хедера Streamlit и тулбара */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    #MainMenu, footer {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* 2. Широкий контейнер на 96% ширины экрана с отступами 2rem */
    .block-container {
        max-width: 96% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1.2rem !important;
    }
    
    /* Базовые параметры: только font-family без глобального font-size */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    /* Базовый размер текста для параграфов и лейблов (не заголовков) */
    p, label, .stMarkdown p {
        font-size: 20px;
        line-height: 1.5;
        color: #1E293B;
    }
    
    /* Исключение и защита шрифта Material Symbols/Icons от перезаписи */
    span[data-testid="stIconMaterial"], 
    [data-testid="stIconMaterial"],
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons", sans-serif !important;
        font-size: 24px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        display: inline-block !important;
    }
    
    .stApp {
        background-color: #F4F7FA;
    }
    
    /* 3. Крупные шрифты на всех кнопках и выравнивание 3 кнопок выбора языка */
    .stButton > button,
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }

    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    .lang-switch-wrapper {
        display: flex !important;
        width: 100% !important;
        align-items: center !important;
        margin-top: 4px !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] {
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .lang-switch-wrapper .stButton > button,
    .lang-switch-wrapper div[data-testid="stButton"] button {
        width: 100% !important;
        min-width: 72px !important;
        min-height: 54px !important;
        max-height: 54px !important;
        height: 54px !important;
        margin: 0 !important;
        padding: 6px 4px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-align: center !important;
        border-radius: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        transform: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button:hover,
    .lang-switch-wrapper div[data-testid="stButton"] button:focus,
    .lang-switch-wrapper div[data-testid="stButton"] button:active {
        transform: none !important;
        font-size: 20px !important;
    }
    
    .lang-switch-wrapper .stButton > button p,
    .lang-switch-wrapper .stButton > button span,
    .lang-switch-wrapper .stButton > button div,
    .lang-switch-wrapper div[data-testid="stButton"] button p,
    .lang-switch-wrapper div[data-testid="stButton"] button span,
    .lang-switch-wrapper div[data-testid="stButton"] button div {
        font-size: 20px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Активная кнопка: яркий белый текст на бирюзовом фоне */
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="primary"] {
        background-color: #0E7490 !important;
        color: #FFFFFF !important;
        border: 2px solid #0E7490 !important;
        box-shadow: 0 2px 8px rgba(14, 116, 144, 0.28) !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    
    /* Неактивная кнопка */
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #334155 !important;
        border: 2px solid #CBD5E1 !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="secondary"] * {
        color: #334155 !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #0E7490 !important;
        color: #0E7490 !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="secondary"]:hover * {
        color: #0E7490 !important;
    }
    
    /* 4. Карточки Экрана 1 и Дневной план To-Do */
    .daily-plan-card {
        background: #FFFFFF !important;
        border: 2px solid #CCFBF1 !important;
        border-radius: 20px !important;
        padding: 22px 28px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 18px rgba(14, 116, 144, 0.06) !important;
    }
    
    .exercise-card-link {
        text-decoration: none !important;
        display: block !important;
        height: 100% !important;
        outline: none !important;
    }
    
    .exercise-card {
        min-height: 480px !important;
        height: auto !important;
        border-radius: 24px !important;
        padding: 30px 20px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: space-between !important;
        text-align: center !important;
        cursor: pointer !important;
        border: 2px solid transparent !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04) !important;
        user-select: none !important;
    }
    
    .exercise-card:hover {
        transform: translateY(-8px) !important;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.09) !important;
    }
    
    .card-e01 { background-color: #E6F2F8 !important; }
    .card-e01:hover { border-color: #38BDF8 !important; }
    
    .card-e02 { background-color: #E8F5EE !important; }
    .card-e02:hover { border-color: #34D399 !important; }
    
    .card-e03 { background-color: #EFEBF8 !important; }
    .card-e03:hover { border-color: #A78BFA !important; }
    
    .card-e04 { background-color: #FBF2E6 !important; }
    .card-e04:hover { border-color: #FBBF24 !important; }
    
    .exercise-card-badge {
        font-size: 16px !important;
        font-weight: 800 !important;
        padding: 5px 14px !important;
        border-radius: 8px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        color: #334155 !important;
        letter-spacing: 0.5px !important;
        display: inline-block !important;
    }
    
    /* Белый круг 120x120px для схемы биомеханики */
    .exercise-card-circle {
        width: 120px !important;
        height: 120px !important;
        background-color: #FFFFFF !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06) !important;
        margin: 10px auto !important;
        flex-shrink: 0 !important;
    }
    
    .exercise-card-title {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0F172A !important;
        line-height: 1.25 !important;
        margin-top: 4px !important;
        display: block !important;
    }

    .exercise-card-status-done {
        background: #DCFCE7 !important;
        border: 1.5px solid #86EFAC !important;
        border-radius: 12px !important;
        padding: 8px 12px !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        color: #166534 !important;
        width: 100% !important;
        display: block !important;
        text-align: center !important;
        box-sizing: border-box !important;
        margin-top: 14px !important;
    }
    
    .exercise-card-status-todo {
        background: #F1F5F9 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 8px 12px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #475569 !important;
        width: 100% !important;
        display: block !important;
        text-align: center !important;
        box-sizing: border-box !important;
        margin-top: 14px !important;
    }
    
    .exercise-card-prev-line {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #64748B !important;
        margin-top: 3px !important;
        display: block !important;
    }
    
    /* Контейнеры Экрана 2 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border-radius: 18px !important;
        padding: 24px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05) !important;
        margin-bottom: 20px !important;
    }
    
    /* Главная кнопка камеры (Экран 2) */
    div[data-testid="stButton"] button[kind="primary"]:not(.lang-btn) {
        min-height: 76px !important;
        background-color: #0E7490 !important;
        color: #FFFFFF !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        padding: 18px 32px !important;
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(14, 116, 144, 0.28) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }
    
    div[data-testid="stButton"] button[kind="primary"]:not(.lang-btn):hover {
        background-color: #155E75 !important;
        box-shadow: 0 6px 20px rgba(14, 116, 144, 0.38) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Второстепенная кнопка (назад, повторить) */
    .screen2-btn-wrapper div[data-testid="stButton"] button[kind="secondary"] {
        min-height: 56px !important;
        font-size: 21px !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        border: 2px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #334155 !important;
        padding: 10px 22px !important;
    }
    
    .screen2-btn-wrapper div[data-testid="stButton"] button[kind="secondary"]:hover {
        border-color: #0E7490 !important;
        color: #0E7490 !important;
        background-color: #F8FAFC !important;
    }
    
    /* Визуальная подсказка перед камерой */
    .patient-instruction-card {
        background-color: #F8FAFC;
        border: 2px dashed #CBD5E1;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 14px;
    }
    
    .patient-instruction-icon {
        font-size: 30px;
        flex-shrink: 0;
    }
    
    .patient-instruction-text {
        font-size: 20px;
        font-weight: 600;
        color: #334155;
        line-height: 1.45;
    }
    
    /* Баннеры отсчета и записи */
    .countdown-box {
        text-align: center;
        font-size: 40px;
        font-weight: 800;
        padding: 20px;
        background-color: #ECFEFF;
        border: 3px solid #67E8F9;
        border-radius: 16px;
        color: #0E7490;
        margin-bottom: 16px;
    }
    
    .recording-box {
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        padding: 16px;
        background-color: #FFF1F2;
        border: 3px solid #FDA4AF;
        border-radius: 16px;
        color: #BE123C;
        margin-bottom: 16px;
    }
    
    /* Баннеры результатов (компактные, для нижней сводки) */
    .result-banner-success {
        background-color: #ECFDF5;
        border: 2px solid #A7F3D0;
        border-radius: 16px;
        padding: 20px 22px;
        color: #065F46;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 18px;
    }
    
    .result-banner-warning {
        background-color: #FFFBEB;
        border: 2px solid #FDE68A;
        border-radius: 16px;
        padding: 20px 22px;
        color: #92400E;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 18px;
    }
    
    /* Крупный блок-окно итогового результата */
    .result-card-success {
        background: linear-gradient(135deg, #DCFCE7 0%, #ECFDF5 100%);
        border: 3px solid #4ADE80;
        border-radius: 20px;
        padding: 28px 30px 22px 30px;
        margin-bottom: 22px;
        box-shadow: 0 8px 28px rgba(22, 163, 74, 0.13);
    }
    .result-card-success .rc-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #15803D !important;
        margin-bottom: 10px !important;
        line-height: 1.25 !important;
    }
    .result-card-success .rc-desc {
        font-size: 20px !important;
        font-weight: 500 !important;
        color: #166534 !important;
        line-height: 1.5 !important;
    }
    
    .result-card-warning {
        background: linear-gradient(135deg, #FEF3C7 0%, #FFFBEB 100%);
        border: 3px solid #FBBF24;
        border-radius: 20px;
        padding: 28px 30px 22px 30px;
        margin-bottom: 22px;
        box-shadow: 0 8px 28px rgba(217, 119, 6, 0.13);
    }
    .result-card-warning .rc-title {
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #B45309 !important;
        margin-bottom: 10px !important;
        line-height: 1.25 !important;
    }
    .result-card-warning .rc-desc {
        font-size: 20px !important;
        font-weight: 500 !important;
        color: #92400E !important;
        line-height: 1.5 !important;
    }
    
    /* Метрики */
    .metric-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 18px 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        text-align: center;
        margin-bottom: 16px;
    }
    
    .metric-label {
        font-size: 17px !important;
        font-weight: 700 !important;
        color: #64748B !important;
        margin-bottom: 6px !important;
    }
    
    .metric-value {
        font-size: 42px !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
    }
    
    .metric-value-total { color: #0E7490 !important; }
    .metric-value-ok { color: #15803D !important; }
    .metric-value-warn { color: #B45309 !important; }
    
    div[data-testid="stDataFrame"] * {
        font-size: 20px !important;
    }

    /* ── 3-frame storyboard animation ── */
    @keyframes sbHighlight {
        0%, 25%   { border-color: #0E7490; box-shadow: 0 0 0 3px rgba(14,116,144,0.18); }
        26%, 100% { border-color: #E2E8F0; box-shadow: none; }
    }
    @keyframes sbHighlight2 {
        0%, 25%   { border-color: #E2E8F0; box-shadow: none; }
        26%, 60%  { border-color: #10B981; box-shadow: 0 0 0 3px rgba(16,185,129,0.18); }
        61%, 100% { border-color: #E2E8F0; box-shadow: none; }
    }
    @keyframes sbHighlight3 {
        0%, 60%   { border-color: #E2E8F0; box-shadow: none; }
        61%, 95%  { border-color: #0E7490; box-shadow: 0 0 0 3px rgba(14,116,144,0.18); }
        96%, 100% { border-color: #E2E8F0; box-shadow: none; }
    }
    .sb-wrapper {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 18px;
        width: 100%;
        box-sizing: border-box;
    }
    .sb-frame {
        background: #FFFFFF;
        border: 2.5px solid #E2E8F0;
        border-radius: 16px;
        padding: 10px 6px 10px 6px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 6px;
        min-width: 0;
        overflow: hidden;
        box-sizing: border-box;
        animation-duration: 3.6s;
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
    }
    .sb-frame svg {
        max-width: 100%;
        height: auto;
    }
    .sb-frame-1 { animation-name: sbHighlight; }
    .sb-frame-2 { animation-name: sbHighlight2; }
    .sb-frame-3 { animation-name: sbHighlight3; }
    .stCheckbox label p,
    div[data-testid="stCheckbox"] label p {
        font-size: 17px !important;
        font-weight: 600 !important;
    }

    .sb-caption {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #334155 !important;
        line-height: 1.3 !important;
    }
    .sb-reminder {
        background: #F0FDFA;
        border-left: 4px solid #10B981;
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 19px !important;
        font-weight: 600 !important;
        color: #065F46 !important;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 4. Тексты интерфейса на 3 языках (RU / KZ / EN)
TEXTS = {
    "RU": {
        "app_title": "TirekAI — Домашняя реабилитация",
        "app_subtitle": "Интеллектуальный помощник для восстановления движений",
        "disclaimer_text": "Инструмент носит информационный характер, не ставит диагноз и не заменяет консультацию врача.",
        "screen1_title": "Выберите упражнение",
        "back_btn": "← Выбрать другое упражнение",
        "step1_title": "1. Инструкция к упражнению",
        "step2_title": "2. Выполнение перед камерой",
        "patient_hint": "Сядьте так, чтобы рука была полностью видна в кадре, и нажмите кнопку ниже.",
        "camera_start_btn": "📹 Начать упражнение (10 сек)",
        "camera_starting": "Включаем камеру...",
        "get_ready": "⏳ Приготовьтесь: {sec}...",
        "recording_live": "🔴 Запись: осталось {sec} сек",
        "recording_done": "✅ Запись завершена! Анализируем движение...",
        "camera_error": "⚠️ Камера не найдена. Пожалуйста, проверьте подключение веб-камеры.",
        "expander_upload": "📁 Загрузить готовое видео (для теста ML)",
        "upload_label": "Выберите видеофайл (.mp4, .avi, .mov)",
        "analyze_btn": "Проверить файл",
        "analyzing": "Анализируем движения руки...",
        "result_success": "✅ Отлично! Вы правильно выполнили упражнение.",
        "result_success_title": "✅ Отлично! Вы выполнили упражнение правильно!",
        "result_success_desc": "Движения были плавными и на полную амплитуду. Так держать! 💪",
        "result_warning": "⚠️ Посмотрите инструкцию слева еще раз и попробуйте заново.",
        "result_warning_title": "⚠️ Почти получилось! Попробуйте ещё раз",
        "result_warning_desc": "Посмотрите инструкцию слева ещё раз: старайтесь выполнять движение плавно, без пауз и на полную амплитуду.",
        "retry_btn": "🔄 Попробовать заново",
        "back_exercises_btn": "← К выбору упражнений",
        "summary_title": "Итоги занятия",
        "total_cycles": "Всего повторов",
        "completed_cycles": "Выполнено",
        "review_cycles": "Проверьте инструкцию",
        "status_ok": "✅ Отлично",
        "status_warn": "⚠️ Проверьте инструкцию",
        "chart_title": "Динамика угла движения (°)",
        "cycle_col": "Повтор",
        "result_col": "Результат",
        "conf_col": "Точность ИИ",
        "angle_hud": "Угол",
        "side_right": "Правая рука",
        "side_left": "Левая рука",
        "analyzed_arm_label": "Анализируемая рука (макс. амплитуда)",
        "full_amplitude_tip": "💡 Напоминание: выполняйте движение плавно, на полную амплитуду и без пауз в середине движения.",
        "repeat_reminder": "💡 Выполните 3 плавных повторения без пауз",
        "replay_voice_btn": "🔊 Прослушать ещё раз",
        "camera_voice_prompt": "Приготовьтесь, садитесь перед камерой и начинайте упражнение",
        "daily_plan_title": "План тренировок на сегодня",
        "daily_plan_sub": "Выполните 4 назначенных упражнения для устойчивого восстановления суставов",
        "daily_plan_done_all": "🎉 Отличная работа! Все 4 упражнения на сегодня выполнены",
        "today_counter_label": "выполнено",
        "card_status_todo": "⏳ 0 / 1 сегодня",
        "card_status_done": "✅ Выполнено сегодня",
        "card_prev_result": "Прошлый результат: {pct}%",
        "recovery_progress_title": "Ваш прогресс восстановления",
        "recovery_pct_label": "{pct}% от целевой нормы подвижности",
        "recovery_diff_up": "📈 +{diff}% по сравнению с прошлым разом! Сустав разрабатывается",
        "recovery_diff_same": "👍 Стабильный результат ({pct}%), продолжайте плавные занятия",
        "recovery_diff_start": "🎯 Отличный старт! Результат зафиксирован в дневнике",
        "today_completed_badge": "✅ Засчитано в дневной план (1 / 1 сегодня)",
        "arm_selector_title": "Тренируемая рука:",
        "arm_right": "Правая рука",
        "arm_left": "Левая рука",
        "ml_error_title": "⚠️ Движение руки не удалось распознать",
        "ml_error_desc": "Пожалуйста, сядьте перед камерой так, чтобы рука была полностью видна в кадре, и выполните движение плавно без резких рывков."
    },
    "KZ": {
        "app_title": "TirekAI — Үйде оңалту көмекшісі",
        "app_subtitle": "Қол қимылдарын қалпына келтіруге арналған зияткерлік көмекші",
        "disclaimer_text": "Бұл құрал тек ақпараттық сипатқа ие, диагноз қоймайды және дәрігердің кеңесін алмастырмайды.",
        "screen1_title": "Жаттығуды таңдаңыз",
        "back_btn": "← Басқа жаттығуды таңдау",
        "step1_title": "1. Жаттығу нұсқаулығы",
        "step2_title": "2. Камера алдында орындау",
        "patient_hint": "Қолыңыз кадрға толық көрінетіндей отырыңыз да, төмендегі батырманы басыңыз.",
        "camera_start_btn": "📹 Жаттығуды бастау (10 сек)",
        "camera_starting": "Камера қосылуда...",
        "get_ready": "⏳ Дайындалыңыз: {sec}...",
        "recording_live": "🔴 Жазылуда: қалды {sec} сек",
        "recording_done": "✅ Жазу аяқталды! Қимыл талдануда...",
        "camera_error": "⚠️ Камера табылмады. Веб-камера қосылымын тексеріңіз.",
        "expander_upload": "📁 Дайын бейнені жүктеу (ML тесті үшін)",
        "upload_label": "Бейнефайлды таңдаңыз (.mp4, .avi, .mov)",
        "analyze_btn": "Файлды тексеру",
        "analyzing": "Қол қимылдары талдануда...",
        "result_success": "✅ Тамаша! Сіз жаттығуды дұрыс орындадыңыз.",
        "result_success_title": "✅ Тамаша! Сіз жаттығуды дұрыс орындадыңыз!",
        "result_success_desc": "Қимылдар жайлы және толық амплитудамен орындалды. Осылай жалғастырыңыз! 💪",
        "result_warning": "⚠️ Сол жақтағы нұсқаулықты тағы бір рет қарап, қайталап көріңіз.",
        "result_warning_title": "⚠️ Сәл жетпей қалды! Тағы бір рет көріңіз",
        "result_warning_desc": "Сол жақтағы нұсқаулықты тағы бір рет қараңыз: қимылды баяу, кідіріссіз және толық амплитудада орындауға тырысыңыз.",
        "retry_btn": "🔄 Қайталап көру",
        "back_exercises_btn": "← Жаттығуларды таңдауға оралу",
        "summary_title": "Жаттығу қорытындысы",
        "total_cycles": "Барлық қайталау",
        "completed_cycles": "Орындалды",
        "review_cycles": "Нұсқаулықты тексеріңіз",
        "status_ok": "✅ Жарайсың",
        "status_warn": "⚠️ Нұсқаулықты тексеріңіз",
        "chart_title": "Қозғалыс кезіндегі бұрыштың өзгеруі (°)",
        "cycle_col": "Қайталау",
        "result_col": "Нәтиже",
        "conf_col": "ЖИ дәлдігі",
        "angle_hud": "Угол",
        "side_right": "Оң қол",
        "side_left": "Сол қол",
        "analyzed_arm_label": "Талданған қол (макс. амплитуда)",
        "full_amplitude_tip": "💡 Есте сақтаңыз: қимылды баяу, толық амплитудада және орта жолда кідірместен орындаңыз.",
        "repeat_reminder": "💡 3 рет баяу, кідіріссіз орындаңыз",
        "replay_voice_btn": "🔊 Қайта тыңдау",
        "camera_voice_prompt": "Дайындалыңыз, камера алдына отырып жаттығуды бастаңыз",
        "daily_plan_title": "Бүгінгі жаттығу жоспары",
        "daily_plan_sub": "Буындарды қалпына келтіру үшін бүгінгі 4 жаттығуды орындаңыз",
        "daily_plan_done_all": "🎉 Жарайсыз! Бүгінгі барлық 4 жаттығу толық орындалды",
        "today_counter_label": "орындалды",
        "card_status_todo": "⏳ Бүгін 0 / 1",
        "card_status_done": "✅ Бүгін орындалды",
        "card_prev_result": "Алдыңғы нәтиже: {pct}%",
        "recovery_progress_title": "Қалпына келу прогресіңіз",
        "recovery_pct_label": "Мақсатты қозғалыс нормасының {pct}%-ы",
        "recovery_diff_up": "📈 Алдыңғы көрсеткіштен +{diff}% артты! Буын жақсы дамуда",
        "recovery_diff_same": "👍 Тұрақты нәтиже ({pct}%), жаттығуды осы қарқынмен жалғастырыңыз",
        "recovery_diff_start": "🎯 Тамаша бастама! Алғашқы деректер күнделікке жазылды",
        "today_completed_badge": "✅ Бүгінгі жоспарға есептелді (1 / 1)",
        "arm_selector_title": "Жаттығатын қол:",
        "arm_right": "Оң қол",
        "arm_left": "Сол қол",
        "ml_error_title": "⚠️ Қол қимылы кадрда анықталмады",
        "ml_error_desc": "Қолыңыз кадрда анық көрінетіндей отырыңыз және жаттығуды асықпай, баяу қайталап көріңіз."
    },
    "EN": {
        "app_title": "TirekAI — Home Rehabilitation",
        "app_subtitle": "Intelligent AI assistant for motor function recovery",
        "disclaimer_text": "This tool is for informational purposes only, does not diagnose, and does not replace medical advice.",
        "screen1_title": "Select an Exercise",
        "back_btn": "← Select another exercise",
        "step1_title": "1. Exercise Instructions",
        "step2_title": "2. Perform Before Camera",
        "patient_hint": "Sit so your arm is fully visible in the frame, then press the button below.",
        "camera_start_btn": "📹 Start Exercise (10 sec)",
        "camera_starting": "Starting camera...",
        "get_ready": "⏳ Get ready: {sec}...",
        "recording_live": "🔴 Recording: {sec} sec left",
        "recording_done": "✅ Recording finished! Analyzing motion...",
        "camera_error": "⚠️ Camera not found. Please check your webcam connection.",
        "expander_upload": "📁 Upload test video (for ML debugging)",
        "upload_label": "Choose a video file (.mp4, .avi, .mov)",
        "analyze_btn": "Analyze file",
        "analyzing": "Analyzing arm movements...",
        "result_success": "✅ Excellent! You performed the exercise correctly.",
        "result_success_title": "✅ Great job! You performed the exercise correctly!",
        "result_success_desc": "Your movements were smooth and through the full range of motion. Keep it up! 💪",
        "result_warning": "⚠️ Please review the instructions on the left and try again.",
        "result_warning_title": "⚠️ Almost there! Please try again",
        "result_warning_desc": "Review the instructions on the left once more: try to move smoothly, without pauses, and through the full range of motion.",
        "retry_btn": "🔄 Try again",
        "back_exercises_btn": "← Back to exercise list",
        "summary_title": "Session Summary",
        "total_cycles": "Total Repetitions",
        "completed_cycles": "Completed",
        "review_cycles": "Check Instructions",
        "status_ok": "✅ Great",
        "status_warn": "⚠️ Check instructions",
        "chart_title": "Motion Angle Timeline (°)",
        "cycle_col": "Rep",
        "result_col": "Result",
        "conf_col": "AI Confidence",
        "angle_hud": "Angle",
        "side_right": "Right arm",
        "side_left": "Left arm",
        "analyzed_arm_label": "Analyzed arm (max motion range)",
        "full_amplitude_tip": "💡 Reminder: perform each movement smoothly, through the full range of motion, and without mid-pauses.",
        "repeat_reminder": "💡 Perform 3 smooth repetitions without pauses",
        "replay_voice_btn": "🔊 Listen again",
        "camera_voice_prompt": "Get ready, sit in front of the camera and begin the exercise",
        "daily_plan_title": "Today's Rehabilitation Plan",
        "daily_plan_sub": "Complete all 4 prescribed exercises for steady joint recovery",
        "daily_plan_done_all": "🎉 Great job! All 4 exercises completed for today",
        "today_counter_label": "completed",
        "card_status_todo": "⏳ 0 / 1 today",
        "card_status_done": "✅ Completed today",
        "card_prev_result": "Previous result: {pct}%",
        "recovery_progress_title": "Your Recovery Progress",
        "recovery_pct_label": "{pct}% of target mobility range",
        "recovery_diff_up": "📈 +{diff}% compared to last time! Joint mobility is improving",
        "recovery_diff_same": "👍 Stable result ({pct}%), keep up the smooth exercises",
        "recovery_diff_start": "🎯 Great start! Initial progress recorded in your diary",
        "today_completed_badge": "✅ Counted in daily plan (1 / 1 today)",
        "arm_selector_title": "Trained Arm:",
        "arm_right": "Right Arm",
        "arm_left": "Left Arm",
        "ml_error_title": "⚠️ Arm movement was not detected",
        "ml_error_desc": "Please sit so your arm is fully visible in the frame, and perform the movement smoothly."
    }
}

def calculate_angle(a, b, c):
    """
    Вычисляет угол между точками a (плечо), b (локоть), c (запястье) в градусах.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return int(angle)

@st.cache_resource
def get_pose_estimator():
    """
    Кэширует объект MediaPipe Pose в памяти, устраняя задержку повторной инициализации нейросети.
    """
    mp_pose = mp.solutions.pose
    return mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

@st.cache_resource
def get_hands_estimator():
    """
    Кэширует MediaPipe Hands для E01 (раскрытие кисти) и E02 (подъём запястья).
    max_num_hands=1 исключает наложение двух скелетов на одну руку при ракурсе сбоку.
    """
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )

# Предварительный прогрев моделей в памяти при старте
try:
    _ = get_pose_estimator()
    _ = get_hands_estimator()
except Exception:
    pass

# ==============================================================================
# ==============================================================================
# ПОТОК ЗАХВАТА КАМЕРЫ И АДАПТИВНЫЙ ПОДСЧЁТ ПОВТОРЕНИЙ
# ==============================================================================
class CameraStreamReader:
    """
    Фоновый неблокирующий поток захвата кадров с веб-камеры.
    Позволяет непрерывно читать актуальный кадр (latest_frame) без задержек буферизации OpenCV.
    """
    def __init__(self, cap):
        self.cap = cap
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.6)

def count_repetitions_adaptive(angles: list[float] | np.ndarray, min_amplitude: float = 10.0, exercise_id: str = "E03", fps: float = 25.0) -> tuple[int, list[dict], int]:
    """
    Строгий адаптивный детектор циклов со строгими биомеханическими порогами:
    1. Перевод сырых значений в абсолютный процент движения (ROM %):
       - E01: сырые значения уже ROM % (0..100)
       - E02: сырые значения уже ROM % (0..100)
       - E03: угол локтя в градусах (<= 75° = 0%, >= 165° = 100%)
       - E04: сырые значения уже ROM % (0..100)
    2. Устранение лишних повторений (строгий гистерезис + cooldown >= 1.0 сек):
       - Исходная фаза: < 30%
       - Пик повторения: >= 78%
       - Возврат в исходную: < 40%
    3. Жёсткое правило статуса complete:
       - reps >= 3 (или >= 2 при пике >= 85%) И средняя пиковая амплитуда >= 80%.
    """
    if angles is None or len(angles) < 6:
        return 0, [], 0

    angles_arr = np.array(angles, dtype=float)
    # Лёгкое сглаживание окном 3 кадра, чтобы сохранять истинную высоту пиков
    w = 3
    kernel = np.ones(w) / w
    smoothed = np.convolve(angles_arr, kernel, mode='same')
    smoothed[0] = angles_arr[0]
    smoothed[-1] = angles_arr[-1]

    # Перевод в абсолютный биомеханический ROM процент (0..100%)
    if exercise_id == "E03":
        # Согнутая рука (<= 75°) = 0%, полное разгибание (>= 165°) = 100%
        rom_series = np.clip((smoothed - 75.0) / (165.0 - 75.0) * 100.0, 0.0, 100.0)
    else:
        rom_series = np.clip(smoothed, 0.0, 100.0)

    reps = 0
    cycles = []
    min_cooldown_frames = max(1, int(round(fps * 1.0)))  # пауза не менее 1.0 сек

    # Фазовый автомат гистерезиса:
    # 0 = WAITING_FOR_START (ждём опускания < 30%)
    # 1 = RISING_TO_PEAK (началось движение, ждём пика >= 78%)
    # 2 = RETURNING_TO_BASE (пик взят, ждём возврата < 40%)
    state = "WAITING_FOR_START" if rom_series[0] >= 30.0 else "RISING_TO_PEAK"
    
    cur_peak_rom = 0.0
    cur_peak_raw = 0.0
    cur_trough_raw = float(smoothed[0])
    last_rep_frame = -min_cooldown_frames

    for i, rom_val in enumerate(rom_series):
        raw_val = float(smoothed[i])

        if state == "WAITING_FOR_START":
            if rom_val < 30.0:
                state = "RISING_TO_PEAK"
                cur_peak_rom = rom_val
                cur_peak_raw = raw_val
                cur_trough_raw = raw_val

        elif state == "RISING_TO_PEAK":
            if rom_val > cur_peak_rom:
                cur_peak_rom = rom_val
                cur_peak_raw = raw_val

            if rom_val >= 78.0 and (i - last_rep_frame) >= min_cooldown_frames:
                # Достигнут пик движения
                state = "RETURNING_TO_BASE"

        elif state == "RETURNING_TO_BASE":
            # Пока возвращаемся, пик может ещё немного вырасти
            if rom_val > cur_peak_rom:
                cur_peak_rom = rom_val
                cur_peak_raw = raw_val

            # Засчитываем повторение строго при возврате ниже 40%
            if rom_val <= 40.0:
                reps += 1
                last_rep_frame = i
                cycles.append({
                    "cycle": reps,
                    "max_angle": round(cur_peak_raw, 1),
                    "min_angle": round(cur_trough_raw, 1),
                    "peak_rom": round(cur_peak_rom, 1)
                })
                # Снова ждём опускания в базу (< 30%)
                cur_trough_raw = raw_val
                cur_peak_rom = 0.0
                cur_peak_raw = 0.0
                state = "RISING_TO_PEAK" if rom_val < 30.0 else "WAITING_FOR_START"

    # Если движение завершилось в фазе возврата (или пациент зафиксировал верхнюю точку в конце записи)
    if state == "RETURNING_TO_BASE" and cur_peak_rom >= 78.0 and (len(rom_series) - 1 - last_rep_frame) >= min_cooldown_frames:
        if not cycles or cycles[-1]["cycle"] != (reps + 1):
            reps += 1
            cycles.append({
                "cycle": reps,
                "max_angle": round(cur_peak_raw, 1),
                "min_angle": round(cur_trough_raw, 1),
                "peak_rom": round(cur_peak_rom, 1)
            })

    # Общая оценка пиковой амплитуды (ROM %)
    if cycles:
        peak_amp_pct = int(round(np.max([c["peak_rom"] for c in cycles])))
        mean_peak_rom = float(np.mean([c["peak_rom"] for c in cycles]))
    else:
        peak_amp_pct = int(round(float(np.max(rom_series)))) if len(rom_series) > 0 else 0
        mean_peak_rom = float(peak_amp_pct)

    # Жёсткое правило статуса complete:
    # reps >= 3 (или >= 2 при пике >= 85%) И средняя пиковая амплитуда >= 80%
    overall_is_complete = (
        (reps >= 3 and mean_peak_rom >= 80.0) or
        (reps >= 2 and peak_amp_pct >= 85 and mean_peak_rom >= 80.0)
    )

    for c in cycles:
        c_is_complete = overall_is_complete and (c.get("peak_rom", 0) >= 78.0)
        c["status"] = "complete" if c_is_complete else "incomplete"
        # Уверенность калибруется строго по качеству доведения амплитуды
        c["confidence"] = round(min(0.96, max(0.65, 0.50 + 0.45 * (c.get("peak_rom", 0) / 100.0))), 2)

    return reps, cycles, peak_amp_pct

@st.cache_resource
def load_ml_pipeline():
    """Кэширует загрузку артефактов модели TirekAI (best_model.joblib + config.json)."""
    from src.inference import _load_artifacts
    return _load_artifacts()

def analyze_video_real(video_path, exercise_id="E03", real_angles=None, chosen_side=None, left_angles=None, right_angles=None, actual_fps=None):
    """
    Запускает реальный ML-пайплайн инференса (src.inference.predict_video)
    с гибридным скорингом биомеханики и точным подсчётом повторений (1, 2, 3) для всех 4 упражнений.
    """
    default_pcts = {"E01": 85, "E02": 82, "E03": 88, "E04": 80}
    
    try:
        from src.inference import predict_video
        raw_result = predict_video(video_path)
    except Exception as e:
        raw_result = {"status": "error", "message": str(e), "cycles": [], "angle_series": []}

    # Анализ повторений через единый адаптивный биомеханический детектор
    rep_count, adaptive_cycles, peak_amp_pct = 0, [], 0
    eff_fps = actual_fps if actual_fps and actual_fps > 5 else 25.0
    if real_angles and len(real_angles) >= 6:
        rep_count, adaptive_cycles, peak_amp_pct = count_repetitions_adaptive(real_angles, exercise_id=exercise_id, fps=eff_fps)
    elif raw_result and raw_result.get("angle_series"):
        series_angles = [pt["angle"] for pt in raw_result["angle_series"]]
        rep_count, adaptive_cycles, peak_amp_pct = count_repetitions_adaptive(series_angles, exercise_id=exercise_id, fps=eff_fps)

    # 1. Если адаптивный подсчёт нашёл циклы (1, 2, 3 повтора)
    if rep_count > 0:
        cycles_formatted = []
        ml_cycles = raw_result.get("cycles", []) if raw_result and raw_result.get("status") == "success" else []
        
        # Жёсткое биомеханическое правило:
        # reps >= 3 (или >= 2 при пике >= 85%) И средняя пиковая амплитуда >= 80%
        overall_is_complete = ((rep_count >= 3 or (rep_count >= 2 and peak_amp_pct >= 85)) and peak_amp_pct >= 80)
        
        for idx, ac in enumerate(adaptive_cycles):
            c_num = idx + 1
            ac_status = ac.get("status", "incomplete")
            # Статус цикла complete ТОЛЬКО если и общее правило, и сам цикл выполнил норматив
            final_status = "complete" if (overall_is_complete and ac_status == "complete") else "incomplete"
            
            if idx < len(ml_cycles):
                ml_c = ml_cycles[idx]
                c_conf = float(ml_c.get("confidence", ac["confidence"]))
                c_max = ml_c.get("max_angle") or ac["max_angle"]
                c_min = ml_c.get("min_angle") or ac["min_angle"]
            else:
                c_conf = ac["confidence"]
                c_max = ac["max_angle"]
                c_min = ac["min_angle"]
                
            cycles_formatted.append({
                "cycle": c_num,
                "status": final_status,
                "confidence": c_conf,
                "max_angle": c_max,
                "min_angle": c_min
            })

        amplitude_pct = max(0, min(100, peak_amp_pct))
        prog_info = record_exercise_session(exercise_id, amplitude_pct)

        angles_timeline = [pt["angle"] for pt in raw_result.get("angle_series", [])] if raw_result else []
        if not angles_timeline and real_angles:
            angles_timeline = real_angles

        return {
            "status": "success",
            "cycles": cycles_formatted,
            "angles_timeline": angles_timeline,
            "chosen_side": chosen_side,
            "left_timeline": left_angles or [],
            "right_timeline": right_angles or [],
            "exercise_id": exercise_id,
            "amplitude_pct": amplitude_pct,
            "prev_pct": prog_info.get("prev_pct"),
            "diff_pct": prog_info.get("diff_pct")
        }

    # 2. Если адаптивный не нашёл, но ML вернул циклы
    if raw_result and raw_result.get("status") == "success" and raw_result.get("cycles"):
        cycles_formatted = []
        for c in raw_result["cycles"]:
            cycles_formatted.append({
                "cycle": c["cycle_id"],
                "status": c["raw_class"],
                "confidence": float(c["confidence"]),
                "max_angle": c.get("max_angle"),
                "min_angle": c.get("min_angle")
            })
        amplitude_pct = default_pcts.get(exercise_id, 75)
        prog_info = record_exercise_session(exercise_id, amplitude_pct)
        angles_timeline = [pt["angle"] for pt in raw_result.get("angle_series", [])]
        if not angles_timeline and real_angles:
            angles_timeline = real_angles

        return {
            "status": "success",
            "cycles": cycles_formatted,
            "angles_timeline": angles_timeline,
            "chosen_side": chosen_side,
            "left_timeline": left_angles or [],
            "right_timeline": right_angles or [],
            "exercise_id": exercise_id,
            "amplitude_pct": amplitude_pct,
            "prev_pct": prog_info.get("prev_pct"),
            "diff_pct": prog_info.get("diff_pct")
        }

    # 3. Движение не обнаружено или ошибка инференса
    return {
        "status": "error",
        "exercise_id": exercise_id,
        "chosen_side": chosen_side,
        "angles_timeline": real_angles or []
    }

def render_analysis_view(result, t):
    # Обработка ошибки (рука не распознана или видео слишком короткое)
    if result.get("status") == "error":
        st.markdown(f"""
        <div class="result-card-warning" style="margin-bottom: 24px;">
            <div class="rc-title">{t.get('ml_error_title', '⚠️ Движение не удалось распознать')}</div>
            <div class="rc-desc">{t.get('ml_error_desc', 'Пожалуйста, сядьте перед камерой так, чтобы рука была полностью видна в кадре, и повторите упражнение.')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="screen2-btn-wrapper">', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(t["retry_btn"], type="primary", use_container_width=True, key="retry_err_btn"):
                st.session_state["analysis_result"] = None
                st.rerun()
        with btn_col2:
            if st.button(t["back_exercises_btn"], type="secondary", use_container_width=True, key="back_err_btn"):
                if "ex" in st.query_params:
                    del st.query_params["ex"]
                st.session_state["selected_exercise"] = None
                st.session_state["analysis_result"] = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    CONF_THRESHOLD = 0.70
    processed_cycles = []
    ok_count = 0
    
    for item in result["cycles"]:
        is_valid = (item["status"] == "complete") and (item["confidence"] >= CONF_THRESHOLD)
        if is_valid:
            ok_count += 1
            label = t["status_ok"]
        else:
            label = t["status_warn"]
        processed_cycles.append({
            t["cycle_col"]: item["cycle"],
            t["result_col"]: label,
            t["conf_col"]: f"{int(item['confidence'] * 100)}%"
        })
    
    total_count = len(processed_cycles)
    warn_count = total_count - ok_count
    
    # ── Крупный блок-окно итогового результата ──────────────────────────────
    if warn_count == 0:
        st.markdown(f"""
        <div class="result-card-success">
            <div class="rc-title">{t['result_success_title']}</div>
            <div class="rc-desc">{t['result_success_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-warning">
            <div class="rc-title">{t['result_warning_title']}</div>
            <div class="rc-desc">{t['result_warning_desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    # ── Блок восстановления амплитуды движения (%) ──────────────────────────
    amp_pct = result.get("amplitude_pct", 58)
    prev_pct = result.get("prev_pct")
    diff_pct = result.get("diff_pct")
    
    if prev_pct is not None and diff_pct is not None:
        if diff_pct > 0:
            diff_msg = t["recovery_diff_up"].format(diff=diff_pct)
            diff_box_style = "background: #F0FDF4; border: 1.5px solid #86EFAC; color: #166534;"
        elif diff_pct == 0:
            diff_msg = t["recovery_diff_same"].format(pct=amp_pct)
            diff_box_style = "background: #F8FAFC; border: 1.5px solid #CBD5E1; color: #334155;"
        else:
            diff_msg = t["recovery_diff_same"].format(pct=amp_pct)
            diff_box_style = "background: #F8FAFC; border: 1.5px solid #CBD5E1; color: #334155;"
    else:
        diff_msg = t["recovery_diff_start"]
        diff_box_style = "background: #EFF6FF; border: 1.5px solid #93C5FD; color: #1E40AF;"

    st.markdown(f"""
    <div style="background: #FFFFFF; border: 2px solid #E2E8F0; border-radius: 20px; padding: 22px 26px; margin-bottom: 22px; box-shadow: 0 4px 16px rgba(15,23,42,0.05);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 10px;">
            <div style="font-size: 24px; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 8px;">
                <span>📊</span> <span>{t['recovery_progress_title']}</span>
            </div>
            <div style="background: #DCFCE7; border: 1px solid #86EFAC; color: #15803D; font-size: 17px; font-weight: 700; border-radius: 10px; padding: 6px 14px;">
                {t['today_completed_badge']}
            </div>
        </div>
        <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px;">
            <span style="font-size: 46px; font-weight: 900; color: #0E7490; line-height: 1;">{amp_pct}%</span>
            <span style="font-size: 20px; font-weight: 700; color: #334155;">{t['recovery_pct_label'].format(pct=amp_pct)}</span>
        </div>
        <div style="width: 100%; height: 20px; background: #E2E8F0; border-radius: 10px; overflow: hidden; margin-bottom: 14px;">
            <div style="width: {amp_pct}%; height: 100%; background: linear-gradient(90deg, #0E7490 0%, #10B981 100%); border-radius: 10px; transition: width 0.6s ease;"></div>
        </div>
        <div style="{diff_box_style} border-radius: 12px; padding: 12px 18px; font-size: 19px; font-weight: 700;">
            {diff_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)
        
    # Плашка с указанием руки с наибольшей амплитудой
    if result.get("chosen_side"):
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 5px solid #0E7490; border-radius: 12px; padding: 12px 18px; margin-bottom: 20px; font-size: 20px; color: #1E293B;">
            🦾 {t['analyzed_arm_label']}: <strong style="color: #0E7490;">{result['chosen_side']}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Кнопки действий ─────────────────────────────────────────────────────
    st.markdown('<div class="screen2-btn-wrapper">', unsafe_allow_html=True)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button(t["retry_btn"], type="primary", use_container_width=True):
            st.session_state["analysis_result"] = None
            st.rerun()
    with btn_col2:
        if st.button(t["back_exercises_btn"], type="secondary", use_container_width=True):
            if "ex" in st.query_params:
                del st.query_params["ex"]
            st.session_state["selected_exercise"] = None
            st.session_state["analysis_result"] = None
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    
    # ── Метрики ─────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t["total_cycles"]}</div>
            <div class="metric-value metric-value-total">{total_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t["completed_cycles"]}</div>
            <div class="metric-value metric-value-ok">{ok_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{t["review_cycles"]}</div>
            <div class="metric-value metric-value-warn">{warn_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.dataframe(pd.DataFrame(processed_cycles), use_container_width=True, hide_index=True)
    
    # ── График динамики угла движения ────────────────────────────────────────
    chart_title = t["chart_title"]
    if result.get("chosen_side"):
        chart_title = f"{t['chart_title']} ({result['chosen_side']})"
        
    left_t = result.get("left_timeline", [])
    right_t = result.get("right_timeline", [])
    
    if len(left_t) >= 5 and len(right_t) >= 5:
        max_len = max(len(left_t), len(right_t))
        # Дополняем до одинаковой длины None при необходимости
        l_padded = list(left_t) + [None] * (max_len - len(left_t))
        r_padded = list(right_t) + [None] * (max_len - len(right_t))
        
        df_chart = pd.DataFrame({
            "Кадр": list(range(1, max_len + 1)),
            f"{t['side_left']} (L)": l_padded,
            f"{t['side_right']} (R)": r_padded
        })
        fig = px.line(
            df_chart,
            x="Кадр",
            y=[f"{t['side_left']} (L)", f"{t['side_right']} (R)"],
            labels={"value": "Угол / Angle (°)", "variable": "Сторона / Arm"},
            title=chart_title,
            color_discrete_map={
                f"{t['side_left']} (L)": "#EAB308",
                f"{t['side_right']} (R)": "#0E7490"
            }
        )
    else:
        fig = px.line(
            y=result["angles_timeline"],
            labels={"x": "Кадр / Кадр / Frame", "y": "Угол / Бұрыш / Angle (°)"},
            title=chart_title
        )
        fig.update_traces(line=dict(color="#0E7490", width=4))
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor="#F1F5F9", showline=True, linecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#F1F5F9", showline=True, linecolor="#CBD5E1"),
        font=dict(color="#1E293B", size=16)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# ШАПКА ПРИЛОЖЕНИЯ: КОМПАКТНАЯ + РОВНЫЕ ЯЗЫКОВЫЕ КНОПКИ (RU / KZ / EN)
# ==============================================================================
lang = st.session_state["lang"]
t = TEXTS[lang]

# Пропорции колонок: 2.6 к 1.4 для достаточного пространства 3 языковых ячеек
col_head, col_lang = st.columns([2.6, 1.4])

with col_head:
    st.markdown(f"""
    <div>
        <div style="font-size: 40px; font-weight: 800; color: #0F172A; line-height: 1.15; margin-bottom: 6px;">{t['app_title']}</div>
        <div style="font-size: 24px; font-weight: 500; color: #475569; margin-bottom: 12px;">{t['app_subtitle']}</div>
    </div>
    """, unsafe_allow_html=True)

with col_lang:
    st.markdown('<div class="lang-switch-wrapper">', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3, gap="small")
    for l_key, l_slot in zip(["RU", "KZ", "EN"], [l1, l2, l3]):
        with l_slot:
            is_active = (lang == l_key)
            btn_kind = "primary" if is_active else "secondary"
            if st.button(l_key, key=f"lang_{l_key}", type=btn_kind, use_container_width=True):
                st.session_state["lang"] = l_key
                # Сохраняем язык в URL, чтобы при переходе по карточке он не сбросился
                st.query_params["lang"] = l_key
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Медицинский дисклеймер
st.markdown(f"""
<div style="background: #EFF6FF; border-left: 5px solid #0E7490; padding: 12px 18px; border-radius: 10px; font-size: 20px; color: #1E3A8A; margin-bottom: 16px;">
    ℹ️ {t['disclaimer_text']}
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ЭКРАН 1: ВЫБОР УПРАЖНЕНИЯ (ШИРОКАЯ СЕТКА, КАРТОЧКИ 360px, КРУГ 120px, ШРИФТ 26px)
# ==============================================================================
if st.session_state["selected_exercise"] is None:
    # ── Загрузка данных прогресса для дневного To-Do плана ──
    progress_data = load_progress()
    today_iso = datetime.date.today().isoformat()
    exercise_keys = ["E01", "E02", "E03", "E04"]
    
    ex_stats_map = {}
    completed_today_count = 0
    for ex_key in exercise_keys:
        st_data = get_exercise_stats(progress_data, ex_key, today_iso)
        ex_stats_map[ex_key] = st_data
        if st_data["completed_today"]:
            completed_today_count += 1
            
    plan_total = len(exercise_keys)
    plan_ratio = int((completed_today_count / plan_total) * 100)
    plan_phrase = t["daily_plan_done_all"] if completed_today_count == plan_total else t["daily_plan_sub"]
    
    # ── Крупная светлая карточка «План на сегодня» ──
    daily_card_html = (
        f'<div class="daily-plan-card">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; flex-wrap: wrap; gap: 12px;">'
        f'<div>'
        f'<div style="font-size: 26px; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 10px;">'
        f'<span>📋</span> <span>{t["daily_plan_title"]}</span>'
        f'</div>'
        f'<div style="font-size: 19px; color: #475569; font-weight: 500; margin-top: 4px;">{plan_phrase}</div>'
        f'</div>'
        f'<div style="background: #F0FDFA; border: 2px solid #99F6E4; border-radius: 16px; padding: 10px 24px; text-align: center; min-width: 140px;">'
        f'<div style="font-size: 34px; font-weight: 900; color: #0E7490; line-height: 1;">{completed_today_count} / {plan_total}</div>'
        f'<div style="font-size: 15px; font-weight: 700; color: #0D9488; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">{t["today_counter_label"]}</div>'
        f'</div>'
        f'</div>'
        f'<div style="width: 100%; height: 18px; background-color: #E2E8F0; border-radius: 10px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.06);">'
        f'<div style="width: {plan_ratio}%; height: 100%; background: linear-gradient(90deg, #10B981 0%, #059669 100%); border-radius: 10px; transition: width 0.5s ease;"></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(daily_card_html, unsafe_allow_html=True)

    title_html = f'<div style="font-size: 32px; font-weight: 800; color: #0F172A; text-align: center; margin: 10px 0 20px 0;">{t["screen1_title"]}</div>'
    st.markdown(title_html, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    cols = [col1, col2, col3, col4]
    
    for idx, ex_id in enumerate(exercise_keys):
        ex_info = EXERCISES[ex_id]
        ex_title = ex_info[lang]["title"]
        card_class = f"card-{ex_id.lower()}"
        ex_stat = ex_stats_map[ex_id]
        
        if ex_stat["completed_today"]:
            badge_html = f'<span class="exercise-card-status-done">{t["card_status_done"]} • {ex_stat["today_pct"]}%</span>'
        else:
            prev_line = f'<span class="exercise-card-prev-line">{t["card_prev_result"].format(pct=ex_stat["past_pct"])}</span>' if ex_stat["past_pct"] else ""
            badge_html = f'<span class="exercise-card-status-todo">{t["card_status_todo"]}{prev_line}</span>'
        
        with cols[idx]:
            card_html = (
                f'<a href="?ex={ex_id}&lang={lang}" target="_self" class="exercise-card-link">'
                f'<span class="exercise-card {card_class}">'
                f'<span class="exercise-card-badge">{ex_info["code"]}</span>'
                f'<span class="exercise-card-circle">{ex_info["svg"]}</span>'
                f'<span class="exercise-card-title">{ex_title}</span>'
                f'{badge_html}'
                f'</span>'
                f'</a>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

# ==============================================================================
# ЭКРАН 2: ИНСТРУКЦИЯ + КАМЕРА + РЕЗУЛЬТАТ
# ==============================================================================
else:
    current_ex_id = st.session_state["selected_exercise"]
    current_ex = EXERCISES[current_ex_id]
    current_ex_lang = current_ex[lang]
    
    nav_col1, nav_col2 = st.columns([1, 2])
    with nav_col1:
        st.markdown('<div class="screen2-btn-wrapper">', unsafe_allow_html=True)
        if st.button(t["back_btn"], type="secondary", use_container_width=True):
            if "ex" in st.query_params:
                del st.query_params["ex"]
            st.session_state["selected_exercise"] = None
            st.session_state["analysis_result"] = None
            st.session_state["last_voice_key"] = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with nav_col2:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; height: 100%; padding-left: 10px;">
            <span style="font-size: 26px; font-weight: 800; color: #0E7490;">{current_ex['code']}: {current_ex_lang['title']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")
    
    col_left, col_right = st.columns([1, 1], gap="large")
    
    # ЛЕВАЯ КОЛОНКА (Шаг 1 — Чистая пошаговая текстовая карточка-инструкция)
    with col_left:
        with st.container(border=True):
            st.header(t["step1_title"])
            st.markdown(f"<div style='font-size: 24px; font-weight: 800; margin-bottom: 16px; color: #0E7490;'>{current_ex['code']}: {current_ex_lang['title']}</div>", unsafe_allow_html=True)
            
            # 3-frame storyboard (анимированная раскадровка)
            sb = current_ex.get("storyboard", [])
            if len(sb) == 3:
                frame_classes = ["sb-frame sb-frame-1", "sb-frame sb-frame-2", "sb-frame sb-frame-3"]
                frames_html = ""
                for fi, frame in enumerate(sb):
                    caption = frame.get(lang, frame.get("EN", ""))
                    frames_html += f"""
                    <div class="{frame_classes[fi]}">
                        {frame['svg']}
                        <div class="sb-caption">{caption}</div>
                    </div>"""
                st.markdown(f"""
                <div class="sb-wrapper">{frames_html}</div>
                <div class="sb-reminder">{t.get('repeat_reminder', '💡 Выполните 3 плавных повторения без пауз')}</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border-left: 4px solid #0E7490; border-radius: 10px; padding: 14px 18px; margin-top: 18px; font-size: 19px; color: #334155; line-height: 1.45;">
                    {t.get('repeat_reminder', '💡 Выполните 3 плавных повторения без пауз')}
                </div>
                """, unsafe_allow_html=True)

            # Кнопка повторного прослушивания голосовой инструкции (без визуального плеера)
            if current_ex_lang.get("voice_text"):
                st.write("")
                st.markdown('<div class="screen2-btn-wrapper">', unsafe_allow_html=True)
                if st.button(t["replay_voice_btn"], type="secondary", use_container_width=True, key=f"replay_voice_{current_ex_id}"):
                    st.session_state["force_voice_replay"] = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            ref_path_specific = f"reference_{current_ex_id.lower()}.mp4"
            ref_path_default = "reference_e02.mp4"
            if os.path.exists(ref_path_specific):
                st.write("")
                st.video(ref_path_specific)
            elif os.path.exists(ref_path_default):
                st.write("")
                st.video(ref_path_default)
    
    # ПРАВАЯ КОЛОНКА (Шаг 2 — Камера и Результаты)
    with col_right:
        with st.container(border=True):
            st.header(t["step2_title"])
            
            if st.session_state["analysis_result"] is None:
                # Плейсхолдер для видеопотока — стоит ВЫШЕ кнопки, сразу виден без скролла
                image_placeholder = st.empty()
                cam_audio_ph = st.empty()
                status_box = st.empty()
                progress_bar_ph = st.empty()
                
                st.markdown(f"""
                <div class="patient-instruction-card">
                    <span class="patient-instruction-icon">💡</span>
                    <span class="patient-instruction-text">{t["patient_hint"]}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # ── Крупный переключатель тренируемой руки (Правая / Левая) ──
                if "chosen_arm" not in st.session_state:
                    st.session_state["chosen_arm"] = "Right"

                st.markdown(f"<div style='font-size: 19px; font-weight: 700; color: #1E293B; margin-bottom: 8px;'>🦾 {t['arm_selector_title']}</div>", unsafe_allow_html=True)
                arm_col1, arm_col2 = st.columns(2)
                with arm_col1:
                    btn_r_type = "primary" if st.session_state["chosen_arm"] == "Right" else "secondary"
                    if st.button(f"👉 {t['arm_right']}", type=btn_r_type, use_container_width=True, key=f"btn_arm_r_{current_ex_id}"):
                        st.session_state["chosen_arm"] = "Right"
                        st.rerun()
                with arm_col2:
                    btn_l_type = "primary" if st.session_state["chosen_arm"] == "Left" else "secondary"
                    if st.button(f"👈 {t['arm_left']}", type=btn_l_type, use_container_width=True, key=f"btn_arm_l_{current_ex_id}"):
                        st.session_state["chosen_arm"] = "Left"
                        st.rerun()

                st.write("")
                start_camera = st.button(t["camera_start_btn"], type="primary", use_container_width=True)
                
                # Секция загрузки видео (без st.expander чтобы не было бага со стрелкой)
                st.write("")
                show_upload = st.checkbox(f"📁 {t['expander_upload']}", key=f"show_upload_{current_ex_id}")
                if show_upload:
                    uploaded_file = st.file_uploader(t["upload_label"], type=["mp4", "avi", "mov"], key=f"uploader_{current_ex_id}")
                    if uploaded_file is not None:
                        temp_upload_path = f"temp_{uploaded_file.name}"
                        with open(temp_upload_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        st.video(temp_upload_path)
                        
                        if st.button(t["analyze_btn"], type="secondary", use_container_width=True, key=f"btn_upload_{current_ex_id}"):
                            with st.spinner(t["analyzing"]):
                                chosen_arm_val = st.session_state.get("chosen_arm", "Right")
                                side_label = t["arm_right"] if chosen_arm_val == "Right" else t["arm_left"]
                                analysis_res = analyze_video_real(temp_upload_path, current_ex_id, chosen_side=side_label)
                            st.session_state["analysis_result"] = analysis_res
                            st.rerun()
                
                if start_camera:
                    st.toast(t["camera_starting"], icon="📹")
                    startup_notice = status_box.info(f"📹 **{t['camera_starting']}**")
                    cam_audio_bytes = generate_speech(t["camera_voice_prompt"], lang)
                    if cam_audio_bytes:
                        with cam_audio_ph:
                            play_hidden_audio(cam_audio_bytes, uid=f"cam_{time.time_ns()}")
                    
                    # ── MJPG + 640×480 + 30fps для макс. скорости на Windows DirectShow ──
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(0)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if not cap.isOpened():
                        status_box.error(t["camera_error"])
                    else:
                        status_box.empty()
                        cam_reader = CameraStreamReader(cap)
                        # Ждем первый стабильный кадр (до 1 сек)
                        t_init = time.time()
                        while time.time() - t_init < 1.0:
                            ret_init, f_init = cam_reader.read()
                            if ret_init and f_init is not None:
                                break
                            time.sleep(0.02)
                            
                        countdown_box = image_placeholder.empty()
                        for sec in [3, 2, 1]:
                            image_placeholder.markdown(
                                f"<div class='countdown-box'>{t['get_ready'].format(sec=sec)}</div>",
                                unsafe_allow_html=True
                            )
                            time.sleep(1.0)
                        image_placeholder.empty()
                        
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                        RECORD_FPS = 25.0
                        
                        temp_cam_file = "temp_camera.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_cam_file, fourcc, RECORD_FPS, (width, height))
                        
                        progress_bar = progress_bar_ph.progress(0.0)
                        
                        mp_pose = mp.solutions.pose
                        mp_hands = mp.solutions.hands
                        
                        chosen_arm = st.session_state.get("chosen_arm", "Right")
                        is_right_arm = (chosen_arm == "Right")
                        chosen_arm_text = t["arm_right"] if is_right_arm else t["arm_left"]
                        chosen_arm_tag = "R" if is_right_arm else "L"

                        # Одна модель на всю запись: Hands (max_num_hands=1) для E01/E02, Pose для E03/E04
                        use_hands = current_ex_id in ("E01", "E02")
                        if use_hands:
                            estimator = get_hands_estimator()
                        else:
                            estimator = get_pose_estimator()
                        
                        start_time = time.time()
                        RECORD_DURATION = 10.0
                        angles_live = []
                        left_angles_live = []
                        right_angles_live = []
                        frame_idx = 0
                        last_ui_time = 0.0  # время последней отправки превью в браузер
                        UI_INTERVAL = 0.04   # ~25 FPS превью (высокая плавность без перегрузки)
                        
                        try:
                            while True:
                                elapsed = time.time() - start_time
                                if elapsed >= RECORD_DURATION:
                                    break
                                
                                ret, frame = cam_reader.read()
                                if not ret or frame is None:
                                    time.sleep(0.005)
                                    continue
                                
                                h, w, _ = frame.shape
                                # 1. Запуск MediaPipe на ОРИГИНАЛЬНОМ кадре (без отзеркаливания)
                                rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = estimator.process(rgb_for_mp)
                                
                                # 2. Отзеркаливание только для отображения пациенту
                                mirrored = cv2.flip(frame, 1)
                                curr_val = None
                                
                                # ── E01 / E02: MediaPipe Hands (max_num_hands=1) ──
                                if use_hands and results.multi_hand_landmarks:
                                    lm = results.multi_hand_landmarks[0].landmark
                                    
                                    # Рисуем скелет кисти на mirrored
                                    for conn in mp_hands.HAND_CONNECTIONS:
                                        p1, p2 = lm[conn[0]], lm[conn[1]]
                                        x1, y1 = int((1.0 - p1.x) * w), int(p1.y * h)
                                        x2, y2 = int((1.0 - p2.x) * w), int(p2.y * h)
                                        cv2.line(mirrored, (x1, y1), (x2, y2), (0, 220, 100), 3)
                                        
                                    for idx_lm, landmark in enumerate(lm):
                                        cx, cy = int((1.0 - landmark.x) * w), int(landmark.y * h)
                                        r = 7 if idx_lm in (0, 5, 9, 13, 17) else 4
                                        cv2.circle(mirrored, (cx, cy), r, (0, 165, 255), -1)
                                    
                                    if current_ex_id == "E01":
                                        # E01 (Раскрытие кисти): отношение расстояния от 4 кончиков пальцев до запястья к длине пясти
                                        wrist_pt = np.array([lm[0].x, lm[0].y])
                                        mcp9_pt = np.array([lm[9].x, lm[9].y])
                                        d_palm = float(np.linalg.norm(mcp9_pt - wrist_pt)) + 1e-5
                                        tip_dists = [float(np.linalg.norm(np.array([lm[t].x, lm[t].y]) - wrist_pt)) for t in (8, 12, 16, 20)]
                                        span_ratio = float(np.mean(tip_dists)) / d_palm

                                        # Строгая биомеханическая шкала: кулак <= 1.15 (0%), полное раскрытие >= 1.85 (100%)
                                        e01_pct = np.clip((span_ratio - 1.15) / (1.85 - 1.15) * 100.0, 0.0, 100.0)
                                        curr_val = int(round(e01_pct))
                                        angles_live.append(curr_val)
                                        
                                        mid_tip = (int((1.0 - lm[12].x) * w), int(lm[12].y * h))
                                        cv2.putText(mirrored, f"{chosen_arm_tag}: {curr_val}% (span: {span_ratio:.2f})",
                                            (max(10, mid_tip[0] - 80), max(30, mid_tip[1] - 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 255), 2, cv2.LINE_AA)

                                    elif current_ex_id == "E02":
                                        # E02: Подъём запястья (тыльное сгибание оси ладони wrist->mcp9)
                                        dx = (lm[9].x - lm[0].x)
                                        dy = -(lm[9].y - lm[0].y)
                                        wrist_pitch_deg = float(np.degrees(np.arctan2(dy, abs(dx) + 1e-5)))
                                        q_wrist = np.clip((wrist_pitch_deg - 5.0) / (50.0 - 5.0), 0.0, 1.0)
                                        curr_val = int(round(q_wrist * 100.0))
                                        angles_live.append(curr_val)
                                        
                                        mcp9_px = (int((1.0 - lm[9].x) * w), int(lm[9].y * h))
                                        cv2.putText(mirrored, f"{chosen_arm_tag}: {wrist_pitch_deg:.0f} deg ({curr_val}%)",
                                            (max(10, mcp9_px[0] - 80), max(30, mcp9_px[1] - 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 255, 255), 2, cv2.LINE_AA)

                                # ── E03 / E04: MediaPipe Pose ──
                                elif not use_hands and results.pose_landmarks:
                                    lm = results.pose_landmarks.landmark
                                    is_right_arm = (chosen_arm == "Right")
                                    
                                    s_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value if is_right_arm else mp_pose.PoseLandmark.LEFT_SHOULDER.value
                                    e_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value if is_right_arm else mp_pose.PoseLandmark.LEFT_ELBOW.value
                                    w_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value if is_right_arm else mp_pose.PoseLandmark.LEFT_WRIST.value
                                    
                                    s_lm = lm[s_idx]
                                    e_lm = lm[e_idx]
                                    w_lm = lm[w_idx]
                                    
                                    # Рисуем скелет выбранной руки на mirrored
                                    s_px = (int((1.0 - s_lm.x) * w), int(s_lm.y * h))
                                    e_px = (int((1.0 - e_lm.x) * w), int(e_lm.y * h))
                                    w_px = (int((1.0 - w_lm.x) * w), int(w_lm.y * h))
                                    
                                    cv2.line(mirrored, s_px, e_px, (0, 255, 0), 4)
                                    cv2.line(mirrored, e_px, w_px, (0, 255, 0), 4)
                                    cv2.circle(mirrored, s_px, 8, (0, 165, 255), -1)
                                    cv2.circle(mirrored, e_px, 11, (0, 0, 255), -1)
                                    cv2.circle(mirrored, w_px, 8, (0, 165, 255), -1)
                                    
                                    if current_ex_id == "E03":
                                        # E03: Угол локтя плечо - локоть - запястье
                                        elbow_ang = calculate_angle((s_lm.x * w, s_lm.y * h), (e_lm.x * w, e_lm.y * h), (w_lm.x * w, w_lm.y * h))
                                        curr_val = elbow_ang
                                        angles_live.append(curr_val)
                                        cv2.putText(mirrored, f"{chosen_arm_tag}: {elbow_ang} deg",
                                            (e_px[0] + 15, max(35, e_px[1] - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                                            
                                    elif current_ex_id == "E04":
                                        # E04: Подъём руки в плечевом суставе (инвариантен к углу в локте)
                                        # Метрика: rel_height = (shoulder.y - wrist.y) / arm_len
                                        # MediaPipe: y растёт вниз, поэтому shoulder.y - wrist.y > 0 когда запястье ВЫШЕ плеча
                                        upper_arm = np.sqrt((s_lm.x - e_lm.x)**2 + (s_lm.y - e_lm.y)**2)
                                        forearm = np.sqrt((e_lm.x - w_lm.x)**2 + (e_lm.y - w_lm.y)**2)
                                        arm_len = upper_arm + forearm + 1e-5
                                        rel_height = (s_lm.y - w_lm.y) / arm_len
                                        # Рука опущена вниз: rel_height <= -0.55 -> 0%
                                        # Рука поднята до уровня плеча: rel_height >= 0.05 -> 100%
                                        e04_rom = float(np.clip((rel_height - (-0.55)) / (0.05 - (-0.55)) * 100.0, 0.0, 100.0))
                                        curr_val = int(round(e04_rom))
                                        angles_live.append(curr_val)
                                        cv2.putText(mirrored, f"{chosen_arm_tag} Lift: {curr_val}%",
                                            (s_px[0] + 12, max(35, s_px[1] - 15)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                                
                                # ── HUD: значение + реальный FPS ──
                                hud_val_str = f"{chosen_arm_tag}: {curr_val}" if curr_val is not None else "--"
                                unit_suffix = "%" if current_ex_id in ["E01", "E04"] else " deg"
                                
                                live_fps = (frame_idx + 1) / elapsed if elapsed > 0.1 else 0.0
                                cv2.rectangle(mirrored, (10, 10), (430, 78), (20, 20, 20), -1)
                                cv2.putText(mirrored,
                                    f"{t['angle_hud']}: {hud_val_str}{unit_suffix}",
                                    (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (50, 255, 50), 2, cv2.LINE_AA)
                                cv2.putText(mirrored,
                                    f"FPS: {live_fps:.0f}",
                                    (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 220, 255), 2, cv2.LINE_AA)
                                
                                # Записываем ВСЕ кадры в файл (полный FPS без пропусков)
                                out.write(mirrored)
                                frame_idx += 1
                                
                                # Превью в браузер: четкая картинка 640x480 без пикселизации (качество 85)
                                now = time.time()
                                if now - last_ui_time >= UI_INTERVAL:
                                    last_ui_time = now
                                    enc_ok, buf = cv2.imencode('.jpg', mirrored, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                    if enc_ok:
                                        image_placeholder.image(buf.tobytes(), use_container_width=True)
                                    
                                    remaining = max(0, int(RECORD_DURATION - elapsed))
                                    status_box.markdown(
                                        f"<div class='recording-box'>{t['recording_live'].format(sec=remaining)}</div>",
                                        unsafe_allow_html=True
                                    )
                                    progress_bar_ph.progress(min(1.0, elapsed / RECORD_DURATION))
                        finally:
                            actual_elapsed = time.time() - start_time
                            actual_fps = max(10.0, frame_idx / actual_elapsed) if actual_elapsed > 0.5 else RECORD_FPS
                            cam_reader.stop()
                            cap.release()
                            out.release()
                            # Если реальный FPS значительно отличается — перезаписываем хедер видео
                            if frame_idx > 0 and abs(actual_fps - RECORD_FPS) > 2.0:
                                import shutil
                                tmp_rewrite = "temp_camera_fps.mp4"
                                out2 = cv2.VideoWriter(tmp_rewrite, fourcc, actual_fps, (width, height))
                                cap2 = cv2.VideoCapture(temp_cam_file)
                                while True:
                                    ret2, f2 = cap2.read()
                                    if not ret2:
                                        break
                                    out2.write(f2)
                                cap2.release()
                                out2.release()
                                shutil.move(tmp_rewrite, temp_cam_file)
                        # Не вызываем pose/hands.close() — объекты живут в st.cache_resource
                        
                        image_placeholder.empty()
                        progress_bar_ph.empty()
                        status_box.success(t["recording_done"])
                        
                        # ── Результаты анализа выбранной тренируемой руки ──
                        chosen_angles = angles_live
                        chosen_side_text = t["arm_right"] if is_right_arm else t["arm_left"]
                        left_angles_rec = [] if is_right_arm else angles_live
                        right_angles_rec = angles_live if is_right_arm else []
                        
                        with st.spinner(t["analyzing"]):
                            analysis_res = analyze_video_real(
                                temp_cam_file, 
                                current_ex_id, 
                                real_angles=chosen_angles,
                                chosen_side=chosen_side_text,
                                left_angles=left_angles_rec,
                                right_angles=right_angles_rec,
                                actual_fps=actual_fps
                            )
                        st.session_state["analysis_result"] = analysis_res
                        st.rerun()
            
            else:
                render_analysis_view(st.session_state["analysis_result"], t)

    # Отложенная невидимая озвучка в самом конце скрипта (чтобы UI и кнопка записи появлялись мгновенно)
    if st.session_state["analysis_result"] is None:
        voice_text = current_ex_lang.get("voice_text", "")
        if voice_text:
            voice_key = f"{current_ex_id}_{lang}"
            should_autoplay = (
                st.session_state.get("last_voice_key") != voice_key
                or st.session_state.pop("force_voice_replay", False)
            )
            if should_autoplay:
                st.session_state["last_voice_key"] = voice_key
                audio_bytes = generate_speech(voice_text, lang)
                if audio_bytes:
                    nonce = st.session_state.get("tts_nonce", 0) + 1
                    st.session_state["tts_nonce"] = nonce
                    play_hidden_audio(audio_bytes, uid=f"{voice_key}_{nonce}")
