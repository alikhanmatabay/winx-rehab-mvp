import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(
    page_title="SerIQ — Домашняя реабилитация",
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
            "title": "Раскрытие кисти"
        },
        "KZ": {
            "title": "Саусақтарды ашу"
        },
        "EN": {
            "title": "Opening the hand"
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
            "title": "Подъём запястья"
        },
        "KZ": {
            "title": "Білекті көтеру"
        },
        "EN": {
            "title": "Lifting the wrist"
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
            "title": "Разгибание локтя"
        },
        "KZ": {
            "title": "Шынтақты жазу"
        },
        "EN": {
            "title": "Extending the elbow"
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
            "title": "Подъём предмета"
        },
        "KZ": {
            "title": "Затты көтеру"
        },
        "EN": {
            "title": "Lifting an object"
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
    
    /* 3. Выравнивание 3 кнопок выбора языка строго в одну линию с ярким белым активным текстом */
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
    
    .lang-switch-wrapper div[data-testid="stButton"] button {
        width: 100% !important;
        min-width: 72px !important;
        min-height: 52px !important;
        max-height: 52px !important;
        height: 52px !important;
        margin: 0 !important;
        padding: 6px 4px !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-align: center !important;
        border-radius: 12px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transform: none !important;
        transition: background-color 0.15s ease, border-color 0.15s ease !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button:hover,
    .lang-switch-wrapper div[data-testid="stButton"] button:focus,
    .lang-switch-wrapper div[data-testid="stButton"] button:active {
        transform: none !important;
        font-size: 20px !important;
    }
    
    .lang-switch-wrapper div[data-testid="stButton"] button p,
    .lang-switch-wrapper div[data-testid="stButton"] button span {
        font-size: 20px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Активная кнопка: яркий белый текст на бирюзовом фоне */
    .lang-switch-wrapper div[data-testid="stButton"] button[kind="primary"] {
        background-color: #0E7490 !important;
        color: #FFFFFF !important;
        border: none !important;
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
    
    /* 4. Карточки Экрана 1 (min-height: 360px, круг 120x120px, название 26px) */
    .exercise-card-link {
        text-decoration: none !important;
        display: block !important;
        height: 100% !important;
        outline: none !important;
    }
    
    .exercise-card {
        min-height: 440px !important;
        height: 440px !important;
        border-radius: 24px !important;
        padding: 36px 24px !important;
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
    .sb-caption {
        font-size: 15px !important;
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
        "app_title": "SerIQ — Домашняя реабилитация",
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
        "repeat_reminder": "💡 Выполните 3 плавных повторения без пауз"
    },
    "KZ": {
        "app_title": "SerIQ — Үйде оңалту көмекшісі",
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
        "angle_hud": "Бұрыш",
        "side_right": "Оң қол",
        "side_left": "Сол қол",
        "analyzed_arm_label": "Талданған қол (макс. амплитуда)",
        "full_amplitude_tip": "💡 Есте сақтаңыз: қимылды баяу, толық амплитудада және орта жолда кідірместен орындаңыз.",
        "repeat_reminder": "💡 3 рет баяу, кідіріссіз орындаңыз"
    },
    "EN": {
        "app_title": "SerIQ — Home Rehabilitation",
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
        "repeat_reminder": "💡 Perform 3 smooth repetitions without pauses"
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
    max_num_hands=2 позволяет отслеживать обе руки одновременно.
    """
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# Предварительный прогрев моделей в памяти при старте
try:
    _ = get_pose_estimator()
    _ = get_hands_estimator()
except Exception:
    pass

# ==============================================================================
# ТОЧКА ИНТЕГРАЦИИ ML-МОДУЛЯ: analyze_video(video_path, exercise_id)
# ==============================================================================
def mock_analyze_video(video_path, exercise_id="E03", real_angles=None, chosen_side=None, left_angles=None, right_angles=None):
    time.sleep(1.2)
    angles = real_angles if (real_angles and len(real_angles) >= 5) else [45, 70, 110, 155, 172, 140, 90, 48, 65, 100, 128, 105, 60, 46, 80, 135, 170, 120, 55]
    max_ang = max(angles) if angles else 168
    min_ang = min(angles) if angles else 45
    amplitude = max_ang - min_ang
    
    # Оценка правильности на основе достаточного разгибания/амплитуды
    is_complete_c1 = (max_ang >= 140 or amplitude >= 55)
    
    return {
        "cycles": [
            {"cycle": 1, "status": "complete" if is_complete_c1 else "incomplete", "confidence": 0.94},
            {"cycle": 2, "status": "incomplete", "confidence": 0.58},
            {"cycle": 3, "status": "complete", "confidence": 0.89},
        ],
        "angles_timeline": angles,
        "chosen_side": chosen_side,
        "left_timeline": left_angles or [],
        "right_timeline": right_angles or []
    }

def render_analysis_view(result, t):
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
    st.markdown(f"""
    <div style="font-size: 32px; font-weight: 800; color: #0F172A; text-align: center; margin: 10px 0 20px 0;">
        {t['screen1_title']}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    cols = [col1, col2, col3, col4]
    exercise_keys = ["E01", "E02", "E03", "E04"]
    
    for idx, ex_id in enumerate(exercise_keys):
        ex_info = EXERCISES[ex_id]
        ex_title = ex_info[lang]["title"]
        card_class = f"card-{ex_id.lower()}"
        
        with cols[idx]:
            card_html = f"""
            <a href="?ex={ex_id}&lang={lang}" target="_self" class="exercise-card-link">
                <div class="exercise-card {card_class}">
                    <div class="exercise-card-badge">{ex_info['code']}</div>
                    <div class="exercise-card-circle">
                        {ex_info['svg']}
                    </div>
                    <div class="exercise-card-title">{ex_title}</div>
                </div>
            </a>
            """
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
                status_box = st.empty()
                progress_bar_ph = st.empty()
                
                st.markdown(f"""
                <div class="patient-instruction-card">
                    <span class="patient-instruction-icon">💡</span>
                    <span class="patient-instruction-text">{t["patient_hint"]}</span>
                </div>
                """, unsafe_allow_html=True)
                
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
                                analysis_res = mock_analyze_video(temp_upload_path, current_ex_id)
                            st.session_state["analysis_result"] = analysis_res
                            st.rerun()
                
                if start_camera:
                    st.toast(t["camera_starting"], icon="📹")
                    startup_notice = status_box.info(f"📹 **{t['camera_starting']}**")
                    
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    if not cap.isOpened():
                        status_box.error(t["camera_error"])
                    else:
                        status_box.empty()
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
                        fps = 20.0
                        
                        temp_cam_file = "temp_camera.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(temp_cam_file, fourcc, fps, (width, height))
                        
                        progress_bar = progress_bar_ph.progress(0.0)
                        
                        mp_pose = mp.solutions.pose
                        mp_hands = mp.solutions.hands
                        
                        # Выбираем модель по упражнению
                        use_hands = current_ex_id in ("E01", "E02")
                        if use_hands:
                            estimator = get_hands_estimator()
                        else:
                            estimator = get_pose_estimator()
                        
                        start_time = time.time()
                        RECORD_DURATION = 10.0
                        left_angles_live = []
                        right_angles_live = []
                        
                        while True:
                            elapsed = time.time() - start_time
                            if elapsed >= RECORD_DURATION:
                                break
                            
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            mirrored = cv2.flip(frame, 1)
                            h, w, _ = mirrored.shape
                            
                            rgb_for_mp = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                            results = estimator.process(rgb_for_mp)
                            
                            curr_l_angle = None
                            curr_r_angle = None
                            
                            # ── E01 / E02: MediaPipe Hands (отслеживание и отрисовка ОБЕИХ кистей) ──
                            if use_hands and results.multi_hand_landmarks:
                                for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                                    lm = hand_landmarks.landmark
                                    wrist_x = lm[0].x
                                    # На зеркальном экране: левая половина x < 0.5, правая x >= 0.5
                                    is_right = (wrist_x >= 0.5)
                                    side_color = (0, 220, 100) if is_right else (255, 180, 0)
                                    side_tag = "R" if is_right else "L"
                                    
                                    # Отрисовка скелета кисти
                                    for conn in mp_hands.HAND_CONNECTIONS:
                                        p1, p2 = lm[conn[0]], lm[conn[1]]
                                        x1, y1 = int(p1.x * w), int(p1.y * h)
                                        x2, y2 = int(p2.x * w), int(p2.y * h)
                                        cv2.line(mirrored, (x1, y1), (x2, y2), side_color, 3)
                                        
                                    for idx_lm, landmark in enumerate(lm):
                                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                                        r = 7 if idx_lm in (0, 5, 9, 13, 17) else 4
                                        cv2.circle(mirrored, (cx, cy), r, (0, 165, 255), -1)
                                    
                                    if current_ex_id == "E01":
                                        # E01: средний угол раскрытия по 4 пальцам
                                        finger_configs = [
                                            (5, 6, 8),   # Указательный: MCP-PIP-TIP
                                            (9, 10, 12), # Средний
                                            (13, 14, 16),# Безымянный
                                            (17, 18, 20) # Мизинец
                                        ]
                                        angles_list = []
                                        for mcp_i, pip_i, tip_i in finger_configs:
                                            mcp = (int(lm[mcp_i].x * w), int(lm[mcp_i].y * h))
                                            pip = (int(lm[pip_i].x * w), int(lm[pip_i].y * h))
                                            tip = (int(lm[tip_i].x * w), int(lm[tip_i].y * h))
                                            angles_list.append(calculate_angle(mcp, pip, tip))
                                        if angles_list:
                                            h_ang = int(sum(angles_list) / len(angles_list))
                                            mid_tip = (int(lm[12].x * w), int(lm[12].y * h))
                                            cv2.putText(mirrored, f"{side_tag}: {h_ang} deg",
                                                (mid_tip[0] - 25, max(30, mid_tip[1] - 12)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                                            if is_right:
                                                curr_r_angle = h_ang
                                                right_angles_live.append(h_ang)
                                            else:
                                                curr_l_angle = h_ang
                                                left_angles_live.append(h_ang)
                                    
                                    elif current_ex_id == "E02":
                                        # E02: угол подъёма запястья (запястье 0 -> MCP 9 -> PIP 10)
                                        wrist = (int(lm[0].x * w), int(lm[0].y * h))
                                        mcp9  = (int(lm[9].x * w), int(lm[9].y * h))
                                        pip10 = (int(lm[10].x * w), int(lm[10].y * h))
                                        h_ang = calculate_angle(wrist, mcp9, pip10)
                                        cv2.putText(mirrored, f"{side_tag}: {h_ang} deg",
                                            (mcp9[0] - 25, max(30, mcp9[1] - 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                                        if is_right:
                                            curr_r_angle = h_ang
                                            right_angles_live.append(h_ang)
                                        else:
                                            curr_l_angle = h_ang
                                            left_angles_live.append(h_ang)
                            
                            # ── E03 / E04: MediaPipe Pose (отслеживание и отрисовка ОБЕИХ рук: 11-13-15 и 12-14-16) ──
                            elif not use_hands and results.pose_landmarks:
                                lm = results.pose_landmarks.landmark
                                
                                # Правая рука: плечо(12) - локоть(14) - запястье(16)
                                r_s = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                                r_e = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                                r_w = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                                r_vis = (r_s.visibility + r_e.visibility + r_w.visibility) / 3.0
                                
                                # Левая рука: плечо(11) - локоть(13) - запястье(15)
                                l_s = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                                l_e = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                                l_w = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                                l_vis = (l_s.visibility + l_e.visibility + l_w.visibility) / 3.0
                                
                                if r_vis > 0.35:
                                    s_r_px = (int(r_s.x * w), int(r_s.y * h))
                                    e_r_px = (int(r_e.x * w), int(r_e.y * h))
                                    w_r_px = (int(r_w.x * w), int(r_w.y * h))
                                    curr_r_angle = calculate_angle(s_r_px, e_r_px, w_r_px)
                                    right_angles_live.append(curr_r_angle)
                                    
                                    cv2.line(mirrored, s_r_px, e_r_px, (0, 255, 0), 4)
                                    cv2.line(mirrored, e_r_px, w_r_px, (0, 255, 0), 4)
                                    cv2.circle(mirrored, s_r_px, 8, (0, 165, 255), -1)
                                    cv2.circle(mirrored, e_r_px, 11, (0, 0, 255), -1)
                                    cv2.circle(mirrored, w_r_px, 8, (0, 165, 255), -1)
                                    cv2.putText(mirrored, f"R: {curr_r_angle} deg",
                                        (e_r_px[0] + 15, max(35, e_r_px[1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                                
                                if l_vis > 0.35:
                                    s_l_px = (int(l_s.x * w), int(l_s.y * h))
                                    e_l_px = (int(l_e.x * w), int(l_e.y * h))
                                    w_l_px = (int(l_w.x * w), int(l_w.y * h))
                                    curr_l_angle = calculate_angle(s_l_px, e_l_px, w_l_px)
                                    left_angles_live.append(curr_l_angle)
                                    
                                    cv2.line(mirrored, s_l_px, e_l_px, (255, 190, 0), 4)
                                    cv2.line(mirrored, e_l_px, w_l_px, (255, 190, 0), 4)
                                    cv2.circle(mirrored, s_l_px, 8, (0, 165, 255), -1)
                                    cv2.circle(mirrored, e_l_px, 11, (0, 0, 255), -1)
                                    cv2.circle(mirrored, w_l_px, 8, (0, 165, 255), -1)
                                    cv2.putText(mirrored, f"L: {curr_l_angle} deg",
                                        (max(10, e_l_px[0] - 120), max(35, e_l_px[1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
                            
                            # HUD-блок с отображением обеих рук
                            hud_parts = []
                            if curr_l_angle is not None:
                                hud_parts.append(f"L: {curr_l_angle} deg")
                            if curr_r_angle is not None:
                                hud_parts.append(f"R: {curr_r_angle} deg")
                            hud_angle_str = " | ".join(hud_parts) if hud_parts else "-- deg"
                            
                            cv2.rectangle(mirrored, (15, 15), (420, 72), (25, 25, 25), -1)
                            cv2.putText(
                                mirrored,
                                f"{t['angle_hud']}: {hud_angle_str}",
                                (25, 55),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (50, 255, 50),
                                2,
                                cv2.LINE_AA
                            )
                            
                            out.write(mirrored)
                            
                            rgb_display = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                            image_placeholder.image(rgb_display, channels="RGB", output_format="JPEG", use_container_width=True)
                            
                            remaining = max(0, int(RECORD_DURATION - elapsed))
                            status_box.markdown(
                                f"<div class='recording-box'>{t['recording_live'].format(sec=remaining)}</div>",
                                unsafe_allow_html=True
                            )
                            progress_bar_ph.progress(min(1.0, elapsed / RECORD_DURATION))
                        
                        cap.release()
                        out.release()
                        # Не вызываем pose.close() — объект живёт в st.cache_resource
                        
                        image_placeholder.empty()
                        progress_bar_ph.empty()
                        status_box.success(t["recording_done"])
                        
                        # ── Автоматический выбор руки с наибольшей амплитудой движения (размахом угла) ──
                        def calc_range(arr):
                            return (max(arr) - min(arr)) if len(arr) >= 4 else 0
                        
                        r_amp = calc_range(right_angles_live)
                        l_amp = calc_range(left_angles_live)
                        
                        if r_amp >= l_amp and len(right_angles_live) >= 5:
                            chosen_angles = right_angles_live
                            chosen_side_text = t["side_right"]
                        elif len(left_angles_live) >= 5:
                            chosen_angles = left_angles_live
                            chosen_side_text = t["side_left"]
                        elif len(right_angles_live) >= 5:
                            chosen_angles = right_angles_live
                            chosen_side_text = t["side_right"]
                        else:
                            chosen_angles = right_angles_live if len(right_angles_live) >= len(left_angles_live) else left_angles_live
                            chosen_side_text = t["side_right"] if len(right_angles_live) >= len(left_angles_live) else t["side_left"]
                        
                        with st.spinner(t["analyzing"]):
                            analysis_res = mock_analyze_video(
                                temp_cam_file, 
                                current_ex_id, 
                                real_angles=chosen_angles,
                                chosen_side=chosen_side_text,
                                left_angles=left_angles_live,
                                right_angles=right_angles_live
                            )
                        st.session_state["analysis_result"] = analysis_res
                        st.rerun()
            
            else:
                render_analysis_view(st.session_state["analysis_result"], t)
