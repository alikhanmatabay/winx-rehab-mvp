import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
import cv2
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="SerIQ — Домашняя реабилитация", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px !important; }
    .stButton > button {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 16px 24px !important;
        border-radius: 14px !important;
    }
    .big-card { padding: 20px; border-radius: 12px; margin-bottom: 12px; font-size: 20px; font-weight: 600; }
    .ok-card { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
    .warn-card { background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba; }
    .countdown-box {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        padding: 24px;
        background-color: #e3f2fd;
        border: 3px solid #90caf9;
        border-radius: 16px;
        color: #0d47a1;
        margin-bottom: 15px;
    }
    .recording-box {
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        padding: 14px;
        background-color: #ffebee;
        border: 2px solid #ffcdd2;
        border-radius: 12px;
        color: #b71c1c;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

TEXTS = {
    "RU": {
        "title": "SerIQ — Помощник для домашней реабилитации",
        "subtitle": "Упражнение E02: Разгибание локтя",
        "disclaimer": "ℹ️ Инструмент носит информационный характер, не ставит диагноз и не заменяет консультацию врача.",
        "step1": "Шаг 1. Посмотрите видеоинструкцию",
        "step2": "Шаг 2. Выполните упражнение на камеру",
        "camera_start_btn": "📹 Начать запись с камеры (10 сек)",
        "get_ready": "⏳ Приготовьтесь: {sec}...",
        "recording_live": "🔴 Идет запись с камеры... Осталось: {sec} сек",
        "recording_done": "✅ Запись завершена! Анализируем движение...",
        "camera_error": "⚠️ Не удалось открыть веб-камеру. Пожалуйста, проверьте подключение камеры или права доступа в системе.",
        "expander_upload": "📁 Загрузить готовое видео (для теста)",
        "upload_label": "Выберите видеофайл (.mp4, .avi, .mov)",
        "analyze_btn": "Проверить загруженное видео",
        "analyzing": "Анализируем движения руки...",
        "summary_title": "Сводка занятия",
        "total_cycles": "Распознано циклов",
        "completed_cycles": "Выполнено",
        "review_cycles": "Проверьте инструкцию",
        "status_ok": "✅ Выполнено",
        "status_warn": "⚠️ Проверьте инструкцию",
        "low_conf_note": "Низкая уверенность модели или неполный цикл — пожалуйста, посмотрите видеоинструкцию еще раз.",
        "chart_title": "Динамика угла разгибания локтя (в градусах)",
        "cycle_col": "Повторение",
        "result_col": "Результат",
        "conf_col": "Уверенность ИИ",
        "camera_hint": "💡 Нажмите кнопку выше, отойдите на 1-2 метра, чтобы рука полностью помещалась в кадр, и начните упражнение.",
        "angle_hud": "Угол локтя"
    },
    "KZ": {
        "title": "SerIQ — Үйде оңалтуға арналған көмекші",
        "subtitle": "E02 жаттығуы: Шынтақты жазу",
        "disclaimer": "ℹ️ Бұл құрал тек ақпараттық сипатқа ие, диагноз қоймайды және дәрігердің кеңесін алмастырмайды.",
        "step1": "1-қадам. Бейненұсқаулықты қараңыз",
        "step2": "2-қадам. Жаттығуды камера алдында орындаңыз",
        "camera_start_btn": "📹 Камерадан жазуды бастау (10 сек)",
        "get_ready": "⏳ Дайындалыңыз: {sec}...",
        "recording_live": "🔴 Камерадан жазылуда... Қалды: {sec} сек",
        "recording_done": "✅ Жазу аяқталды! Қимыл талдануда...",
        "camera_error": "⚠️ Веб-камераны ашу мүмкін болмады. Камера қосылымын немесе жүйелік рұқсаттарды тексеріңіз.",
        "expander_upload": "📁 Дайын бейнені жүктеу (тест үшін)",
        "upload_label": "Бейнефайлды таңдаңыз (.mp4, .avi, .mov)",
        "analyze_btn": "Жүктелген бейнені тексеру",
        "analyzing": "Қол қимылдары талдануда...",
        "summary_title": "Жаттығу қорытындысы",
        "total_cycles": "Анықталған циклдар",
        "completed_cycles": "Орындалды",
        "review_cycles": "Нұсқаулықты тексеріңіз",
        "status_ok": "✅ Орындалды",
        "status_warn": "⚠️ Нұсқаулықты тексеріңіз",
        "low_conf_note": "Модель сенімділігі төмен немесе цикл толық емес — бейненұсқаулықты қайта қарап шығыңыз.",
        "chart_title": "Шынтақтың жазылу бұрышының өзгеруі (градуспен)",
        "cycle_col": "Қайталау",
        "result_col": "Нәтиже",
        "conf_col": "ЖИ сенімділігі",
        "camera_hint": "💡 Жоғарыдағы батырманы басып, қолыңыз кадрға толық көрінуі үшін 1-2 метрге шегініңіз және жаттығуды бастаңыз.",
        "angle_hud": "Шынтақ бұрышы"
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

def mock_analyze_video(video_path, real_angles=None):
    time.sleep(1.5)
    angles = real_angles if (real_angles and len(real_angles) >= 5) else [45, 70, 110, 155, 172, 140, 90, 48, 65, 100, 128, 105, 60, 46, 80, 135, 170, 120, 55]
    max_ang = max(angles) if angles else 168
    return {
        "cycles": [
            {"cycle": 1, "status": "complete" if max_ang >= 150 else "incomplete", "confidence": 0.94},
            {"cycle": 2, "status": "incomplete", "confidence": 0.52},
            {"cycle": 3, "status": "complete", "confidence": 0.89},
        ],
        "angles_timeline": angles
    }

def display_results(result, t):
    st.divider()
    st.header(t["summary_title"])
    
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
    
    m1, m2, m3 = st.columns(3)
    m1.metric(t["total_cycles"], total_count)
    m2.metric(t["completed_cycles"], ok_count)
    m3.metric(t["review_cycles"], warn_count)
    
    st.dataframe(pd.DataFrame(processed_cycles), use_container_width=True, hide_index=True)
    
    if warn_count > 0:
        st.warning(t["low_conf_note"])
    
    fig = px.line(
        y=result["angles_timeline"],
        labels={"x": "Кадр / Кадр", "y": "Угол / Бұрыш (°)"},
        title=t["chart_title"]
    )
    st.plotly_chart(fig, use_container_width=True)

col_head, col_lang = st.columns([4, 1])
with col_lang:
    lang = st.radio("Тіл / Язык", ["RU", "KZ"], horizontal=True)

t = TEXTS[lang]

with col_head:
    st.title(t["title"])
    st.subheader(t["subtitle"])

st.info(t["disclaimer"])
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.header(t["step1"])
    if os.path.exists("reference_e02.mp4"):
        st.video("reference_e02.mp4")
    else:
        st.warning("Положите файл reference_e02.mp4 в папку проекта для отображения инструкции.")

with col2:
    st.header(t["step2"])
    st.caption(t["camera_hint"])
    
    start_camera = st.button(t["camera_start_btn"], type="primary", use_container_width=True)
    
    if start_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error(t["camera_error"])
        else:
            # 1. Отсчет 3 секунды ("Приготовьтесь...")
            countdown_box = st.empty()
            for sec in [3, 2, 1]:
                countdown_box.markdown(
                    f"<div class='countdown-box'>{t['get_ready'].format(sec=sec)}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(1.0)
            countdown_box.empty()
            
            # 2. Настройка параметров записи (temp_camera.mp4, кодек mp4v, 20 fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            fps = 20.0
            
            temp_cam_file = "temp_camera.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_cam_file, fourcc, fps, (width, height))
            
            status_box = st.empty()
            image_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            
            # Инициализация MediaPipe Pose (легкая модель complexity=0 для максимальной скорости)
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0
            )
            
            start_time = time.time()
            RECORD_DURATION = 10.0
            live_angles = []
            
            # 3. Считывание кадров, поиск ключевых точек MediaPipe и отрисовка линий
            while True:
                elapsed = time.time() - start_time
                if elapsed >= RECORD_DURATION:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Зеркальное отображение
                mirrored = cv2.flip(frame, 1)
                h, w, _ = mirrored.shape
                
                # Обработка через MediaPipe Pose
                rgb_for_mp = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_for_mp)
                
                current_angle = None
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    
                    # Проверяем видимость правой и левой руки
                    r_s = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    r_e = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                    r_w = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    r_vis = (r_s.visibility + r_e.visibility + r_w.visibility) / 3.0
                    
                    l_s = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    l_e = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_w = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_vis = (l_s.visibility + l_e.visibility + l_w.visibility) / 3.0
                    
                    # Выбираем наиболее видимую руку
                    if r_vis > 0.4 and r_vis >= l_vis:
                        s_pt, e_pt, w_pt = r_s, r_e, r_w
                    elif l_vis > 0.4:
                        s_pt, e_pt, w_pt = l_s, l_e, l_w
                    else:
                        s_pt, e_pt, w_pt = None, None, None
                    
                    if s_pt and e_pt and w_pt:
                        s_px = (int(s_pt.x * w), int(s_pt.y * h))
                        e_px = (int(e_pt.x * w), int(e_pt.y * h))
                        w_px = (int(w_pt.x * w), int(w_pt.y * h))
                        
                        # Вычисление угла локтя
                        current_angle = calculate_angle(s_px, e_px, w_px)
                        live_angles.append(current_angle)
                        
                        # Отрисовка линий скелета: плечо -> локоть -> запястье
                        cv2.line(mirrored, s_px, e_px, (0, 255, 0), 4)
                        cv2.line(mirrored, e_px, w_px, (0, 255, 0), 4)
                        
                        # Точки суставов
                        cv2.circle(mirrored, s_px, 8, (0, 165, 255), -1)
                        cv2.circle(mirrored, e_px, 10, (0, 0, 255), -1)
                        cv2.circle(mirrored, w_px, 8, (0, 165, 255), -1)
                        
                        # Текстовая метка угла рядом с локтем
                        cv2.putText(
                            mirrored,
                            f"{current_angle} deg",
                            (e_px[0] + 15, max(30, e_px[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )
                
                # Верхний информационный HUD поверх видео
                cv2.rectangle(mirrored, (15, 15), (320, 60), (30, 30, 30), -1)
                hud_angle_str = f"{current_angle} deg" if current_angle is not None else "-- deg"
                cv2.putText(
                    mirrored,
                    f"{t['angle_hud']}: {hud_angle_str}",
                    (25, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (50, 255, 50),
                    2,
                    cv2.LINE_AA
                )
                
                # Запись размеченного кадра в файл
                out.write(mirrored)
                
                # Быстрый вывод в UI через JPEG
                rgb_display = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                image_placeholder.image(rgb_display, channels="RGB", output_format="JPEG", use_container_width=True)
                
                remaining = max(0, int(RECORD_DURATION - elapsed))
                status_box.markdown(
                    f"<div class='recording-box'>{t['recording_live'].format(sec=remaining)}</div>",
                    unsafe_allow_html=True
                )
                progress_bar.progress(min(1.0, elapsed / RECORD_DURATION))
            
            # Освобождение ресурсов
            cap.release()
            out.release()
            pose.close()
            
            image_placeholder.empty()
            progress_bar.empty()
            status_box.success(t["recording_done"])
            
            # 4. Передача записанного ролика и реальных углов в анализ
            with st.spinner(t["analyzing"]):
                result = mock_analyze_video(temp_cam_file, real_angles=live_angles)
            display_results(result, t)
            
    # Загрузка готового видео для теста
    st.write("")
    with st.expander(t["expander_upload"]):
        uploaded_file = st.file_uploader(t["upload_label"], type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_upload_path = f"temp_{uploaded_file.name}"
            with open(temp_upload_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.video(temp_upload_path)
            
            if st.button(t["analyze_btn"], type="secondary", use_container_width=True):
                with st.spinner(t["analyzing"]):
                    result = mock_analyze_video(temp_upload_path)
                display_results(result, t)
