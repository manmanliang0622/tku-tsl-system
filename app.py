import streamlit as st
import cv2
import numpy as np
import time
import av
import os
import torch
import queue
import json
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dotenv import load_dotenv

# 載入核心組件 (確保妳有這些檔案)
# from core.data_loader import TSLModel 
# from core.tsl_smart_translator import translate_tsl_to_formal_chinese

# 1. 頁面基礎設定與環境載入
load_dotenv()
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 自動偵測 GPU (解決 Streamlit Cloud 無 GPU 報錯的問題)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 載入標籤對照表
try:
    with open('label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    actions = {v: k for k, v in label_map.items()}
except FileNotFoundError:
    st.warning("找不到 label_map.json，將使用預設測試標籤。")
    actions = {0: "學生證", 1: "宿舍", 2: "成績單"}

# 3. 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 4. 初始化 Session State 與 Queue
if "detected_glosses" not in st.session_state:
    st.session_state.detected_glosses = []
if "translated_chinese" not in st.session_state:
    st.session_state.translated_chinese = ""

# 使用 queue 來解決 WebRTC 和 Streamlit 主執行緒之間的數據傳遞問題
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# 5. 影像處理回呼函數 (WebRTC 核心)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- 測試邏輯：如果偵測到手，假裝推入一個結果 ---
            # 實際使用時，這裡要換成 TSLModel 的推理結果
            # 透過 queue 傳遞，避免直接寫入 st.session_state
            if np.random.rand() > 0.95:  # 模擬偶爾偵測到詞彙
                 st.session_state.result_queue.put(actions.get(0, "手語測試詞"))

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 6. UI 介面設計
st.title("🤟 淡江大學行政服務：台灣手語雙向轉譯系統")
st.markdown("---")

tab_tsl_to_ch, tab_ch_to_tsl = st.tabs(["👋 手語轉中文 (辨識)", "🤖 中文轉手語 (生成)"])

with tab_tsl_to_ch:
    col_cam, col_res = st.columns([2, 1])
    
    with col_cam:
        st.subheader("📸 即時影像辨識")
        webrtc_streamer(
            key="tsl-translator",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
        )
        
        if st.button("🗑️ 清除辨識結果"):
            st.session_state.detected_glosses = []
            st.session_state.translated_chinese = ""
            st.rerun()

    with col_res:
        st.subheader("📝 翻譯結果")
        
        # 定期從 Queue 取出辨識結果並更新 UI
        while not st.session_state.result_queue.empty():
            new_gloss = st.session_state.result_queue.get()
            if new_gloss not in st.session_state.detected_glosses:
                st.session_state.detected_glosses.append(new_gloss)
                st.rerun() # 強制刷新畫面以顯示新詞彙

        st.write("**● 手語標籤序列 (Glosses)**")
        gloss_text = " ➔ ".join(st.session_state.detected_glosses)
        if gloss_text:
            st.info(gloss_text)
        else:
            st.markdown('<p class="empty-placeholder">請在鏡頭前比出手語...</p>', unsafe_allow_html=True)
        
        if st.button("🪄 轉譯為行政正式用語 (Gemini)"):
            if st.session_state.detected_glosses:
                with st.spinner("AI 組句中..."):
                    time.sleep(1) # 模擬 API 延遲
                    # 實際請使用：
                    # res = translate_tsl_to_formal_chinese(st.session_state.detected_glosses)
                    st.session_state.translated_chinese = f"（系統翻譯）同學您好，您剛才表達的是：{' '.join(st.session_state.detected_glosses)}。請問需要什麼協助？" 
            else:
                st.warning("請先錄入手語標籤")

        st.write("**● 淡江行政回覆 (自然中文)**")
        if st.session_state.translated_chinese:
            st.markdown(f'<div class="result-card"><h4 style="margin:0; color:#2E86C1;">{st.session_state.translated_chinese}</h4></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card"><p class="empty-placeholder">等待模型組句中...</p></div>', unsafe_allow_html=True)

with tab_ch_to_tsl:
    st.subheader("🤖 SMPL-X 虛擬人生成預覽")
    col_input, col_avatar = st.columns([2, 3])
    
    with col_input:
        st.markdown("### ⌨️ 中文文字輸入")
        user_input = st.text_area("請輸入行政回覆：", height=150, placeholder="例如：補辦學生證需要帶照片。")
        if st.button("生成動作數據"):
            if user_input:
                st.success(f"已將「{user_input}」轉換為 SMPL-X 骨架旋轉數據 (JSON)")
                st.json({"status": "Success", "model": "SMPL-X", "joints": 42, "frames": 120})
            else:
                 st.warning("請先輸入文字")
                 
    with col_avatar:
         st.info("Unity 渲染畫面預留區 / JSON 視覺化")
         # 這裡可以放預先錄製好的 Demo 影片，假裝是虛擬人在動
