import streamlit as st
import cv2
import numpy as np
import av
import os
import torch
import json
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dotenv import load_dotenv

# 假設妳的核心組件路徑正確
# from core.data_loader import TSLModel 
# from core.tsl_smart_translator import translate_tsl_to_formal_chinese

# 1. 基礎設定
load_dotenv()
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 2. 共享變數與狀態管理
if "detected_glosses" not in st.session_state:
    st.session_state.detected_glosses = []
if "translated_chinese" not in st.session_state:
    st.session_state.translated_chinese = ""

# 用於在 WebRTC 執行緒與 Streamlit 主執行緒之間傳遞數據
result_queue = queue.Queue()

# 3. 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 4. 影像處理回呼函數 (WebRTC 核心)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr22")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 繪製骨架線條
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 這裡之後可以加入妳的 PyTorch 模型推理 (TSLModel)
            # 範例邏輯：若偵測到特定手勢，推入 queue
            # result_queue.put("宿舍") 

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 5. UI 介面設計
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
        
        # 從 Queue 中提取辨識到的手語標籤 (Gloss)
        while not result_queue.empty():
            gloss = result_queue.get()
            if gloss not in st.session_state.detected_glosses:
                st.session_state.detected_glosses.append(gloss)

        st.write("**● 手語標籤序列 (Glosses)**")
        gloss_display = " ➔ ".join(st.session_state.detected_glosses) if st.session_state.detected_glosses else "等待偵測..."
        st.info(gloss_display)

        if st.button("🪄 轉譯為行政正式用語 (Gemini)"):
            if st.session_state.detected_glosses:
                with st.spinner("AI 組句中..."):
                    # 調用妳的 Gemini 翻譯邏輯
                    # res = translate_tsl_to_formal_chinese(st.session_state.detected_glosses)
                    st.session_state.translated_chinese = "（範例）我想詢問學生證補辦流程。" # 這裡代入翻譯結果
            else:
                st.warning("請先錄入手語標籤")

        st.write("**● 行政正式中文回覆**")
        if st.session_state.translated_chinese:
            st.success(st.session_state.translated_chinese)

with tab_ch_to_tsl:
    st.subheader("🤖 SMPL-X 虛擬人生成預覽")
    input_text = st.text_input("輸入行政指令：", placeholder="例如：學生證遺失")
    if st.button("生成動作數據"):
        st.write(f"正在將「{input_text}」轉換為 SMPL-X 骨架旋轉數據 (JSON)...")
        st.json({"status": "Success", "model": "SMPL-X", "joints": 42})
