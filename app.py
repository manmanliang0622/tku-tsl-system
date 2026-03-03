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

# 1. 環境與頁面基礎設定
load_dotenv()
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 自動偵測 GPU (解決雲端無 GPU 會崩潰的問題)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 共享狀態管理 (Thread-safe Queue)
# 這是為了解決 WebRTC 背景處理與 Streamlit 主畫面通訊的關鍵
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if "detected_glosses" not in st.session_state:
    st.session_state.detected_glosses = []
if "translated_chinese" not in st.session_state:
    st.session_state.translated_chinese = ""

# 3. 載入模型與標籤 (使用快取避免重複載入)
@st.cache_resource
def load_resources():
    # 載入標籤對照表
    try:
        with open('label_map.json', 'r', encoding='utf-8') as f:
            l_map = json.load(f)
        actions = {v: k for k, v in l_map.items()}
    except:
        actions = {0: "測試標籤"}

    # 載入妳的 TSL 模型
    # from core.data_loader import TSLModel
    # model = TSLModel(...) 
    # model.load_state_dict(torch.load('models/tsl_model.pth', map_location=device))
    # model.eval().to(device)
    return actions # , model

actions = load_resources()

# 4. MediaPipe 初始化
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 5. 影像處理回呼函數 (WebRTC 背景執行緒)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # 鏡像處理
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # MediaPipe 偵測
    results = holistic.process(img_rgb)
    
    # 這裡實作妳的模型預測邏輯 (簡化示範)
    # 辨識成功後，將結果推入 queue 而非 session_state
    # example_action = "宿舍" 
    # st.session_state.result_queue.put(example_action)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 6. UI 介面佈局
st.title("🤟 淡江大學行政服務：台灣手語雙向轉譯系統")
st.info("本系統已整合 Gemini 2.5 Flash 進行行政用語轉譯，並支援 SMPL-X 虛擬人動作封裝。")

tab1, tab2 = st.tabs(["👋 手語辨識 (SLR)", "🤖 動作生成 (SLP)"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📸 鏡頭即時偵測")
        webrtc_streamer(
            key="tsl-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_right:
        st.subheader("📝 辨識結果與翻譯")
        
        # 從 Queue 中提取背景辨識到的資料
        while not st.session_state.result_queue.empty():
            val = st.session_state.result_queue.get()
            if not st.session_state.detected_glosses or val != st.session_state.detected_glosses[-1]:
                st.session_state.detected_glosses.append(val)
                st.rerun() # 發現新詞彙時更新 UI

        st.write("**● 偵測到的手語序列 (Glosses)**")
        st.success(" ➔ ".join(st.session_state.detected_glosses) if st.session_state.detected_glosses else "等待偵測中...")

        if st.button("🪄 進行行政用語轉譯"):
            if st.session_state.detected_glosses:
                with st.spinner("Gemini 正在轉譯中..."):
                    # 串接妳的翻譯函數
                    # st.session_state.translated_chinese = translate_tsl_to_formal_chinese(st.session_state.detected_glosses)
                    st.session_state.translated_chinese = "您好，請問需要辦理宿舍申請嗎？" # 範例內容
            else:
                st.warning("目前無偵測數據")

        st.write("**● 翻譯結果 (正式中文)**")
        st.info(st.session_state.translated_chinese if st.session_state.translated_chinese else "尚未轉譯")

with tab2:
    st.subheader("🤖 虛擬人生成預覽 (SMPL-X Standard)")
    input_text = st.text_input("輸入回覆內容：", placeholder="例如：補辦學生證需至行政大樓。")
    if st.button("生成動作 JSON"):
        st.json({"status": "success", "format": "SMPL-X", "data": "joints_rotation_data..."})
