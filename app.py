import streamlit as st
import cv2
import numpy as np
import av
import torch
import json
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dotenv import load_dotenv

# --- 1. 環境設定 ---
load_dotenv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 雲端會自動跳 CPU

# 用於在背景影像處理與主畫面 UI 之間傳遞資料
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()

# --- 2. 載入模型邏輯 (保持妳原本的架構，但加入錯誤處理) ---
@st.cache_resource
def load_trained_model():
    # 這裡請確保妳的 core 資料夾有正確上傳
    from core.data_loader import TSLModel 
    with open('label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    actions = {v: k for k, v in label_map.items()}
    
    model = TSLModel(input_size=126, hidden_size=128, num_layers=2, num_classes=len(actions))
    # 注意：如果 models 資料夾沒上傳會報錯
    if os.path.exists('models/tsl_model.pth'):
        model.load_state_dict(torch.load('models/tsl_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, actions

try:
    tsl_ai_model, actions = load_trained_model()
except Exception as e:
    st.error(f"模型載入失敗: {e}")

# --- 3. WebRTC 影像處理 (修正後) ---
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 建立一個區域變數存序列，不要用 session_state
frame_sequence = []

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    
    # 提取特徵 (簡化版)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    keypoints = np.concatenate([lh, rh])

    frame_sequence.append(keypoints)
    if len(frame_sequence) > 30:
        frame_sequence.pop(0)

    if len(frame_sequence) == 30:
        input_tensor = torch.FloatTensor(np.expand_dims(frame_sequence, axis=0)).to(device)
        with torch.no_grad():
            res = tsl_ai_model(input_tensor)
            max_prob, idx = torch.max(torch.softmax(res, dim=1), dim=1)
            
            if max_prob.item() > 0.85:
                # 關鍵：將結果放入 Queue 傳回主畫面
                st.session_state.result_queue.put(actions[idx.item()])

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. 主 UI 介面 ---
st.title("🤟 淡江大學 TSL 手語轉譯系統")

# 定期檢查 Queue 並更新 UI
while not st.session_state.result_queue.empty():
    new_gloss = st.session_state.result_queue.get()
    if not st.session_state.get('detected_glosses') or new_gloss != st.session_state.detected_glosses[-1]:
        st.session_state.setdefault('detected_glosses', []).append(new_gloss)

# 這裡放置妳原本的展示邏輯...
st.write("偵測到的詞彙：", " ➔ ".join(st.session_state.get('detected_glosses', [])))
