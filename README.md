# TKU TSL System — SignAvatar 虛擬人

手譯橋 SignBridge 的 3D 手語虛擬人系統：中文句子 → 台灣手語 gloss → VRM 虛擬人動作播放。

## 內容

| 路徑 | 說明 |
|---|---|
| `signavatar-core/` | 虛擬人核心工具鏈：動作錄製、重定向（retarget）、Blender 匯入、品質稽核（Python） |
| `avatar3d.js` | three.js + VRM 虛擬人渲染與動作驅動（骨架、表情 blendshape） |
| `composer.js` | 手語合成核心：文字 → gloss → 動作片段拼接（複合詞、數字、標記處理） |
| `signavatar-bridge.js` | 前端橋接：接管播放控制、逐詞時間軸、載入動畫 |
| `signavatar-config.js` | 端點設定（translateUrl 等） |
| `vendor/` | three.module.js、three-vrm、GLTFLoader、kalidokit |
| `models/` | VRM 虛擬人模型（avatar / orion / student） |
| `models.json` | 可用模型清單 |
| `recordings/lexicon.json` | 詞庫索引：gloss → 錄製檔與時間區段（17,000+ 詞） |
| `viewer.html` | 虛擬人檢視／合成工具頁 |
| `generate.html` | 文字轉手語播放頁 |

## 執行需求

前端頁面需搭配 SignBridge 主站（[TKU-TSL-Avatar-Project](https://github.com/manmanliang0622/TKU-TSL-Avatar-Project)）的
`bundle_server.py`：它同源提供 `/translate`（Gemma 4 QLoRA 中文→gloss）與
`/clips`（動作片段切割）。純靜態託管時退回整檔下載（`recordings/*.json`，本 repo 僅含索引）。

`signavatar-core` 的 Python 環境見其目錄內 `README.md` 與 `pyproject.toml`。
