# 台語文字轉語音 (Taiwanese TTS) API

這是將 Tacotron2 + WaveGlow 模型整理成易用 API 的 Python 模組。

## 檔案說明

- `tts_inference.py` - 主要的 TTS 模組，包含所有必要的功能
- `example_usage.py` - 使用範例程式碼
- `README.md` - 本說明檔案

## 安裝需求

```bash
# 基本套件
pip install torch numpy scipy

# 確保有以下模組（應該在你的 tacotron2 專案中）
# - hparams
# - model (Tacotron2)
# - layers
# - audio_processing
# - train
# - text
# - denoiser
```

## 快速開始

### 1. 最簡單的使用方式

```python
from tts_inference import text_to_speech

# 輸入文字，輸出音檔
text = "Hit8 e7 thik8-senn7 na3 hoo3 gua1 tu2--tioh8"
audio, sampling_rate = text_to_speech(text, output_path="output.wav")
```

### 2. 進階使用（重複使用模型）

```python
from tts_inference import TaiwaneseTextToSpeech

# 初始化（只載入一次模型）
tts = TaiwaneseTextToSpeech()

# 批次處理多個文字
texts = ["Gua2 ai3 li2.", "Li2 ho2 bo5?"]
for i, text in enumerate(texts):
    audio, sr = tts.synthesize(text)
    tts.save_audio(audio, f"output_{i}.wav", sr)
```

### 3. 整合到 Flask API

```python
from flask import Flask, request, send_file
from tts_inference import text_to_speech
import io
import scipy.io.wavfile as wavfile
import numpy as np

app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def tts_api():
    # 從請求中取得文字
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return {'error': 'No text provided'}, 400
    
    # 執行 TTS
    audio, sr = text_to_speech(text)
    
    # 將音頻轉換為 WAV 格式的 bytes
    wav_io = io.BytesIO()
    wavfile.write(wav_io, sr, (audio * 32767).astype(np.int16))
    wav_io.seek(0)
    
    # 回傳音檔
    return send_file(
        wav_io,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='output.wav'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 4. 命令列使用

```bash
# 基本使用
python tts_inference.py "Gua2 ai3 li2." -o output.wav

# 自訂參數
python tts_inference.py "Gua2 ai3 li2." \
    --output custom_output.wav \
    --device cuda \
    --denoise-strength 0.02 \
    --sigma 0.7
```

## API 參考

### `text_to_speech()` 函式

簡單的便利函式，適合一次性使用。

**參數：**
- `text` (str): 要轉換的台語文字
- `output_path` (str, optional): 輸出音檔路徑
- `tacotron_checkpoint` (str, optional): Tacotron2 模型路徑
- `waveglow_path` (str, optional): WaveGlow 模型路徑
- `device` (str): 使用的裝置 ("cuda" 或 "cpu")
- `use_denoiser` (bool): 是否使用去噪
- `denoiser_strength` (float): 去噪強度 (0.0-1.0)
- `sigma` (float): WaveGlow sigma 參數

**回傳：**
- tuple: (音頻 numpy array, 採樣率)

### `TaiwaneseTextToSpeech` 類別

適合需要重複使用或批次處理的情況。

**方法：**
- `__init__()`: 初始化模型
- `synthesize(text, sigma, apply_post_processing)`: 合成語音
- `save_audio(audio, output_path, sampling_rate)`: 儲存音檔

## 參數調整建議

### sigma 參數
- **0.5-0.6**: 較清晰但可能較不自然
- **0.666** (預設): 平衡的設定
- **0.7-0.8**: 較自然但可能較模糊

### 去噪強度
- **0.0**: 不去噪
- **0.01** (預設): 輕微去噪
- **0.02-0.05**: 中度去噪
- **>0.05**: 強力去噪（可能影響音質）

## 注意事項

1. **模型路徑**: 請確認修改程式碼中的模型路徑：
   - Tacotron2: `D:/TaiwaneseTTS/tacotron2/outdir/taiwanese_num_sandhi/checkpoint_260000`
   - WaveGlow: `D:/TaiwaneseTTS/tacotron2/waveglow/waveglow_256channels_ljs_v3.pt`

2. **GPU 記憶體**: 使用 CUDA 時需要足夠的 GPU 記憶體（建議 4GB 以上）

3. **相依套件**: 確保所有必要的模組都在 Python 路徑中

4. **文字格式**: 輸入的文字應該使用正確的台語拼音格式

## 效能優化建議

1. **批次處理**: 使用 `TaiwaneseTextToSpeech` 類別避免重複載入模型
2. **GPU 加速**: 使用 CUDA 可大幅提升合成速度
3. **快取模型**: 在 API 服務中，將模型保持在記憶體中

## 疑難排解

### 問題：找不到模組
確保將 tacotron2 專案目錄加入 Python 路徑：
```python
import sys
sys.path.append('/path/to/tacotron2')
```

### 問題：CUDA 記憶體不足
- 降低批次大小
- 使用 CPU 模式：`device="cpu"`
- 清理 GPU 記憶體：`torch.cuda.empty_cache()`

### 問題：音質問題
- 調整 sigma 參數
- 調整去噪強度
- 確認輸入文字格式正確

## 聯絡與支援

如有問題或需要協助，請聯絡開發團隊。
