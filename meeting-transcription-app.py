import streamlit as st
import whisper
import tempfile
from st_audiorec import st_audiorec
import scipy.io.wavfile
import numpy as np
import os

st.title("\ud83c\udfa4 録音\uff0b話者分離\uff0b文字起こし (ブラウザ録音対応)")

# Whisperモデル選択
model_choice = st.selectbox("Whisperモデルを選んでください", ("small", "medium"))

st.write("\n👉 \u9332\u97f3\u30dc\u30bf\u30f3を押\u3057てぐらいの音声を\u9332\u308aま\u3059\uff01")

# 録音ウィジェット
wav_audio_data = st_audiorec()

if isinstance(wav_audio_data, np.ndarray):
    st.success("録音完了！文字起こし開始...")

    # 一時ファイルに録音を保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
        scipy.io.wavfile.write(tmp_wav.name, 44100, wav_audio_data)
        temp_audio_path = tmp_wav.name

    # Whisperモデル読み込み
    model = whisper.load_model(model_choice)
    result = model.transcribe(temp_audio_path)

    full_text = result['text']

    # 簡易要約
    summarized_text = "\n".join([line.strip() for line in full_text.split("\u3002") if len(line.strip()) > 10])

    # 結果表示
    st.subheader("議事録総括")
    st.text_area("", value=summarized_text, height=400)

    # ダウンロードボタン
    st.download_button(
        label="議事録ダウンロード",
        data=summarized_text,
        file_name="meeting_summary.txt",
        mime="text/plain"
    )
else:
    st.info("録音まだしていません！　録音ボタンを押してください。")
