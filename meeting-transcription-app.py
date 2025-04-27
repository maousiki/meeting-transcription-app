import streamlit as st
import sounddevice as sd
import scipy.io.wavfile
import tempfile
import whisper
import os

st.title("録音 + 文字起こし + 議事録作成アプリ")

# モデル選択
model_choice = st.selectbox("文字起こしモデルを選んでください", ("small", "medium"))

# 録音時間指定（分単位）
record_minutes = st.number_input("録音時間（分）を指定してください", min_value=1, max_value=60, value=1)
record_seconds = record_minutes * 60

# 録音スタート用一時変数
recording = None
recording_file_path = None

# 録音スタートボタン
if st.button("録音スタート"):
    st.info("録音中... 終了ボタンを押して録音停止してください")
    recording = sd.rec(int(record_seconds * 44100), samplerate=44100, channels=1)

# 録音終了ボタン
if st.button("録音終了・文字起こし開始"):
    if recording is not None:
        sd.stop()
        # 保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            scipy.io.wavfile.write(tmp_wav.name, 44100, recording)
            recording_file_path = tmp_wav.name
            st.success(f"録音ファイルを保存しました: {recording_file_path}")

        # モデルロードと文字起こし
        st.info("文字起こし中...")
        model = whisper.load_model(model_choice)
        result = model.transcribe(recording_file_path)
        full_text = result["text"]

        # 要点抽出（超簡単版）
        st.info("要点抽出中...")
        key_sentences = "\n".join([line.strip() for line in full_text.split("。") if len(line.strip()) > 10])

        # 結果表示
        st.subheader("議事録要約")
        st.text_area("", value=key_sentences, height=400)

        # ダウンロードボタン
        st.download_button(
            label="議事録ダウンロード",
            data=key_sentences,
            file_name="meeting_summary.txt",
            mime="text/plain"
        )
    else:
        st.warning("先に録音をスタートしてください！")
