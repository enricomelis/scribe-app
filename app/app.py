import streamlit as st
from transcriber import transcribe_audio 
from summarizer import summarize_text
import tempfile

st.title('Audio Transcription and Summarization App')

uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        transcribed_text = transcribe_audio(tmp_file.name)
        summary = summarize_text(transcribed_text)
        st.subheader('Transcription')
        st.write(transcribed_text)
        
        st.subheader('Summary')
        st.write(summary[0]['summary_text'])