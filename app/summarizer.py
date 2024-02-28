import os
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(input_text):
    summary = summarizer(input_text, do_sample=False)
    return summary