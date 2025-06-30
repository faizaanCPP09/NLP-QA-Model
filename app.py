import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from streamlit_lottie import st_lottie
import json

# Load Lottie animation
def load_lottie(path: str):
    with open(path, "r") as f:
        return json.load(f)

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model")  # Custom model folder
    model = AutoModelForQuestionAnswering.from_pretrained("./model")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()
lottie_globe = load_lottie("glowing_rotating_globe.json")

# Page configuration
st.set_page_config(page_title="Multilingual QA", layout="centered")

# --- UI Starts ---
st_lottie(lottie_globe, height=200, key="globe")

st.markdown(
    "<h1 style='text-align: center; color: #3399ff;'>🌐 Multilingual Question Answering</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Ask anything from a given context. Supports multiple languages depending on model capabilities.</p>",
    unsafe_allow_html=True
)

# Input form
with st.form("qa_form"):
    context = st.text_area("📘 Enter the context:", height=200)
    question = st.text_input("❓ Enter your question:")
    submit = st.form_submit_button("Get Answer")

if submit:
    if not context or not question:
        st.warning("Please enter both context and question.")
    else:
        with st.spinner("Thinking..."):
            result = qa_pipeline(question=question, context=context)
            answer = result['answer']
            st.success(f"🧠 **Answer:** {answer}")