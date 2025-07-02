import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForQuestionAnswering.from_pretrained("./model")
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

st.set_page_config(page_title="Multilingual QA", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #3399ff;'>🌍 Multilingual Question Answering BERT</h1>
    <p style='text-align: center; font-size: 18px;'>Ask questions in any language — just give the context. Let the model do the thinking!</p>
""", unsafe_allow_html=True)

with st.form("qa_form"):
    context = st.text_area("🟦 Enter the context:", height=200)
    question = st.text_input("❓ Enter your question:")
    submit = st.form_submit_button("Get Answer")

if submit:
    if not context or not question:
        st.warning("⚠️ Please enter both the context and question.")
    else:
        with st.spinner("🤖 Thinking..."):
            result = qa_pipeline(question=question, context=context)
            st.success(f"🧠 **Answer:** {result['answer']}")