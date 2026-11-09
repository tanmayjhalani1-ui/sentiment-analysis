from __future__ import annotations

import pandas as pd
import streamlit as st

from src.inference import load_metadata, load_model, predict_sentiment

st.set_page_config(page_title="Sentiment Intelligence", page_icon="💬", layout="wide")
st.title("Sentiment Analysis")
st.caption("TF-IDF classification with Naive Bayes and Logistic Regression candidates.")

@st.cache_resource
def get_model():
    return load_model()

metadata = load_metadata()
input_col, report_col = st.columns([1.1, 0.9], gap="large")

with input_col:
    st.subheader("Analyze text")
    text = st.text_area("Enter a review or sentence", height=150, placeholder="The product quality is excellent...")
    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Enter some text first.")
        else:
            try:
                result = predict_sentiment(text, get_model())
                st.metric("Predicted sentiment", result["sentiment"].title(), f"{result.get('confidence', 0):.1%} model confidence")
                if result["probabilities"]:
                    st.bar_chart(pd.Series(result["probabilities"], name="Model probability"))
            except FileNotFoundError as error:
                st.warning(str(error))

with report_col:
    st.subheader("Experiment report")
    if metadata:
        st.metric("Selected model", metadata.get("selected_model", "—"))
        st.dataframe(pd.DataFrame(metadata.get("candidates", []))[[
            "model", "cv_accuracy", "cv_macro_f1", "test_accuracy", "test_macro_f1"
        ]], hide_index=True, use_container_width=True)
        st.caption(f"Training dataset rows: {metadata.get('dataset_rows', '—')}")
        st.warning(metadata.get("limitations", ""))
    else:
        st.info("Train the model before inference.")
        st.code("python -m src.train", language="bash")

st.divider()
st.subheader("Try examples")
examples = [
    "I absolutely love this! Best purchase ever!",
    "This is terrible and doesn't work at all.",
    "The product costs $29.99.",
]
for example in examples:
    if st.button(example):
        st.session_state["example_text"] = example
