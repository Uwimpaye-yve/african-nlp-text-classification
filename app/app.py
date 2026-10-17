import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Page configuration
st.set_page_config(
    page_title="Swahili News Topic Classifier",
    page_icon="🌍",
    layout="centered"
)

# Header & Introduction
st.title("🌍 Swahili News Topic Classifier")
st.markdown(
    """
    This web application classifies Swahili news headlines and articles into **7 categories** 
    using a fine-tuned **AfriBERTa** transformer model trained on the **MasakhaNEWS** benchmark.
    """
)

# Hugging Face Model Hub identifier
MODEL_ID = "uyvette/swahili-news-afriberta"

@st.cache_resource
def load_model():
    """Download and cache tokenizer and model directly from Hugging Face Hub."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model

# Load model with status indication
try:
    with st.spinner("Loading AfriBERTa model from Hugging Face Hub..."):
        tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sample Swahili news headlines for quick testing
example_prompts = {
    "Select an example...": "",
    "Michezo (Sports)": "Simba SC yapata ushindi mnono dhidi ya watani wao wa jadi uwanja wa Mkapa jijini Dar es Salaam.",
    "Afya (Health)": "Wizara ya Afya yatangaza mikakati mipya ya kupambana na maambukizi ya ugonjwa wa malaria na kipindupindu.",
    "Siasa (Politics)": "Bunge la Jamhuri ya Muungano wa Tanzania limepitisha muswada mpya wa sheria kuhusu taratibu za uchaguzi mkuu.",
    "Teknolojia (Technology)": "Kampuni za mawasiliano zazindua mtandao wa kasi wa 5G ili kuboresha upatikanaji wa huduma za kidijitali nchini.",
    "Biashara (Business)": "Benki Kuu yatangaza viwango vipya vya riba huku mfumuko wa bei ukishuka na kukuza biashara za ndani."
}

selected_example = st.selectbox("Choose a pre-filled Swahili example (or type below):", list(example_prompts.keys()))
default_text = example_prompts[selected_example] if selected_example != "Select an example..." else ""

# Input text box
user_input = st.text_area(
    "Enter Swahili News Headline & Text:",
    value=default_text,
    height=150,
    placeholder="Andika au bandika habari ya Kiswahili hapa..."
)

# Classification button and inference pipeline
if st.button("Classify Article", type="primary"):
    if not user_input.strip():
        st.warning("Tafadhali ingiza maandishi ya habari (Please enter news text to analyze).")
    else:
        with st.spinner("Analyzing text and computing category probabilities..."):
            # Tokenize input
            inputs = tokenizer(
                user_input,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            predicted_id = torch.argmax(probabilities).item()
            predicted_label = model.config.id2label[predicted_id]
            confidence = probabilities[predicted_id].item()

            st.divider()

            # Prediction Output Display
            st.markdown(f"### Predicted Category: **{predicted_label.upper()}**")
            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            st.progress(min(max(confidence, 0.0), 1.0))

            # Probability distribution chart across all 7 categories
            st.markdown("#### Probability Distribution Across Classes:")
            prob_dict = {
                model.config.id2label[i]: float(probabilities[i])
                for i in range(len(model.config.id2label))
            }
            st.bar_chart(prob_dict)