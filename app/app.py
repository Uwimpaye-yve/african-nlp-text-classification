import streamlit as st
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Swahili News Classifier", page_icon="🌍", layout="centered")

st.title("🌍 Swahili News Topic Classifier")
st.markdown(
    """
    This application classifies Swahili news articles into 7 categories using a fine-tuned 
    **AfriBERTa** model trained on the **MasakhaNEWS** benchmark.
    """
)

@st.cache_resource
def load_model():
    # Resolve absolute path to the directory containing model weights
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "best_afriberta_model"

    if not model_dir.exists():
        # Fallback if best_afriberta_model is placed in the project root instead of app/
        model_dir = base_dir.parent / "best_afriberta_model"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model folder not found at {model_dir}")

    # Pass the resolved string path
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sample Swahili headlines for testing
example_prompts = {
    "Select an example...": "",
    "Sports Example": "Simba SC yapata ushindi mnono dhidi ya watani wao wa jadi uwanja wa Mkapa jijini Dar es Salaam.",
    "Health Example": "Wizara ya Afya yatangaza mikakati mipya ya kupambana na ugonjwa wa malaria na kipindupindu.",
    "Politics Example": "Bunge la Jamhuri ya Muungano wa Tanzania limepitisha muswada mpya wa sheria za uchaguzi.",
    "Technology Example": "Kampuni za mawasiliano zazindua mtandao wa 5G ili kuboresha huduma za kidijitali nchini."
}

selected_example = st.selectbox("Or choose a pre-filled Swahili example:", list(example_prompts.keys()))
default_text = example_prompts[selected_example] if selected_example != "Select an example..." else ""

user_input = st.text_area("Enter Swahili News Headline & Text:", value=default_text, height=150)

if st.button("Classify Article", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text in Swahili to analyze.")
    else:
        with st.spinner("Classifying..."):
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            predicted_id = torch.argmax(probabilities).item()
            predicted_label = model.config.id2label[predicted_id]
            confidence = probabilities[predicted_id].item()

            st.markdown(f"### Predicted Category: **{predicted_label.upper()}**")
            st.markdown(f"**Confidence Score:** `{confidence * 100:.2f}%`")
            st.progress(confidence)

            # Show distribution across all 7 classes
            st.markdown("#### Probability Distribution:")
            prob_dict = {
                model.config.id2label[i]: float(probabilities[i]) 
                for i in range(len(model.config.id2label))
            }
            st.bar_chart(prob_dict)