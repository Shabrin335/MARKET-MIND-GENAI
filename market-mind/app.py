import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Set page configuration for a premium look
st.set_page_config(
    page_title="Market Mind | Financial Sentiment Analysis",
    page_icon="📈",
    layout="centered",
)

# Custom CSS for a sleek, modern design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    .stButton>button {
        width: 100%;
        background-image: linear-gradient(135deg, #2ea043 0%, #238636 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    .sentiment-card {
        padding: 24px;
        border-radius: 12px;
        background: #161b22;
        border: 1px solid #30363d;
        margin-top: 20px;
    }
    .label-positive { color: #3fb950; font-weight: bold; }
    .label-negative { color: #f85149; font-weight: bold; }
    .label-neutral { color: #8b949e; font-weight: bold; }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_pipeline():
    """Load the FinBERT model and tokenizer using the transformers pipeline."""
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=hf_token)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def main():
    st.title("Market Mind")
    st.subheader("Next-Gen Financial Sentiment Analysis")
    
    st.write("Extract powerful insights from financial news, reports, and market chatter using the state-of-the-art **FinBERT** model.")

    # Input section
    with st.container():
        user_input = st.text_area(
            "Enter financial text to analyze:",
            placeholder="e.g., Apple stock surges as quarterly earnings beat estimates despite supply chain concerns...",
            height=200
        )
        
        analyze_button = st.button("Analyze Market Sentiment")

    if analyze_button:
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing market sentiment..."):
                try:
                    sentiment_pipeline = load_sentiment_pipeline()
                    result = sentiment_pipeline(user_input)[0]
                    
                    label = result['label']
                    score = result['score']
                    
                    # Mapping labels to a more user-friendly format if needed
                    # FinBERT typically returns 'positive', 'negative', 'neutral'
                    
                    st.markdown("### Analysis Result")
                    
                    # Display result in a nice card
                    color_class = f"label-{label.lower()}"
                    st.markdown(f"""
                    <div class="sentiment-card">
                        <p style="font-size: 1.2rem; margin-bottom: 8px;">
                            Detected Sentiment: <span class="{color_class}">{label.upper()}</span>
                        </p>
                        <p style="color: #8b949e;">
                            Confidence Score: <b>{score:.2%}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization/Progress bars
                    st.progress(score)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")

    # Footer/Sidebar
    with st.sidebar:
        st.title("About Market Mind")
        st.info("""
        Market Mind uses **FinBERT**, a BERT model pre-trained on a large financial corpus and fine-tuned for financial sentiment analysis. 
        
        It is specifically designed to understand the nuances of financial language.
        """)
        st.write("---")
        st.caption("Built with Streamlit & Hugging Face")

if __name__ == "__main__":
    main()
