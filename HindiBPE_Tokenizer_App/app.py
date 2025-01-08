import streamlit as st
from src.hindi_bpe_scratch import HindiBPE
import pyperclip 

# Page config
st.set_page_config(
    page_title="Hindi BPE Tokenizer",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        color: #1E3A8A;
        font-size: 2.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        margin-bottom: 2rem;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #374151;
        font-size: 1.5rem !important;
        margin-bottom: 1rem;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 0.5rem;
        border: 1px solid #D1D5DB;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: #F9FAFB;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: none;
        margin-top: 1rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Output container styling */
    .output-container {
        background-color: #F3F4F6;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-weight: 600;
    }
    
    /* Toast styling */
    .stToast {
        background-color: #059669;
        color: white;
        border-radius: 0.5rem;
    }
    
    /* Error message styling */
    .stError {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #FCA5A5;
    }
    </style>
""", unsafe_allow_html=True)

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    tokenizer = HindiBPE(vocab_size=5000, min_freq=2)
    try:
        tokenizer.load("results/hindi_bpe_scratch.pkl")
    except FileNotFoundError:
        st.error("Tokenizer model file not found. Please ensure 'results/hindi_bpe_scratch.pkl' exists.")
    return tokenizer

tokenizer = load_tokenizer()

# Header section
st.title("🇮🇳 Hindi BPE Tokenizer")
st.markdown("""
    <div style='background-color: #F0F9FF; padding: 1rem; border-radius: 0.5rem; border: 1px solid #BAE6FD; margin-bottom: 2rem;'>
        <h4 style='color: #0369A1; margin-bottom: 0.5rem;'>About this Tool</h4>
        <p style='color: #0C4A6E; margin-bottom: 0;'>
            A professional Byte-Pair Encoding (BPE) tokenizer specifically designed for Hindi text processing.
            This tool helps in encoding Hindi text into tokens and decoding tokens back into text with high accuracy.
        </p>
    </div>
""", unsafe_allow_html=True)

# Create two columns
col1, col2 = st.columns(2)

def copy_to_clipboard():
    try:
       if st.session_state.get('to_copy'):
            pyperclip.copy(st.session_state.to_copy)
            st.toast("✅ Copied to clipboard!")
    except Exception as e:
        st.error(f"Failed to copy: {str(e)}")

with col1:
    st.markdown("### Encode Hindi Text")
    input_text = st.text_area(
        "Enter Hindi text:",
        height=150,
        placeholder="यहाँ हिंदी टेक्स्ट दर्ज करें...",
        help="Enter the Hindi text you want to tokenize"
    )
    
    if st.button("Encode ⚡", key="encode_button"):
        if input_text:
            encoded = tokenizer.encode(input_text)
            st.markdown("##### Encoded Token IDs:")
            st.code(str(encoded), language="python")
            st.session_state.to_copy = str(encoded)
            st.button("📋 Copy Tokens", key="copy_encoded", on_click=copy_to_clipboard)

with col2:
    st.markdown("### Decode Token IDs")
    input_ids = st.text_area(
        "Enter token IDs (comma-separated):",
        height=150,
        placeholder="Example: 1, 2, 3, 4, 5",
        help="Enter the token IDs separated by commas"
    )
    
    if st.button("Decode 🔄", key="decode_button"):
        try:
            cleaned_input = input_ids.replace('[', '').replace(']', '').strip()
            if not cleaned_input:
                st.error("⚠️ Please enter some token IDs")
                st.stop()
                
            ids = [int(x.strip()) for x in cleaned_input.split(',') if x.strip()]
            
            if not ids:
                st.error("⚠️ No valid token IDs found")
                st.stop()
                
            decoded = tokenizer.decode(ids)
            st.markdown("##### Decoded Text:")
            st.code(decoded, language="hindi")
            st.session_state.to_copy = decoded
            st.button("📋 Copy Text", key="copy_decoded", on_click=copy_to_clipboard)
                    
        except ValueError as e:
            st.error("⚠️ Please enter valid token IDs (comma-separated numbers)")

# Information section with tabs
st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📖 About", "🛠️ Features", "💡 Tips"])

with tab1:
    st.markdown("""
        ### About the Tokenizer
        This Hindi BPE Tokenizer is designed to efficiently process Hindi text while maintaining linguistic accuracy.
        It uses advanced tokenization techniques specifically optimized for Hindi language characteristics.
    """)

with tab2:
    st.markdown("""
        ### Key Features
        - 5,000 token vocabulary size
        - Specialized Hindi character handling
        - Support for common prefixes and suffixes
        - Special token support ([UNK], [PAD], [BOS], [EOS])
        - Efficient encoding and decoding
    """)

with tab3:
    st.markdown("""
        ### Usage Tips
        1. For best results, input clean Hindi text without mixed scripts
        2. Use copy buttons to easily transfer results
        3. Check encoded tokens for debugging
        4. Verify decoded text matches original input
    """)
