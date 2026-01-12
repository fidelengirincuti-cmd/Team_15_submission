import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import io

# Page config
st.set_page_config(
    page_title="ISIC Code Classifier",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .language-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        background-color: #10B981;
        color: white;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .feature-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F3F4F6;
        border-left: 4px solid #4F46E5;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model safely
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_text_classifier.pkl')
        metadata = joblib.load('model_metadata.pkl')
        return model, metadata
    except FileNotFoundError:
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Header
st.title("🌍 Multilingual ISIC Code Classifier")
st.markdown("**Enhanced for Kinyarwanda, English, and French**")
st.markdown("---")

# Load model
model, metadata = load_model()

if model is None:
    st.error("⚠️ Model files not found! Please run the training script first to generate 'best_text_classifier.pkl'")
    st.info("Run the enhanced multilingual training script to create the model file, then restart this app.")
    st.stop()

# Sidebar - Model Info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model Type:** {metadata.get('model_name', 'Unknown')}")
    st.write(f"**Test Accuracy:** {metadata.get('test_accuracy', 0):.2%}")
    st.write(f"**Feature Extraction:** {metadata.get('features', 'TF-IDF')}")
    st.write(f"**Total Training Samples:** {metadata.get('total_samples', 'N/A')}")
    st.write(f"**Number of ISIC Codes:** {metadata.get('num_classes', 'N/A')}")

    st.markdown("---")

    st.subheader("🌐 Supported Languages")
    languages = metadata.get('languages_supported', ['English', 'Kinyarwanda', 'French'])
    for lang in languages:
        st.markdown(f'<span class="language-badge">{lang}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("✨ Special Features")
    st.markdown("""
    <div class="feature-card"><strong>Character N-grams</strong><br>Captures Kinyarwanda morphology</div>
    <div class="feature-card"><strong>Word N-grams</strong><br>Understands semantic meaning</div>
    <div class="feature-card"><strong>Class Balancing</strong><br>Handles rare ISIC codes</div>
    """, unsafe_allow_html=True)

    if 'all_results' in metadata:
        st.markdown("---")
        st.subheader("📊 All Models Performance")
        results_df = pd.DataFrame(metadata['all_results'])
        st.dataframe(results_df, use_container_width=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Single Text Prediction", "📁 Batch CSV Prediction", "ℹ️ About"])

# TAB 1: Single Text Prediction
with tab1:
    st.subheader("Test Your Text Classification")

    with st.expander("💡 Example Texts in Different Languages"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🇬🇧 English**")
            if st.button("Try: Agriculture", key="ex1"):
                st.session_state.test_text = "Growing crops and raising livestock for food production"
        with col2:
            st.markdown("**🇷🇼 Kinyarwanda**")
            if st.button("Try: Ubuhinzi", key="ex2"):
                st.session_state.test_text = "Ubuhinzi bw'ibigori n'ubworozi bw'amatungo"
        with col3:
            st.markdown("**🇫🇷 French**")
            if st.button("Try: Agriculture (FR)", key="ex3"):
                st.session_state.test_text = "Culture de céréales et élevage d'animaux pour la production alimentaire"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Enter Text to Classify")
        input_text = st.text_area(
            "Input Text",
            height=200,
            placeholder="Type or paste your text here in English, Kinyarwanda, or French...",
            value=st.session_state.get('test_text', ''),
            label_visibility="collapsed",
            key="single_text"
        )
        st.caption(f"Characters: {len(input_text)} | Words: {len(input_text.split())}")
    with col2:
        st.subheader("Actions")
        predict_button = st.button("🚀 Classify Text", type="primary")
        if st.button("🗑️ Clear"):
            st.session_state.test_text = ''
            st.rerun()

    if predict_button:
        if not input_text.strip():
            st.warning("⚠️ Please enter some text to classify")
        else:
            with st.spinner("🔄 Analyzing text with multilingual model..."):
                try:
                    prediction = model.predict([input_text])[0]
                    probabilities = model.predict_proba([input_text])[0]
                    classes = model.classes_

                    prob_df = pd.DataFrame({
                        'ISIC Code': classes,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)

                    max_prob = probabilities.max()
                    st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="margin:0;">Predicted ISIC Code</h2>
                            <h1 style="margin:0.5rem 0; font-size: 3rem;">{prediction}</h1>
                            <p style="margin:0; font-size: 1.2rem;">Confidence: {max_prob:.1%}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    if max_prob > 0.7:
                        st.success("✅ High confidence prediction")
                    elif max_prob > 0.5:
                        st.warning("⚠️ Moderate confidence - review recommended")
                    else:
                        st.error("❌ Low confidence - manual review needed")

                    st.subheader("📈 Top 5 Most Likely ISIC Codes")
                    top5 = prob_df.head(5)
                    for _, row in top5.iterrows():
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.write(f"**{row['ISIC Code']}**")
                        with c2:
                            st.progress(float(row['Probability']))
                            st.caption(f"{row['Probability']:.1%}")

                    with st.expander("📋 View All Probabilities"):
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

# TAB 2: Batch CSV Prediction
with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.success(f"✅ File uploaded! {len(df_upload)} rows, {len(df_upload.columns)} columns.")
            st.dataframe(df_upload.head(10), use_container_width=True)

            text_column = st.selectbox("Select text column:", df_upload.columns.tolist())
            confidence_threshold = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.0, 0.05)

            if st.button("🚀 Predict ISIC Codes"):
                with st.spinner(f"Processing {len(df_upload)} rows..."):
                    df_clean = df_upload.dropna(subset=[text_column]).copy()
                    preds = model.predict(df_clean[text_column])
                    probs = model.predict_proba(df_clean[text_column])
                    max_probs = probs.max(axis=1)

                    df_clean['Predicted_ISIC_Code'] = preds
                    df_clean['Confidence'] = max_probs
                    df_clean['Confidence_Percent'] = (max_probs * 100).round(2).astype(str) + '%'

                    def level(c): 
                        return "High" if c > 0.7 else "Moderate" if c > 0.5 else "Low"
                    df_clean['Confidence_Level'] = df_clean['Confidence'].apply(level)

                    if confidence_threshold > 0:
                        df_clean = df_clean[df_clean['Confidence'] >= confidence_threshold]

                    st.dataframe(df_clean, use_container_width=True)

                    csv_buffer = io.StringIO()
                    df_clean.to_csv(csv_buffer, index=False)
                    st.download_button("⬇️ Download Results (CSV)", csv_buffer.getvalue(),
                        file_name="isic_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# TAB 3: About
with tab3:
    st.subheader("📖 About This Application")
    st.markdown(f"""
    ### 🌍 Multilingual ISIC Code Classifier
    
    This app uses advanced machine learning to classify business descriptions into 
    **International Standard Industrial Classification (ISIC)** codes — optimized for 
    **Kinyarwanda**, **English**, and **French** text.

    #### ✨ Key Features
    - Multilingual understanding
    - Word & character n-grams
    - Balanced training data
    - Batch or single-text prediction

    #### 🔧 Technical Details
    - **Algorithm**: {metadata.get('model_name', 'Unknown')}
    - **Test Accuracy**: {metadata.get('test_accuracy', 0):.2%}
    - **Training Samples**: {metadata.get('total_samples', 'N/A')}
    - **ISIC Codes**: {metadata.get('num_classes', 'N/A')} unique codes

    ---
    ### 📞 Support
    For low-confidence predictions, manual review is recommended.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280;'>"
    "🌍 Powered by Multilingual Machine Learning | Enhanced for Kinyarwanda, English & French"
    "</div>",
    unsafe_allow_html=True
)
