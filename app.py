import json
import joblib
import base64
import pandas as pd
import streamlit as st
from tensorflow import keras
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & ASSETS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="🎓 Backbencher's Oracle | Grade Predictor",
    page_icon="🍌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_base64_image(image_path: str) -> str:
    """Convert image to base64 for embedding in HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

# Load banana images as base64
banana_fail_b64 = get_base64_image("assets/banana_fail.png")
banana_mid_b64 = get_base64_image("assets/banana_mid.png")
banana_success_b64 = get_base64_image("assets/banana_success.png")
header_banner_b64 = get_base64_image("assets/header_banner.png")

# -----------------------------------------------------------------------------
# 3. CUSTOM CSS (THEME: WARM WHITE / CREAM / GLASS)
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700&display=swap');
    
    /* GLOBAL WARM WHITE THEME */
    .stApp {{
        background: linear-gradient(135deg, #FAF8F5 0%, #F5F1EB 50%, #EDE8E0 100%);
        font-family: 'Inter', sans-serif;
    }}
    
    /* Hide default Streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* BACKGROUND DECORATIONS */
    .stApp::before {{
        content: '';
        position: fixed;
        top: -50%;
        right: -30%;
        width: 800px;
        height: 800px;
        background: radial-gradient(circle, rgba(255, 220, 150, 0.15) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }}
    
    .stApp::after {{
        content: '';
        position: fixed;
        bottom: -30%;
        left: -20%;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(200, 180, 160, 0.1) 0%, transparent 70%);
        pointer-events: none;
        z-index: 0;
    }}
    
    /* GLASSMORPHISM CARDS */
    .glass-card {{
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 24px;
        box-shadow: 
            0 4px 24px rgba(139, 115, 85, 0.08),
            0 1px 2px rgba(139, 115, 85, 0.04),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        position: relative;
        z-index: 1;
    }}
    
    .glass-card-elevated {{
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.9);
        border-radius: 28px;
        padding: 40px;
        margin-bottom: 24px;
        box-shadow: 
            0 8px 40px rgba(139, 115, 85, 0.12),
            0 2px 4px rgba(139, 115, 85, 0.06),
            inset 0 2px 0 rgba(255, 255, 255, 1);
        position: relative;
        z-index: 1;
    }}
    
    /* HEADER STYLES */
    .hero-section {{
        text-align: center;
        padding: 20px 0 30px 0;
        position: relative;
        z-index: 1;
    }}
    
    .hero-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #1A1715 !important;
        margin-bottom: 12px;
        letter-spacing: -1px;
        text-shadow: none;
        text-align: center !important;
    }}
    
    .hero-subtitle {{
        font-size: 1.15rem;
        color: #4A4744 !important;
        font-weight: 400;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
        text-align: center !important;
    }}
    
    .hero-emoji {{
        font-size: 4rem;
        margin-bottom: 16px;
        display: block;
        animation: bounce 2s ease-in-out infinite;
    }}
    
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
    }}
    
    /* SECTION TITLES */
    .section-title {{
        font-family: 'Outfit', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #3D3A36;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    
    .section-icon {{
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #FFD27D 0%, #F5A623 100%);
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        box-shadow: 0 2px 8px rgba(245, 166, 35, 0.3);
    }}
    
    /* INPUT WIDGET STYLING */
    .stSelectbox > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(200, 190, 180, 0.4) !important;
        color: #3D3A36 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.06);
        transition: all 0.3s ease;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: rgba(245, 166, 35, 0.5) !important;
        box-shadow: 0 4px 16px rgba(245, 166, 35, 0.1);
    }}
    
    .stNumberInput > div > div {{
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(200, 190, 180, 0.4) !important;
        color: #3D3A36 !important;
        border-radius: 12px !important;
    }}
    
    /* NUMBER INPUT - Inner input field styling */
    .stNumberInput input {{
        background: rgba(255, 255, 255, 0.95) !important;
        color: #1A1715 !important;
        border: none !important;
    }}
    
    .stNumberInput > div > div > div {{
        background: rgba(255, 255, 255, 0.95) !important;
    }}
    
    /* Step buttons (+/-) styling */
    .stNumberInput button {{
        background: rgba(250, 248, 245, 1) !important;
        border: 1px solid rgba(200, 190, 180, 0.5) !important;
        color: #3D3A36 !important;
    }}
    
    .stNumberInput button:hover {{
        background: rgba(245, 241, 235, 1) !important;
        border-color: #F5A623 !important;
    }}
    
    /* SLIDER STYLING */
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, #FFD27D 0%, #F5A623 100%) !important;
    }}
    
    /* TOGGLE STYLING */
    .stToggle > label {{
        color: #1A1715 !important;
        font-weight: 500 !important;
    }}
    
    /* ALL FORM LABELS */
    .stForm label, 
    .stSelectbox label, 
    .stNumberInput label, 
    .stSlider label,
    .stToggle label {{
        color: #1A1715 !important;
        font-weight: 500 !important;
    }}
    
    /* Additional Factors text and all strong/bold text */
    .stForm strong, .stForm b, .stMarkdown strong, .stMarkdown b {{
        color: #1A1715 !important;
        font-weight: 600 !important;
    }}
    
    /* Checkbox and toggle text */
    [data-testid="stCheckbox"] label span,
    [data-testid="stToggle"] label span {{
        color: #1A1715 !important;
    }}
    
    /* PREDICT BUTTON */
    .stButton > button {{
        background: linear-gradient(135deg, #3D3A36 0%, #2D2A26 100%);
        color: #FAF8F5 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 14px;
        width: 100%;
        padding: 18px 24px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: none;
        letter-spacing: 0.3px;
        box-shadow: 
            0 4px 20px rgba(45, 42, 38, 0.2),
            0 2px 4px rgba(45, 42, 38, 0.1);
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button::before {{
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        transition: left 0.5s;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 
            0 8px 30px rgba(45, 42, 38, 0.25),
            0 4px 8px rgba(45, 42, 38, 0.15);
    }}
    
    .stButton > button:hover::before {{
        left: 100%;
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* RESULT CARD */
    .result-card {{
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 252, 248, 0.95) 100%);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.95);
        border-radius: 28px;
        padding: 48px;
        text-align: center;
        box-shadow: 
            0 12px 60px rgba(139, 115, 85, 0.15),
            0 4px 8px rgba(139, 115, 85, 0.08),
            inset 0 2px 0 rgba(255, 255, 255, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .result-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        border-radius: 28px 28px 0 0;
    }}
    
    .result-fail::before {{
        background: linear-gradient(90deg, #FF6B6B, #EE5A5A);
    }}
    
    .result-mid::before {{
        background: linear-gradient(90deg, #FFB347, #FFA500);
    }}
    
    .result-success::before {{
        background: linear-gradient(90deg, #4ECDC4, #2ECC71);
    }}
    
    .result-score {{
        font-family: 'Outfit', sans-serif;
        font-size: 4.5rem;
        font-weight: 800;
        margin: 20px 0;
        letter-spacing: -2px;
    }}
    
    .result-score-fail {{
        color: #E74C3C;
        text-shadow: 0 4px 20px rgba(231, 76, 60, 0.2);
    }}
    
    .result-score-mid {{
        color: #F39C12;
        text-shadow: 0 4px 20px rgba(243, 156, 18, 0.2);
    }}
    
    .result-score-success {{
        color: #27AE60;
        text-shadow: 0 4px 20px rgba(39, 174, 96, 0.2);
    }}
    
    .result-max {{
        font-size: 1.5rem;
        color: #8B8580;
        font-weight: 400;
    }}
    
    .result-verdict {{
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 16px;
        padding: 12px 24px;
        border-radius: 50px;
        display: inline-block;
    }}
    
    .verdict-fail {{
        background: rgba(231, 76, 60, 0.1);
        color: #C0392B;
    }}
    
    .verdict-mid {{
        background: rgba(243, 156, 18, 0.1);
        color: #D68910;
    }}
    
    .verdict-success {{
        background: rgba(39, 174, 96, 0.1);
        color: #1E8449;
    }}
    
    .banana-mascot {{
        width: 180px;
        height: 180px;
        margin: 0 auto 20px auto;
        display: block;
        animation: float 3s ease-in-out infinite;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0) rotate(0deg); }}
        50% {{ transform: translateY(-15px) rotate(3deg); }}
    }}
    
    /* PROGRESS BAR */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #FFD27D 0%, #F5A623 100%);
        border-radius: 10px;
    }}
    
    /* DIVIDER */
    .custom-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139, 115, 85, 0.2), transparent);
        margin: 32px 0;
        border: none;
    }}

    /* FOOTER */
    .footer-container {{
        text-align: center;
        padding: 40px 20px;
        margin-top: 40px;
    }}
    
    .footer-text {{
        color: #8B8580;
        font-size: 0.95rem;
        margin-bottom: 24px;
    }}
    
    .footer-credit {{
        font-size: 0.85rem;
        color: #A5A09A;
        margin-top: 24px;
    }}
    
    .social-links {{
        display: flex;
        justify-content: center;
        gap: 16px;
        flex-wrap: wrap;
    }}
    
    .social-link {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 14px 28px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(200, 190, 180, 0.3);
        border-radius: 50px;
        text-decoration: none;
        color: #3D3A36 !important;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(139, 115, 85, 0.06);
    }}
    
    .social-link:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(139, 115, 85, 0.12);
        border-color: rgba(200, 190, 180, 0.5);
        background: rgba(255, 255, 255, 0.95);
    }}
    
    .social-icon {{
        width: 22px;
        height: 22px;
    }}
    
    /* LinkedIn Blue */
    .linkedin-link:hover {{
        border-color: #0077B5;
        box-shadow: 0 8px 24px rgba(0, 119, 181, 0.15);
    }}
    
    /* GitHub Dark */
    .github-link:hover {{
        border-color: #333;
        box-shadow: 0 8px 24px rgba(51, 51, 51, 0.15);
    }}
    
    /* NANO BANANA FLOATING */
    .nano-banana {{
        position: fixed;
        top: 24px;
        right: 24px;
        width: 60px;
        height: 60px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        box-shadow: 0 4px 20px rgba(139, 115, 85, 0.15);
        animation: float 4s ease-in-out infinite;
        z-index: 9999;
        border: 1px solid rgba(255, 255, 255, 0.9);
    }}
    
    /* FORM LABELS */
    .stForm label {{
        color: #4D4A46 !important;
        font-weight: 500 !important;
    }}
    
    /* EXPANDER */
    .streamlit-expanderHeader {{
        background: rgba(255, 255, 255, 0.6) !important;
        border-radius: 12px !important;
        color: #4D4A46 !important;
        font-weight: 500 !important;
    }}
    
    /* RESPONSIVE */
    @media (max-width: 768px) {{
        .hero-title {{
            font-size: 2.2rem;
        }}
        .result-score {{
            font-size: 3rem;
        }}
        .banana-mascot {{
            width: 140px;
            height: 140px;
        }}
        .social-links {{
            flex-direction: column;
            gap: 12px;
        }}
    }}
</style>

<div class="nano-banana">🍌</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. BACKEND LOGIC (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = keras.models.load_model("student_grade_ann_best.keras")
        preprocessor = joblib.load("preprocessor.joblib")
        with open("feature_columns.json", "r") as f:
            feature_cols = json.load(f)
        return model, preprocessor, feature_cols
    except Exception as e:
        st.error(f"⚠️ Could not load model files. Please ensure all required files are in the directory. Error: {e}")
        return None, None, None

model, preprocessor, FEATURE_COLS = load_artifacts()

DEFAULT_NUM = 0
DEFAULT_CAT = "no"

def build_input_row(user_inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame with:
    - exact FEATURE_COLS order
    - numeric/categorical defaults for missing values
    - safe dtypes for the preprocessor
    """
    if not FEATURE_COLS:
        return pd.DataFrame()

    row = {}

    # Columns that are truly numeric in training
    numeric_cols = {
        "age", "studytime", "failures", "famrel", "freetime", "goout",
        "Dalc", "Walc", "health", "absences", "traveltime", "Medu",
        "Fedu"
    }

    # Yes/No categorical flags
    yes_no_cols = {
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    }

    for col in FEATURE_COLS:
        if col in user_inputs:
            val = user_inputs[col]
        else:
            # sensible defaults
            if col in yes_no_cols:
                val = DEFAULT_CAT
            elif col in numeric_cols:
                val = DEFAULT_NUM
            else:
                # for other categoricals (school, address, etc.) put a neutral placeholder
                val = user_inputs.get(col, DEFAULT_CAT)

        row[col] = val

    df = pd.DataFrame([row], columns=FEATURE_COLS)

    # Cast numeric columns explicitly to numeric dtype
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------------------------------------------------------
# 5. MAIN UI LAYOUT
# -----------------------------------------------------------------------------

# Hero Section
st.markdown("""
<div class="hero-section">
    <span class="hero-emoji">🎓</span>
    <h1 class="hero-title">Backbencher's Oracle</h1>
    <p class="hero-subtitle">Discover your academic destiny with AI. Enter your details and let the oracle reveal your predicted grade.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Main Logic Block
if model:
    # --- GLASS CONTAINER FOR INPUTS ---
    st.markdown('<div class="glass-card-elevated">', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-title">
        <span class="section-icon">📝</span>
        Student Profile
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("student_form"):
        user = {}
        
        # Row 1 - Identity & Subject
        c1, c2 = st.columns(2)
        with c1:
            if "sex" in FEATURE_COLS:
                user["sex"] = st.selectbox("👤 Gender", ["F", "M"], help="Select your gender")
            if "age" in FEATURE_COLS:
                user["age"] = st.slider("🎂 Age", 15, 22, 17)
        with c2:
            if "subject" in FEATURE_COLS:
                user["subject"] = st.selectbox("📚 Subject", ["math", "portuguese"])
            if "studytime" in FEATURE_COLS:
                user["studytime"] = st.select_slider(
                    "⏰ Weekly Study Hours", 
                    options=[1, 2, 3, 4], 
                    value=2,
                    format_func=lambda x: {
                        1: "< 2 hours", 
                        2: "2-5 hours", 
                        3: "5-10 hours", 
                        4: "> 10 hours"
                    }[x]
                )

        # Row 2 - Academic History
        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            if "failures" in FEATURE_COLS:
                user["failures"] = st.number_input("❌ Past Failures", 0, 4, 0, help="Number of past class failures")
        with c4:
            if "absences" in FEATURE_COLS:
                user["absences"] = st.number_input("🏃 Absences", 0, 99, 3, help="Number of school absences")

        # Row 3 - Lifestyle Toggles
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🎯 Additional Factors**", unsafe_allow_html=True)
        
        t1, t2, t3, t4 = st.columns(4)
        
        with t1:
            if "schoolsup" in FEATURE_COLS:
                user["schoolsup"] = "yes" if st.toggle("📖 Extra Classes", help="Extra educational support") else "no"
        with t2:
            if "internet" in FEATURE_COLS:
                user["internet"] = "yes" if st.toggle("🌐 Internet Access", value=True, help="Home internet access") else "no"
        with t3:
            if "romantic" in FEATURE_COLS:
                user["romantic"] = "yes" if st.toggle("💕 Relationship", help="In a romantic relationship") else "no"
        with t4:
            if "famsup" in FEATURE_COLS:
                user["famsup"] = "yes" if st.toggle("👨‍👩‍👧 Family Support", value=True, help="Family educational support") else "no"

        st.markdown("<br><br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 Reveal My Grade")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- RESULT SECTION ---
    if submitted:
        x = build_input_row(user)
    
        if x.empty:
            st.error("⚠️ Could not build input row. Please try again.")
        else:
            st.write("🔍 Debug – input to model:")
            st.write("Columns:", list(x.columns))
            st.write(x.dtypes)
    
            with st.spinner("✨ The Oracle is analyzing your fate..."):
                try:
                    x_p = preprocessor.transform(x)
                    x_p = x_p.toarray() if hasattr(x_p, "toarray") else x_p
                    pred = float(model.predict(x_p, verbose=0).flatten()[0])
                except Exception as e:
                    st.error(
                        "⚠️ Something went wrong while preparing your data for the model. "
                        "Check that all required inputs are present and valid."
                    )
                    st.code(str(e))
                    st.stop()


        # Determine result category
        if pred < 10:
            result_class = "fail"
            score_class = "result-score-fail"
            verdict_class = "verdict-fail"
            verdict_text = "💔 Needs Improvement - Time to step up!"
            banana_img = banana_fail_b64
        elif pred < 14:
            result_class = "mid"
            score_class = "result-score-mid"
            verdict_class = "verdict-mid"
            verdict_text = "🌟 Passing - You're on the right track!"
            banana_img = banana_mid_b64
        else:
            result_class = "success"
            score_class = "result-score-success"
            verdict_class = "verdict-success"
            verdict_text = "🏆 Excellent - You're crushing it!"
            banana_img = banana_success_b64

        # Display Result Card
        st.markdown(f"""
        <div class="result-card result-{result_class}">
            <img src="data:image/png;base64,{banana_img}" class="banana-mascot" alt="Banana Mascot">
            <h3 style="color: #6B6560; font-size: 1.1rem; font-weight: 500; margin: 0;">Predicted Grade</h3>
            <div class="result-score {score_class}">{pred:.1f}<span class="result-max"> / 20</span></div>
            <div class="result-verdict {verdict_class}">{verdict_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(min(pred/20, 1.0))
        
        # Detailed breakdown
        with st.expander("📊 View Input Details"):
            st.dataframe(x, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

footer_html = """<div class="footer-container">
<p class="footer-text">Built with ❤️ and a little bit of AI magic</p>
<div class="social-links">
<a href="https://www.linkedin.com/in/mayank-goyal-mg09/" target="_blank" class="social-link linkedin-link">
<svg class="social-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#0077B5"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"></path></svg>
Mayank Goyal
</a>
<a href="https://github.com/mayank-goyal09" target="_blank" class="social-link github-link">
<svg class="social-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#333"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"></path></svg>
GitHub
</a>
</div>
<p class="footer-credit">© 2026 Backbencher's Oracle • Powered by Neural Networks 🧠</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)

