import streamlit as st
import numpy as np
import pandas as pd
import random
import base64
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ------------------------------------------------------
st.set_page_config(page_title="Credit Score Predictor", layout="wide")

# ------------------------------------------------------
# Futuristic UI Styling
# ------------------------------------------------------
def make_random_wallpaper():
        # generate 3 random colors and create a small SVG wallpaper
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(3)]
        svg = f"""<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 800 600'>
        <defs>
            <linearGradient id='g' x1='0' x2='1'>
                <stop offset='0' stop-color='{colors[0]}' stop-opacity='0.95'/>
                <stop offset='0.5' stop-color='{colors[1]}' stop-opacity='0.9'/>
                <stop offset='1' stop-color='{colors[2]}' stop-opacity='0.85'/>
            </linearGradient>
        </defs>
        <rect width='100%' height='100%' fill='url(#g)' />
        <g fill='white' opacity='0.04'>
            <circle cx='120' cy='80' r='120'/>
            <circle cx='700' cy='520' r='200'/>
        </g>
        <g opacity='0.06'>
            <rect x='-50' y='-50' width='900' height='700' fill='{colors[2]}' transform='rotate(12 400 300)'/>
        </g>
        </svg>"""
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode('utf-8')).decode('utf-8')

bg_uri = make_random_wallpaper()

css = """
    <style>
        /* FULL-PAGE CINEMATIC BACKGROUND */
        .main {
            /* red cinematic background with wallpaper overlay */
            background-image: linear-gradient(135deg, rgba(140,12,12,0.88) 0%, rgba(80,8,8,0.88) 60%), url('""" + bg_uri + """');
            background-size: cover;
            background-position: center;
            animation: backgroundMove 18s ease-in-out infinite alternate;
            color: #ffffff;
        }

        /* Smooth animated shimmer movement */
        @keyframes backgroundMove {
            0% {
                background-position: 0px 0px, 0px 0px, 0px 0px;
            }
            100% {
                background-position: 50px 80px, -60px -40px, 0px 0px;
            }
        }

        /* GLASS EFFECT CARD with green gradient border */
        .glass-card {
            background: rgba(255, 255, 255, 0.07);
            padding: 22px;
            border-radius: 18px;
            border: 4px solid;
            border-image: linear-gradient(90deg, #00ff88, #00eaff, #00ff88) 1;
            box-shadow: 0 8px 40px rgba(0,0,0,0.4);
            backdrop-filter: blur(16px);
        }

        /* Futuristic glowing title */
        h1 {
            text-shadow: 0 0 25px #00eaff, 0 0 45px #00bfff;
            font-family: 'Orbitron', sans-serif;
            letter-spacing: 1px;
        }

        h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            color: #d2e9ff;
            border-left: 4px solid #00eaff;
            padding-left: 10px;
        }

        /* Inputs */
        input {
            background-color: rgba(255,255,255,0.12) !important;
            color: white !important;
            border-radius: 12px !important;
        }

        /* Neon Button */
        .stButton > button {
            background: linear-gradient(90deg, #ff7a18, #ff8c00, #ffb199);
            background-size: 200%;
            animation: neonFlow 4s linear infinite;
            padding: 12px 30px;
            border-radius: 14px;
            color: white;
            border: none;
            transition: 0.3s;
            font-size: 18px;
            font-weight: bold;
            box-shadow: 0 6px 18px rgba(255,122,24,0.14);
        }

        /* Orange gradient for main navigation/home buttons */
        button[aria-label="About"],
        button[aria-label="Team"],
        button[aria-label="Model"],
        button[aria-label="Future Scope"] {
            background: linear-gradient(90deg, #ff7a18, #ffb199);
            background-size: 200%;
            color: white !important;
            box-shadow: 0 8px 30px rgba(255,122,24,0.18);
            border-radius: 12px !important;
            border: none !important;
        }
        button[aria-label="About"]:hover,
        button[aria-label="Team"]:hover,
        button[aria-label="Model"]:hover,
        button[aria-label="Future Scope"]:hover {
            transform: scale(1.04);
            box-shadow: 0 12px 40px rgba(255,122,24,0.28);
        }

        /* Button glow animation */
        @keyframes neonFlow {
            0% { background-position: 0% }
            100% { background-position: 200% }
        }

        .stButton > button:hover {
            transform: scale(1.08);
            box-shadow: 0 0 30px rgba(255,122,24,0.28);
        }

        /* Score output */
        .score-box {
            background: rgba(0, 255, 200, 0.12);
            border-left: 4px solid #1ef2c9;
            padding: 18px;
            font-size: 22px;
            border-radius: 14px;
            box-shadow: 0 0 20px rgba(0, 255, 200, 0.3);
        }

        /* Quick-jump selectbox: compact variant */
        .stSelectbox > div { margin-top: -4px; }
        div[data-baseweb="select"] { margin-top: -4px !important; }
        div[data-baseweb="select"] > div > div {
            /* subtle transparent background for contrast while remaining compact */
            background: rgba(0,0,0,0.12) !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            padding: 4px 10px !important;
            height: 36px !important; /* smaller than nav buttons for a compact look */
            display: flex !important;
            align-items: center !important;
            box-shadow: none !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            font-size: 14px !important;
        }
        /* keep the select text white and remove inner borders */
        div[data-baseweb="select"] select, .stSelectbox select {
            color: #ffffff !important;
            background: transparent !important;
            border: none !important;
            font-weight: 600;
            height: 100% !important;
            padding-top: 0 !important;
            font-size: 14px !important;
        }
        /* hide the dropdown caret for a cleaner look */
        div[data-baseweb="select"] svg { display: none !important; }

        /* Reduce horizontal spacing inside top row so items appear evenly spaced */
        .block-container .stButton { margin: 0 4px 0 4px; }

        /* Constrain selectbox width and center inside its column for equal spacing */
        .stSelectbox { max-width: 160px !important; margin: 0 auto !important; }
        div[data-baseweb="select"] { width: 140px !important; margin: 0 auto !important; }
        div[data-baseweb="select"] > div { justify-content: center !important; padding-left: 0 !important; }
    </style>
"""

st.markdown(css, unsafe_allow_html=True)


# ------------------------------------------------------
# Load Dataset + Model
# ------------------------------------------------------
DATA_PATH = "beneficiary_credit_20k.csv"
MODEL_PATH = "ann_credit_score.h5"

# top header will be rendered by nav_bar() below

# -------------------------
# Load dataset + model
# -------------------------
df = pd.read_csv(DATA_PATH)

# Select training features: use all columns from the dataset
# We'll detect numeric features and fit the scaler on those only.
selected_features = df.columns.tolist()


# Define user info fields to exclude from prediction
user_info_fields = ['name', 'gender', 'user_id', 'age']

# Use only numeric columns for scaling and inputs, but exclude user info fields and any label column
label_field = 'credit_score' if 'credit_score' in df.columns else None
numeric_features = df[selected_features].select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in user_info_fields]
if label_field and label_field in numeric_features:
    numeric_features.remove(label_field)

# Fit StandardScaler on numeric features
scaler = StandardScaler()
scaler.fit(df[numeric_features])

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# Determine an eligibility threshold: use median of label if available, otherwise fallback
if label_field and label_field in df.columns:
    try:
        ELIGIBILITY_THRESHOLD = float(df[label_field].median())
    except Exception:
        ELIGIBILITY_THRESHOLD = 600.0
else:
    ELIGIBILITY_THRESHOLD = 600.0

# -------------------------
# Navigation helpers
# -------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'about_section' not in st.session_state:
    st.session_state.about_section = None

def set_page(page_name):
    st.session_state.page = page_name

def nav_bar():
    # Layout: place each nav item in its own equally-sized column
    c0, c1, c2, c3, c4, c5 = st.columns([1,1,1,1,1,1])

    # Left: company name as Home
    with c0:
        if st.button('Creovate', key='nav_home'):
            set_page('Home')

    # Middle-left: compact quick-jump (search-like)
    with c1:
        quick_opts = ['Quick Jump', 'About', 'About Model', 'Team', 'Model']
        sel = st.selectbox('', quick_opts, index=0, key='top_search')
        if sel != quick_opts[0]:
            set_page(sel)
            st.session_state['top_search'] = quick_opts[0]

    # Remaining nav buttons evenly spaced
    with c2:
        if st.button('About', key='nav_about'):
            set_page('About')
    with c3:
        if st.button('About Model', key='nav_about_model'):
            set_page('About Model')
    with c4:
        if st.button('Team', key='nav_team'):
            set_page('Team')
    with c5:
        if st.button('Model', key='nav_model'):
            set_page('Model')

# Render header
nav_bar()

# Prominent centered app title beneath the header
st.markdown("<div style='text-align:center; margin: 18px 0 24px 0;'><h1 style='font-size:34px; color:#00eaff'>Credit Score Predictor</h1></div>", unsafe_allow_html=True)

# -------------------------
# Page renderers
# -------------------------
def render_home():
    st.markdown("""
    <div style='text-align:center; padding:40px;'>
        <h1 style='font-family:Orbitron, sans-serif; color:#00eaff; font-size:56px'>Creovate</h1>
        <h3 style='color:#d2e9ff;'>Beneficiary credit scoring with income verification layer for direct digital lending</h3>
    </div>
    """, unsafe_allow_html=True)

    
def render_about():
    st.header('About')
    st.write('Beneficiary credit scoring with an income-verification layer built for direct digital lending. The system combines applicant financial signals with verified income and household beneficiary data to produce a robust repayment likelihood score.')

    # Sections for detailed project report (buttons)
    sections = [
        'ABSTRACT','INTRODUCTION','LITERATURE SURVEY','PROBLEM STATEMENT','METHODOLOGY',
        'DATASET AND FEATURES','RESULTS AND DISCUSSIONS','CONCLUSION','FUTURE SCOPE','REFERENCES'
    ]

    # Layout buttons in rows of 3
    cols_per_row = 3
    # small summaries to show under each button
    summaries = {
        'ABSTRACT': 'High-level project overview and motivation.',
        'INTRODUCTION': 'Background, problem context and objectives.',
        'LITERATURE SURVEY': 'Related work and research gap.',
        'PROBLEM STATEMENT': 'Formal problem definition and math.',
        'METHODOLOGY': 'Data pipeline, modeling and verification engine.',
        'DATASET AND FEATURES': 'Sources, preprocessing and engineered features.',
        'RESULTS AND DISCUSSIONS': 'Model metrics, comparisons and XAI.',
        'CONCLUSION': 'Summary of findings and impact.',
        'FUTURE SCOPE': 'Possible extensions and integrations.',
        'REFERENCES': 'Key literature and sources.'
    }

    for i in range(0, len(sections), cols_per_row):
        row = sections[i:i+cols_per_row]
        cols = st.columns(len(row))
        for c, name in zip(cols, row):
            with c:
                if st.button(name, key=f'about_btn_{name}'):
                    st.session_state.about_section = name
                st.caption(summaries.get(name, ''))

    # Button to show team name (unique key)
    if st.button('Show Team Name', key='about_show_team'):
        st.success('Team: Creovate')

    # Content mapping
    about_content = {
        'ABSTRACT': """
PROJECT TITLE: AI-Driven Beneficiary Credit Scoring System with Income Verification Layer for Direct Digital Lending

Direct Digital Lending has emerged as a pivotal mechanism for accelerating financial inclusion, particularly for Economically Weaker Sections (EWS) and Backward Class (BC) beneficiaries in India. Institutions such as the National Backward Classes Finance & Development Corporation (NBCFDC) are mandated to extend concessional credit to support self-employment, micro-entrepreneurship, and income-generation activities. However, despite the rapid proliferation of the Unified Payments Interface (UPI) and the Jan Dhan-Aadhaar-Mobile (JAM) trinity, the specific niche of welfare-based lending remains plagued by systemic inefficiencies. The core challenges persist in beneficiary evaluation, income verification, and creditworthiness assessment.

This project proposes and implements a comprehensive, Artificial Intelligence-driven Beneficiary Credit Scoring System integrated with a novel Income Verification Layer. The solution moves beyond traditional heuristic models by deploying a Deep Learning framework. Specifically, an Artificial Neural Network (ANN) is engineered to analyze non-linear relationships between repayment behavior, business activity metrics, financial stability, and socio-economic indicators. A multi-tiered Income Verification Engine employing statistical rule-mining and anomaly detection validates reported income against observed business activity and lifestyle indicators. The framework includes Explainable AI mechanisms to make algorithmic decisions transparent and auditable.
""",

        'INTRODUCTION': """
CHAPTER 1: INTRODUCTION
1.1 Background and Context
Financial inclusion is widely recognized as a cornerstone of national development and poverty alleviation. In the Indian context, the government has made significant strides through schemes like Mudra Yojana and PM SVANidhi. However, marginalized communities—specifically the Backward Classes (BCs)—often face systemic barriers in accessing formal credit. The NBCFDC bridges this gap by providing subsidized loans through a network of State Channelizing Agencies (SCAs) and channel partners. These partners are responsible for the last-mile delivery of credit but the operational model is often antiquated and prone to manual bias.

1.2 The Digital Lending Disconnect
Modern commercial lending uses sophisticated algorithms that analyze thousands of data points. Public sector welfare lending often lags behind, suffering from Data Rich, Information Poor (DRIP), the Income Verification Paradox, and Subjectivity in Risk Assessment.

1.3 Need for Transformation
To ensure sustainability, the system must shift from Asset-Based Lending to Cash-Flow Based Lending and leverage ML to identify repayment patterns and reduce subjectivity.
""",

        'LITERATURE SURVEY': """
CHAPTER 2: LITERATURE REVIEW
A systematic review covers classical statistical methods, machine learning approaches, and welfare lending research gaps. Topics include Logistic Regression, Ensemble Methods (Random Forest, XGBoost), Neural Networks in finance, behavioral scoring, and the research gap in welfare-oriented credit scoring.
""",

        'PROBLEM STATEMENT': """
CHAPTER 3: PROBLEM STATEMENT AND FORMULATION
The NBCFDC and its channel partners face a scalability vs quality dilemma. Manual verification causes Type I and Type II errors. We define the problem as a binary classification task and introduce an Income Verification function V(I_decl, I_calc) that flags discrepancies beyond a threshold δ.
""",

        'METHODOLOGY': """
CHAPTER 4: METHODOLOGY
We adopt CRISP-DM adapted to welfare lending. Steps include Data Collection, Advanced Preprocessing (imputation, winsorization, encoding, class balancing with SMOTE), Feature Engineering (DTI, RCI, Income Stability), Model Architecture (ANN with ReLU, Dropout, Sigmoid output), and Income Verification Engine (heuristic checks, pattern matching, Isolation Forest).
Mathematical formulations and training parameters (loss, optimizer, batch size, epochs) are included in the full report.
""",

        'DATASET AND FEATURES': """
CHAPTER 5: DATASET AND FEATURES
Data is aggregated from channel partners across geographies. Feature types: Demographic, Financial, Historical, and Target. Preprocessing includes imputation, outlier treatment, encoding, and class balancing. Engineered features like DTI and RCI are central to predictive performance.
""",

        'RESULTS AND DISCUSSIONS': """
CHAPTER 6: RESULTS AND DISCUSSION
We prioritize recall and present performance metrics: ANN Accuracy ~91.5%, ROC-AUC 0.94, Recall for defaulter class ~91%. Comparative analysis with Logistic Regression and Random Forest is provided. Explainability via SHAP yields actionable reason codes for decisions.
""",

        'CONCLUSION': """
CHAPTER 9: CONCLUSION
The AI-Driven Beneficiary Credit Scoring System improves decision consistency, transparency, and sustainability of concessional lending by combining ANN scoring with an Income Verification Layer and XAI explanations.
""",

        'FUTURE SCOPE': """
CHAPTER 10: FUTURE SCOPE
Future integrations include Account Aggregator for consented bank statements, psychometric scoring, geo-spatial analytics, and federated learning for privacy-preserving updates.
""",

        'REFERENCES': """
REFERENCES
Kumar, A., & Bhattacharya, D. S. (2021). Credit Score Prediction System using Deep Learning and K-Means Algorithms.
Bhilare, M., Wadajkar, V., & Kale, K. (2024). Empowering Financial Decisions: Precision Credit Scoring through Advanced Machine Learning.
Pavitha, N., & Sugave, S. (2024). Interpretive Ensemble Framework for Credit Default Risk Forecasting.
West, D. (2000). Neural network credit scoring models.
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning.
Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
"""
    }

    # Display selected section in a centered card
    sel = st.session_state.get('about_section')
    if sel:
        content = about_content.get(sel, 'No content available.')
        st.markdown('<div style="width:100%; margin: 18px 6px;">', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"<h3 style='color:#00eaff'>{sel}</h3>", unsafe_allow_html=True)
        st.write(content)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button('Back to About Home', key='about_back'):
            st.session_state.about_section = None
    else:
        st.info('Select a section above to view detailed content.')

def render_team():
    st.header('Team')
    st.markdown('**Meet the Creovate Team and Assigned Roles:**')
    team = {
        'Tharun': 'Model Trining ,streamlit',
        'Vinay': 'Model Traning and Research Paper',
        'Ashish': ' Research Paper ,Data Collection, Model Training',
        'Yashwanth': 'Report and Data Preprocessing',
    }
    for name, role in team.items():
        st.markdown(f"**{name}** — {role}")


def render_about_model():
    st.header('About the Model')
    st.markdown(
        """
        ## 🤖 Models We Used 

        To make loan decisions smarter and fairer, we tested many different AI models. Each one looks at your information in a different way. Here's the simple meaning of each:

        
        🔥 1. ANN – Artificial Neural Network (Our Main Model)

        Think of this as the brain of the system. It learns patterns just like a human would — from income, credit score, experience, and many other factors. It understands complex relationships that other models may miss. This is the model that performed the best, so we used it as our primary engine.

        
        📊 2. Logistic Regression

        This is the simplest model. It works like a yes/no calculator. It helps us understand basic trends but isn’t powerful enough for deep patterns. Good for transparency, not accuracy.

        
        🎯 3. SVM (Support Vector Machine)

        Imagine drawing a line between “Approved” and “Not Approved” people. SVM tries to draw the best possible line. It’s accurate for some problems, but not the best for our data.

        
        🌲 4. Random Forest

        This model makes hundreds of decision trees and then takes a vote. Each tree asks questions like: “Is income high?” or “Is loan amount too large?” Together, they make a stronger decision. It works well, but still not as strong as the ANN.

        
        ⚡ 5. Gradient Boosting

        This model learns from mistakes. Each new tree fixes the errors of the previous one. This makes it smarter over time. Great accuracy, but a bit slower.

        
        🚀 6. XGBoost

        This is like Gradient Boosting on steroids. Super fast, super accurate, widely used in fintech. It came close to the ANN, but ANN still performed better.

        
        ⚡ 7. LightGBM

        A faster, lighter version of boosting. Amazing performance on large datasets. Almost as strong as XGBoost — but again, ANN was ahead.

        
        ⭐ Why We Used So Many Models?

        Because real AI development isn’t guessing — it’s testing. We tried all these models to see which one:
        - Understands people best
        - Predicts approvals most accurately
        - Makes fewer mistakes
        - Works on real-world beneficiary data

        And after all comparisons… 👉 ANN gave the highest accuracy and most reliable decisions.**

        So it became the model powering the final system.
        """,
        unsafe_allow_html=True,
    )

def render_future_scope():
    st.header('Future Scope')
    st.markdown('- Integrate real-time income verification APIs')
    st.markdown('- Add explainability (SHAP/feature attribution) for model outputs')
    st.markdown('- Onboard more data sources for better credit context')
    st.markdown('- Deploy a light-weight mobile-friendly UI and API')

def render_model_page():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Applicant Profile")

    # User info fields (not used in prediction)
    user_name = st.text_input("Name")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    user_id = st.text_input("User ID")
    age = st.number_input("Age", min_value=0, value=18)

    # Image upload / preview (optional)
    st.markdown("<br>", unsafe_allow_html=True)
    image_file = st.file_uploader("Upload applicant image (optional)", type=["png", "jpg", "jpeg"], key='applicant_image')
    if image_file is not None:
        try:
            img = Image.open(image_file)
            st.image(img, caption='Applicant Image Preview', use_column_width=True)
        except Exception:
            st.warning('Uploaded file could not be opened as an image.')

    # Document validity selector (Not used for prediction unless you choose to)
    doc_options = ["Not Provided", "Verified", "Unverified"]
    default_idx = 0 if image_file is None else 2
    doc_validity = st.selectbox("Document Validity", doc_options, index=default_idx, key='document_validity')

    st.markdown("---")

    # Dynamically create inputs for every numeric feature used in prediction (excluding user info fields and label)
    input_values = {}
    for feat in numeric_features:
        default = 0.0
        try:
            default = float(df[feat].median())
        except Exception:
            default = 0.0
        input_values[feat] = st.number_input(f"{feat}", value=default)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Predict Futuristic Credit Score"):
        # Build input vector in the same column order used to fit the scaler
        input_list = [float(input_values[feat]) for feat in numeric_features]
        input_data = np.array(input_list).reshape(1, -1)

        # Scale input (must match the scaler's feature order)
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        # Try to extract a single float value from prediction
        try:
            if hasattr(prediction, 'flatten'):
                predicted_score = float(prediction.flatten()[0])
            else:
                predicted_score = float(prediction[0])

            st.markdown(
                f"""
                <div class="score-box">
                    🔮 <strong>Predicted Credit Score:</strong>
                    <span style="color:#00ffd9; font-weight:bold;">{predicted_score:.2f}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Eligibility message based on threshold
            try:
                threshold = ELIGIBILITY_THRESHOLD
                if predicted_score >= threshold:
                    st.success(f"✅ You are eligible for a credit loan. (Threshold: {threshold:.2f})")
                else:
                    st.warning(f"⚠️ You are not eligible for a credit loan at this time. (Threshold: {threshold:.2f})")
            except Exception:
                st.info('Eligibility could not be determined automatically.')

            # Show document validity status and advise if not verified
            try:
                st.info(f"Document validity: {doc_validity}")
                if doc_validity != 'Verified':
                    st.warning('Document is not verified — eligibility may require manual verification.')
            except Exception:
                pass

        except Exception as e:
            st.error(f"Prediction failed: {e}\nRaw model output: {prediction}")

# -------------------------
# Main page switch
# -------------------------
page = st.session_state.page
if page == 'Home':
    render_home()
elif page == 'About':
    render_about()
elif page == 'Team':
    render_team()
elif page == 'Model':
    render_model_page()
elif page == 'About Model':
    render_about_model()
elif page == 'Future Scope':
    render_future_scope()
