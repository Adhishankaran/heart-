import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set page configuration
st.set_page_config(
    page_title="Heart Disease KNN Classifier Lab",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Slate Dark theme with red and emerald accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek card containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10B981;
        line-height: 1;
        margin-top: 8px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
    }
    
    /* Highlighted badges */
    .badge {
        padding: 6px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-positive { background-color: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-negative { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    
    /* Custom button styles */
    .stButton>button {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load data helper
@st.cache_data
def load_data():
    csv_path = "heart.csv"
    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at {csv_path}!")
        return None
    return pd.read_csv(csv_path)

df_raw = load_data()

if df_raw is not None:
    # Page Header
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-size: 2.8rem; margin-bottom: 5px; background: linear-gradient(135deg, #F87171 0%, #EF4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🩺 Heart Disease KNN Classifier Lab
        </h1>
        <p style='font-size: 1.1rem; color: #94A3B8;'>
            Interactive Machine Learning Dashboard to analyze and predict heart disease risks using K-Nearest Neighbors
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================
    # SIDEBAR SETUP (Hyperparameters & Info)
    # ==========================================
    st.sidebar.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='margin-bottom: 5px; font-size: 1.5rem;'>🛠️ KNN Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    k_val = st.sidebar.slider("Number of Neighbors (K)", min_value=1, max_value=20, value=5, step=1,
                              help="Number of nearest neighbors to search in prediction.")
    
    metric_display = st.sidebar.selectbox("Distance Metric", 
                                         options=["Euclidean (L2 Norm)", "Manhattan (L1 Norm)", "Cosine Similarity"],
                                         index=0,
                                         help="Algorithm used to compute distance in feature space.")
    
    metric_map = {
        "Euclidean (L2 Norm)": "euclidean",
        "Manhattan (L1 Norm)": "manhattan",
        "Cosine Similarity": "cosine"
    }
    metric = metric_map[metric_display]
    
    scale_features = st.sidebar.checkbox("Apply Feature Scaling", value=True,
                                         help="Normalize features to have mean=0 and variance=1. Strongly recommended for KNN.")
    
    remove_duplicates = st.sidebar.checkbox("Remove Duplicate Records", value=True,
                                            help="Remove identical patients from dataset to prevent data leakage during train/test split. (Crucial for realistic validation!)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 💡 KNN Education
    * **Data Leakage Check:**
      * The raw dataset contains **1,025** rows, but only **302** are unique.
      * Leaving duplicates leads to **100% accuracy** due to identical patients split across train/test sets.
      * Toggling **Remove Duplicates** shows the realistic generalization accuracy (~78%-80%).
    * **Why Scale?**
      * Without scaling, features like *Cholesterol* (~240) completely dwarf features like *ST Depression* (~1.5) or *Chest Pain Code* (0-3).
    """)
    
    # ==========================================
    # DATA PREPROCESSING PIPELINE
    # ==========================================
    # Process dataset based on sidebar options
    if remove_duplicates:
        df_processed = df_raw.drop_duplicates().copy()
    else:
        df_processed = df_raw.copy()
        
    X = df_processed.drop(columns=['target'])
    y = df_processed['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    if scale_features:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        
    # Fit Model
    try:
        knn = KNeighborsClassifier(n_neighbors=k_val, metric=metric)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        
        # Performance
        test_accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
    except Exception as e:
        st.error(f"Error training model: {e}")
        test_accuracy = 0
        conf_matrix = np.zeros((2,2))
        clf_report = {}
        
    # Tabs layout
    tab1, tab2 = st.tabs([
        "📊 Model Dashboard & EDA",
        "🔮 Patient Diagnosis & Projection"
    ])
    
    # ==========================================
    # TAB 1: MODEL DASHBOARD & EDA
    # ==========================================
    with tab1:
        st.markdown("### 📊 Dataset Overview & KNN Performance")
        st.write("Understand the dataset features and inspect how hyperparameters change the model's metrics in real-time.")
        
        # Section 1: Quick dataset statistics cards
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <span class="metric-label">Dataset Size</span>
                <div class="metric-value" style="color: #3B82F6;">{len(df_processed)}</div>
                <span style="font-size:0.75rem; color:#94A3B8;">{'Unique patients' if remove_duplicates else 'Raw rows'}</span>
            </div>
            """, unsafe_allow_html=True)
        with col_stat2:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <span class="metric-label">Test Accuracy</span>
                <div class="metric-value">{test_accuracy*100:.2f}%</div>
                <span style="font-size:0.75rem; color:#94A3B8;">On unseen test split</span>
            </div>
            """, unsafe_allow_html=True)
        with col_stat3:
            disease_pct = (y == 1).mean() * 100
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <span class="metric-label">Disease Prevalence</span>
                <div class="metric-value" style="color: #F87171;">{disease_pct:.1f}%</div>
                <span style="font-size:0.75rem; color:#94A3B8;">Target = 1 positive</span>
            </div>
            """, unsafe_allow_html=True)
        with col_stat4:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; padding: 15px;">
                <span class="metric-label">Best CV K-Value</span>
                <div class="metric-value" style="color: #FBBF24;">{k_val}</div>
                <span style="font-size:0.75rem; color:#94A3B8;">Selected by user</span>
            </div>
            """, unsafe_allow_html=True)
            
        # Section 2: Confusion matrix and parameter tuning curves
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.markdown("#### Interactive Confusion Matrix")
            # Build heat map
            z = conf_matrix
            x = ['Predicted Normal (0)', 'Predicted Disease (1)']
            y_labels = ['Actual Normal (0)', 'Actual Disease (1)']
            
            fig_conf = px.imshow(
                z, x=x, y=y_labels,
                text_auto=True,
                color_continuous_scale='Reds',
                labels=dict(x="Prediction", y="Ground Truth")
            )
            fig_conf.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#F1F5F9'),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300
            )
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Classification details
            if clf_report:
                st.write("**Detailed Classification Report:**")
                df_report = pd.DataFrame(clf_report).transpose().iloc[:3, :3] # Keep class 0, class 1, accuracy
                st.dataframe(df_report.style.format("{:.3f}"), use_container_width=True)
                
        with col_plot2:
            st.markdown("#### K-Parameter Tuning Curve")
            # Run CV for K=1 to 20
            k_range = range(1, 21)
            cv_scores = []
            for k in k_range:
                temp_knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
                scores = cross_val_score(temp_knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
                cv_scores.append(scores.mean())
                
            fig_tune = px.line(
                x=list(k_range), y=cv_scores,
                markers=True,
                labels={"x": "Neighbors (K)", "y": "5-Fold Cross-Validation Accuracy"}
            )
            
            # Highlight chosen K
            fig_tune.add_trace(go.Scatter(
                x=[k_val], y=[cv_scores[k_val-1]],
                mode='markers',
                marker=dict(color='#EF4444', size=12, line=dict(color='#F1F5F9', width=2)),
                name='Active K'
            ))
            
            fig_tune.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font=dict(color='#F1F5F9'),
                xaxis=dict(gridcolor='#334155', linecolor='#334155', tickmode='linear', tick0=1, dtick=1),
                yaxis=dict(gridcolor='#334155', linecolor='#334155'),
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_tune, use_container_width=True)
            st.caption("This curve shows model validation accuracy for different K values. Troughs represent underfitting/overfitting areas.")
            
        # Section 3: Exploratory Data Table
        st.markdown("#### Explore Heart Dataset Sample")
        st.dataframe(df_processed.head(10), use_container_width=True)
        
    # ==========================================
    # TAB 2: PATIENT DIAGNOSIS & PROJECTION
    # ==========================================
    with tab2:
        st.markdown("### 🔮 Patient Diagnosis & PCA Cluster Mapping")
        st.write("Input a new patient's medical metrics below to predict their heart disease risk and view their location on the cluster map.")
        
        col_form, col_pca = st.columns([2, 3])
        
        with col_form:
            st.markdown("#### Patient Medical Intake Form")
            
            with st.form("patient_form"):
                col_sub1, col_sub2 = st.columns(2)
                
                with col_sub1:
                    age_input = st.slider("Age (years)", 10, 100, 52, step=1)
                    sex_display = st.selectbox("Sex", options=["Male", "Female"])
                    sex_val = 1 if sex_display == "Male" else 0
                    
                    cp_display = st.selectbox(
                        "Chest Pain Type (cp)",
                        options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]
                    )
                    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
                    cp_val = cp_map[cp_display]
                    
                    trestbps_input = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 125, step=1)
                    chol_input = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 212, step=1)
                    
                    fbs_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=["No", "Yes"])
                    fbs_val = 1 if fbs_display == "Yes" else 0
                    
                    restecg_display = st.selectbox(
                        "Resting ECG Results (restecg)",
                        options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
                    )
                    restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
                    restecg_val = restecg_map[restecg_display]
                    
                with col_sub2:
                    thalach_input = st.slider("Max Heart Rate Achieved (thalach)", 60, 220, 168, step=1)
                    exang_display = st.selectbox("Exercise Induced Angina (exang)", options=["No", "Yes"])
                    exang_val = 1 if exang_display == "Yes" else 0
                    
                    oldpeak_input = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
                    
                    slope_display = st.selectbox(
                        "Slope of Peak Exercise ST Segment",
                        options=["Upsloping", "Flat", "Downsloping"]
                    )
                    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
                    slope_val = slope_map[slope_display]
                    
                    ca_input = st.selectbox("Major Vessels (0-4) Colored by Fluoroscopy", options=[0, 1, 2, 3, 4], index=2)
                    
                    thal_display = st.selectbox(
                        "Thalassemia Type (thal)",
                        options=["Normal", "Fixed Defect", "Reversible Defect", "Null/Unknown"]
                    )
                    # Mapping to codes in dataset [3, 2, 1, 0]
                    # Let's align: Normal=3, Fixed=2, Reversible=1, Null=0
                    thal_map = {"Normal": 3, "Fixed Defect": 2, "Reversible Defect": 1, "Null/Unknown": 0}
                    thal_val = thal_map[thal_display]
                    
                submit_btn = st.form_submit_button("Run Diagnostics")
                
        with col_pca:
            st.markdown("#### Patient Cluster Projection (2D PCA)")
            st.write("This chart projects all existing patients in 2D space. Healthy patients are green, diagnosed patients are red.")
            
            # Compute PCA on X
            pca = PCA(n_components=2)
            if scale_features:
                X_all_scaled = scaler.fit_transform(X)
                coords = pca.fit_transform(X_all_scaled)
            else:
                coords = pca.fit_transform(X.values)
                
            pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
            pca_df["Status"] = y.map({0: "Normal (Low Risk)", 1: "Disease (High Risk)"}).values
            
            fig_pca = px.scatter(
                pca_df, x="PC1", y="PC2", color="Status",
                color_discrete_map={
                    "Normal (Low Risk)": "#10B981",
                    "Disease (High Risk)": "#EF4444"
                },
                labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
            )
            
            fig_pca.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.5)',
                font=dict(color='#F1F5F9'),
                xaxis=dict(gridcolor='#334155', linecolor='#334155'),
                yaxis=dict(gridcolor='#334155', linecolor='#334155'),
                legend=dict(font=dict(color='#F1F5F9'), yanchor="top", y=0.99, xanchor="left", x=0.01),
                margin=dict(t=10, b=10, l=10, r=10),
                height=420
            )
            
            if submit_btn:
                # Prepare patient row
                patient_data = np.array([[
                    age_input, sex_val, cp_val, trestbps_input, chol_input, fbs_val,
                    restecg_val, thalach_input, exang_val, oldpeak_input, slope_val,
                    ca_input, thal_val
                ]])
                
                # Scale if needed
                if scale_features:
                    patient_scaled = scaler.transform(patient_data)
                else:
                    patient_scaled = patient_data
                    
                # Predict
                pred_label = knn.predict(patient_scaled)[0]
                pred_proba = knn.predict_proba(patient_scaled)[0]
                
                # Project patient onto PCA
                patient_pca = pca.transform(patient_scaled)[0]
                
                # Add patient point to scatter plot
                fig_pca.add_trace(go.Scatter(
                    x=[patient_pca[0]], y=[patient_pca[1]],
                    mode='markers',
                    marker=dict(color='#EAB308', size=18, symbol='star', line=dict(color='#F1F5F9', width=2)),
                    name='Current Patient',
                    hovertext='Your Patient Coordinates'
                ))
                
                st.plotly_chart(fig_pca, use_container_width=True)
                
                # Render results card
                if pred_label == 1:
                    status_card = """
                    <div class="glass-card" style="border-left: 5px solid #EF4444;">
                        <span class="metric-label" style="color: #F87171;">DIAGNOSTIC STATUS</span>
                        <div class="metric-value" style="color: #EF4444; font-size: 2rem;">⚠️ HEART DISEASE DETECTED</div>
                        <p style="color: #94A3B8; margin-top: 10px; font-size: 0.9rem;">
                            Patient profile matches heart disease traits based on K-Nearest Neighbors matching. Recommended further clinical evaluation.
                        </p>
                    </div>
                    """
                else:
                    status_card = """
                    <div class="glass-card" style="border-left: 5px solid #10B981;">
                        <span class="metric-label" style="color: #34D399;">DIAGNOSTIC STATUS</span>
                        <div class="metric-value" style="color: #10B981; font-size: 2rem;">✅ NORMAL (LOW RISK)</div>
                        <p style="color: #94A3B8; margin-top: 10px; font-size: 0.9rem;">
                            Patient profile does not show indicators of high disease risk in the nearest neighbors lookup.
                        </p>
                    </div>
                    """
                st.markdown(status_card, unsafe_allow_html=True)
                
                # Probabilities progress bar
                st.write("**Model Prediction Probability:**")
                disease_prob = pred_proba[1] * 100
                st.progress(disease_prob / 100)
                st.write(f"Probability of Heart Disease: **{disease_prob:.1f}%** | Normal: **{(100 - disease_prob):.1f}%**")
                
            else:
                st.plotly_chart(fig_pca, use_container_width=True)
                st.info("💡 Fill out the Patient Form on the left and click **Run Diagnostics** to calculate the risk score and locate them on the PCA cluster graph.")
else:
    st.error("Please ensure 'heart.csv' is present in your project directory!")
