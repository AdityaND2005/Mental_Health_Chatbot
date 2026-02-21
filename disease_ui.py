# ui/disease_ui.py

import streamlit as st
from disease_engine import (
    load_artifacts,
    predict_disease,
    generate_lime_explanation,
    generate_lime_conclusion,
    generate_pdf_report
)

# -----------------------------
# Load Artifacts (Cached)
# -----------------------------
@st.cache_resource
def initialize():
    return load_artifacts()

def run_disease_ui():
    model, le, sbert, severity, desc, prec = initialize()


    # -----------------------------
    # Streamlit UI
    # -----------------------------
    st.title("🩺 AI-Powered Disease Prediction with Explainable AI")

    st.subheader("Patient Information")

    name = st.text_input("Enter your Name:")
    age = st.number_input("Enter your Age:", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Select your Gender:", ["Male", "Female", "Other"])

    st.subheader("Symptom Input")

    symptom_input = st.text_area(
        "Describe your symptoms (separated by spaces):"
    )

    if st.button("🔍 Predict Disease"):

        if not symptom_input.strip():
            st.warning("Please enter symptoms to proceed.")

        else:
            with st.spinner("Analyzing symptoms and predicting..."):

                # -----------------------------
                # Prediction
                # -----------------------------
                prediction_result = predict_disease(
                    symptom_input,
                    model,
                    le,
                    sbert,
                    severity,
                    desc,
                    prec
                )

                details = prediction_result['Top_Disease_Details']

                st.success(
                    f"**Predicted Disease:** {details['Predicted_Disease']}"
                )

                st.write(
                    f"**Risk Level:** {details['Risk_Level']} "
                    f"(Severity Score: {details['Avg_Severity_Score']})"
                )

                st.write(f"**Description:** {details['Description']}")

                st.write("### Recommended Precautions:")
                for i, p in enumerate(details['Precautions'], 1):
                    st.write(f"{i}. {p}")

                # -----------------------------
                # LIME Explainability
                # -----------------------------
                st.subheader("🧠 Explainable AI (LIME) Analysis")

                lime_fig, explanation = generate_lime_explanation(
                    symptom_input,
                    model,
                    le,
                    sbert
                )

                st.pyplot(lime_fig)

                lime_conclusion = generate_lime_conclusion(
                    explanation,
                    details['Predicted_Disease'],
                    le
                )

                st.markdown(f"**Explanation:** {lime_conclusion}")

                # -----------------------------
                # Disclaimer
                # -----------------------------
                st.warning(
                    "Disclaimer: This is an AI-generated prediction "
                    "intended for informational purposes only and is "
                    "not a substitute for professional medical advice."
                )

                # -----------------------------
                # PDF Generation
                # -----------------------------
                pdf_buffer = generate_pdf_report(
                    name,
                    age,
                    gender,
                    prediction_result,
                    lime_conclusion,
                    lime_fig
                )

                st.download_button(
                    label="📄 Download Full Report as PDF",
                    data=pdf_buffer,
                    file_name=f"{name}_AI_Disease_Report.pdf",
                    mime="application/pdf"
                )