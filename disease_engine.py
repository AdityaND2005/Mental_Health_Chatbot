# core/disease_core.py

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import lime.lime_text
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage


# -----------------------------
# Load Models and Data
# -----------------------------
def load_artifacts():

    with open('model_LR.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    df_proc = pd.read_csv('processed_data.csv')
    df_desc = pd.read_csv('symptom_Description.csv')
    df_prec = pd.read_csv('symptom_precaution.csv')

    severity = df_proc.drop_duplicates('Disease')\
        .set_index('Disease')['Avg_Severity'].to_dict()

    desc = df_desc.set_index('Disease')['Description'].to_dict()

    df_prec_melt = df_prec.melt(
        id_vars=['Disease'],
        value_vars=['Precaution_1', 'Precaution_2',
                    'Precaution_3', 'Precaution_4'],
        var_name='Precaution_Type',
        value_name='Precaution'
    )

    prec = df_prec_melt.groupby('Disease')['Precaution']\
        .apply(lambda x: [p for p in x if pd.notna(p)])\
        .to_dict()

    return model, le, sbert, severity, desc, prec


# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(symptoms_text,
                    model,
                    label_encoder,
                    sentence_bert_model,
                    severity_lookup,
                    description_lookup,
                    precaution_lookup):

    cleaned_text = symptoms_text.strip().lower()

    text_vector = sentence_bert_model.encode([cleaned_text])

    probabilities = model.predict_proba(text_vector)[0]

    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    top3_probabilities = probabilities[top3_indices]

    top_predictions = [
        {'Disease': disease}
        for disease, prob in zip(top3_diseases, top3_probabilities)
    ]

    top_disease = top3_diseases[0]
    avg_severity = severity_lookup.get(top_disease, 0)

    if avg_severity >= 5:
        risk_level = "High Risk"
    elif 3 <= avg_severity < 5:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    result = {
        'Top_Predictions': top_predictions,
        'Top_Disease_Details': {
            'Predicted_Disease': top_disease,
            'Description': description_lookup.get(
                top_disease,
                "No description available."
            ),
            'Precautions': precaution_lookup.get(top_disease, []),
            'Risk_Level': risk_level,
            'Avg_Severity_Score': avg_severity
        }
    }

    return result


# -----------------------------
# Explainable AI (LIME)
# -----------------------------
def generate_lime_explanation(symptoms_text,
                              model,
                              label_encoder,
                              sentence_bert_model):

    def sbert_predict_proba(texts):
        embeddings = sentence_bert_model.encode(
            texts,
            show_progress_bar=False
        )
        return model.predict_proba(embeddings)

    explainer = lime.lime_text.LimeTextExplainer(
        class_names=label_encoder.classes_
    )

    predicted_disease = predict_disease(
        symptoms_text,
        model,
        label_encoder,
        sentence_bert_model,
        {},
        {},
        {}
    )['Top_Disease_Details']['Predicted_Disease']

    label_to_explain = label_encoder.transform(
        [predicted_disease]
    )[0]

    explanation = explainer.explain_instance(
        symptoms_text,
        sbert_predict_proba,
        num_features=10,
        labels=[label_to_explain]
    )

    fig = explanation.as_pyplot_figure(label=label_to_explain)
    fig.tight_layout()

    return fig, explanation


# -----------------------------
# LIME Conclusion Generator
# -----------------------------
def generate_lime_conclusion(explanation,
                             predicted_disease,
                             label_encoder):

    label_index = label_encoder.transform(
        [predicted_disease]
    )[0]

    exp_list = explanation.as_list(label=label_index)

    supporting_symptoms = [
        symptom for symptom, weight in exp_list
        if weight > 0
    ]

    if not supporting_symptoms:
        return (
            "The model's prediction was based on a "
            "combination of your symptoms."
        )

    conclusion = (
        f"The prediction of **{predicted_disease}** "
        f"was primarily influenced by "
    )

    if len(supporting_symptoms) > 1:
        top_symptoms_str = ", ".join(
            [f"'{s}'" for s in supporting_symptoms[:-1]]
        ) + f" and '{supporting_symptoms[-1]}'"

        conclusion += f"the presence of {top_symptoms_str}."
    else:
        conclusion += (
            f"the presence of '{supporting_symptoms[0]}'."
        )

    return conclusion


# -----------------------------
# PDF Report Generator
# -----------------------------
def generate_pdf_report(name,
                        age,
                        gender,
                        prediction_result,
                        lime_conclusion,
                        lime_fig):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "<b>AI Disease Prediction Report</b>",
        styles['Title']
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Name:</b> {name}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Age:</b> {age}",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Gender:</b> {gender}",
        styles['Normal']
    ))
    story.append(Spacer(1, 12))

    details = prediction_result['Top_Disease_Details']

    story.append(Paragraph(
        f"<b>Primary Diagnosis:</b> "
        f"{details['Predicted_Disease']}",
        styles['Heading2']
    ))

    story.append(Paragraph(
        f"<b>Risk Level:</b> {details['Risk_Level']} "
        f"(Severity Score: {details['Avg_Severity_Score']})",
        styles['Normal']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        f"<b>Description:</b> "
        f"{details['Description']}",
        styles['Normal']
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "<b>Recommended Precautions:</b>",
        styles['Heading3']
    ))

    for p in details['Precautions']:
        story.append(Paragraph(f"- {p}", styles['Normal']))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Explainable AI Insight:</b>",
        styles['Heading2']
    ))

    story.append(Paragraph(
        lime_conclusion,
        styles['Normal']
    ))

    # Save LIME figure
    lime_img = BytesIO()
    lime_fig.savefig(lime_img, format='png')
    lime_img.seek(0)

    story.append(Spacer(1, 12))
    story.append(RLImage(lime_img, width=400, height=250))

    story.append(Spacer(1, 10))

    disclaimer = (
        "Disclaimer: This is an AI-generated prediction "
        "intended for informational purposes only and is not a "
        "substitute for professional medical advice, diagnosis, "
        "or treatment. Please consult a qualified healthcare "
        "provider with any questions you may have regarding "
        "a medical condition."
    )

    story.append(Spacer(1, 10))
    story.append(Paragraph(disclaimer, styles['Italic']))

    doc.build(story)
    buffer.seek(0)

    return buffer