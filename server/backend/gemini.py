"""
LLM-powered clinical analysis for CARP diagnosis results.

Provider chain (tried in order):
  1. Groq Llama 3.3 70B — GROQ_API_KEY in .env
  2. Gemini 2.5 Flash   — GEMINI_API_KEY in .env

If both fail the caller receives None and no analysis card is shown.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ── Shared prompt builder ─────────────────────────────────────────────────────


def _build_prompt(
    age: float,
    bmi: float,
    glucose: float,
    prediction: int,
    confidence: float,
    feature_contributions: dict | None = None,
) -> str:
    risk_label = (
        "elevated diabetes risk" if prediction == 1 else "no elevated diabetes risk"
    )
    confidence_pct = round(confidence * 100, 1)

    bmi_category = (
        "underweight (BMI < 18.5)"
        if bmi < 18.5
        else "normal weight (BMI 18.5–24.9)"
        if bmi < 25
        else "overweight (BMI 25–29.9)"
        if bmi < 30
        else "obese class I (BMI 30–34.9)"
        if bmi < 35
        else "obese class II (BMI 35–39.9)"
        if bmi < 40
        else "obese class III (BMI ≥ 40)"
    )
    glucose_category = (
        "normal fasting glucose (< 100 mg/dL)"
        if glucose < 100
        else "impaired fasting glucose / pre-diabetic range (100–125 mg/dL)"
        if glucose < 126
        else "diabetic range fasting glucose (≥ 126 mg/dL)"
    )
    age_note = (
        "young adult" if age < 35 else "middle-aged" if age < 55 else "older adult"
    )

    fc = feature_contributions or {}
    fc_age = fc.get("Age", 0)
    fc_bmi = fc.get("BMI", 0)
    fc_glucose = fc.get("Glucose", 0)

    return f"""You are a clinical decision-support assistant helping a medical professional interpret an AI-generated diabetes risk screening result. Be specific, evidence-based, and actionable. Reference the actual numbers — do not give generic advice.

Patient vitals:
- Age: {age} years ({age_note})
- BMI: {bmi} kg/m² — {bmi_category}
- Fasting Blood Glucose: {glucose} mg/dL — {glucose_category}

AI screening model result:
- Prediction: {risk_label}
- Model confidence: {confidence_pct}%
- Feature contributions (SHAP-style; positive = increases risk, negative = decreases risk): Age {fc_age}, BMI {fc_bmi}, Glucose {fc_glucose}. The factor with the largest absolute contribution drove the result most.

Write a structured clinical note with the following four sections. Label each section clearly. Use plain text, no markdown, no bullet symbols.

VITAL INTERPRETATION
In 2-3 sentences, explain what this specific combination of age, BMI, and glucose means for this patient's metabolic health. Reference the actual values and thresholds (e.g. ADA criteria, WHO BMI classifications).

RISK FACTORS
Identify which of the submitted values are concerning and which are within normal range. Quantify how far outside normal any values are. Note any compounding interactions between the values (e.g. high BMI combined with high glucose). State which factor (Age, BMI, or Glucose) contributed most to the model's risk score using the feature contributions above.

RECOMMENDED NEXT STEPS
Give 3-5 specific, actionable recommendations tailored to this patient's numbers. Include: which follow-up tests are indicated (e.g. HbA1c, OGTT, fasting lipids), specific lifestyle targets appropriate to their BMI and glucose level, and whether referral to an endocrinologist or dietitian is warranted given the values.

MODEL CONFIDENCE NOTE
Explain what the {confidence_pct}% confidence means in practical terms for this result. If confidence is low (under 60%) or the prediction is borderline, flag that the clinician should weight clinical judgement heavily. Note that the model uses only three features and may miss context available in a full clinical assessment.

Keep the total response under 250 words."""


# ── Provider 1: Gemini ────────────────────────────────────────────────────────

_gemini_client = None


def _try_gemini(prompt: str) -> str:
    global _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=api_key)

    from google.genai import types

    response = _gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
        ),
    )

    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    text = text.strip()
    logger.info("Gemini response:\n%s", text)
    return text


# ── Provider 2: Groq ──────────────────────────────────────────────────────────

_groq_client = None


def _try_groq(prompt: str) -> str:
    global _groq_client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=api_key)

    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2048,
    )

    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned an empty response.")

    text = text.strip()
    logger.info("Groq response:\n%s", text)
    return text


# ── Public interface ──────────────────────────────────────────────────────────


def analyse_diagnosis(
    age: float,
    bmi: float,
    glucose: float,
    prediction: int,
    confidence: float,
    feature_contributions: dict | None = None,
) -> tuple[str, str]:
    """
    Generate a clinical analysis, trying Groq first then Gemini as fallback.

    feature_contributions: optional dict with keys Age, BMI, Glucose (SHAP-style values).
    Returns (analysis_text, provider_name), or raises if both providers fail.
    """
    prompt = _build_prompt(
        age, bmi, glucose, prediction, confidence,
        feature_contributions=feature_contributions,
    )

    try:
        result = _try_groq(prompt)
        logger.info("Clinical analysis generated via Groq.")
        return result, "Groq (Llama 3.3 70B)"
    except Exception as groq_exc:
        logger.warning("Groq failed (%s), trying Gemini fallback.", groq_exc)

    try:
        result = _try_gemini(prompt)
        logger.info("Clinical analysis generated via Gemini fallback.")
        return result, "Gemini 2.5 Flash"
    except Exception as gemini_exc:
        logger.warning("Gemini fallback also failed: %s", gemini_exc)
        raise RuntimeError(
            f"All LLM providers failed. Groq: {groq_exc}. Gemini: {gemini_exc}."
        )
