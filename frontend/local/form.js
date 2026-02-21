import { useState } from "react";

const INITIAL = {
  age:                      "",
  bmi:                      "",
  glucose:                  "",
  blood_pressure:           "",
  insulin:                  "",
  hypertension:             "0",
  pregnancies:              "",
  DiabetesPedigreeFunction: "",
};
//some of these are not required because they are hard to find
const FIELDS = [
  {//this is age
    key: "age", label: "Age",
    unit: "years", min: 1, max: 120, step: 1,
    required: true,
    hint: "Your current age",
  },
  {//this is bmi
    key: "bmi", label: "BMI",
    unit: "kg/m²", min: 10, max: 80, step: 0.1,
    required: true,
    hint: "Body Mass Index — weight(kg) ÷ height(m)²",
  },
  {//this is glucose
    key: "glucose", label: "Blood Glucose",
    unit: "mg/dL", min: 40, max: 500, step: 1,
    required: true,
    hint: "Fasting blood glucose level",
  },
  {//this is blood pressure
    key: "blood_pressure", label: "Blood Pressure",
    unit: "mmHg", min: 40, max: 200, step: 1,
    required: false,
    hint: "Systolic blood pressure (optional)",
  },
  {// this is insulin
    key: "insulin", label: "Insulin Level",
    unit: "µU/mL", min: 0, max: 900, step: 1,
    required: false,
    hint: "Fasting insulin level (optional)",
  },
  {// this is pregnancies
    key: "pregnancies", label: "Pregnancies",
    unit: "", min: 0, max: 20, step: 1,
    required: false,
    hint: "Number of times pregnant (optional)",
  },
  { // this is diabetes pedigree function
    key: "DiabetesPedigreeFunction", label: "Diabetes Pedigree Function",
    unit: "", min: 0, max: 2.5, step: 0.01,
    required: false,
    hint: "A measure of genetic risk of diabetes (optional)",
  },
];

const SELECT_FIELDS = [
  {
    key: "gender", label: "Biological Sex",
    options: [
      { value: "female", label: "Female" },
      { value: "male",   label: "Male" },
      { value: "other",  label: "Prefer not to say" },
    ],
  },
  {
    key: "hypertension", label: "Hypertension",
    options: [
      { value: "0", label: "No" },
      { value: "1", label: "Yes — diagnosed" },
    ],
  },
];

export default function DiabetesForm({ onSubmit, loading, error }) {
  const [form, setForm]       = useState(INITIAL);
  const [touched, setTouched] = useState({});

  const set   = (key, val) => setForm(f => ({ ...f, [key]: val }));
  const touch = (key)      => setTouched(t => ({ ...t, [key]: true }));

  const isInvalid = (f) => f.required && touched[f.key] && form[f.key] === "";

  const canSubmit = FIELDS
    .filter(f => f.required)
    .every(f => form[f.key] !== "");

  const handleSubmit = () => {
    const allTouched = FIELDS.reduce((acc, f) => ({ ...acc, [f.key]: true }), {});
    setTouched(allTouched);
    if (!canSubmit) return;

    const payload = { ...form };
    ["age", "bmi", "glucose", "blood_pressure", "insulin", "pregnancies", "DiabetesPedigreeFunction"].forEach(k => {
      if (payload[k] !== "") payload[k] = parseFloat(payload[k]);
    });
    payload.hypertension = parseInt(payload.hypertension);
    onSubmit(payload);
  };
//the form is pretty self explanatory, it just collects the data and sends it to the backend for processing. It also has some basic validation and error handling. The form fields are defined in the FIELDS array, which makes it easy to add or remove fields in the future.
  return (
    <div className="form-card">
      <div className="form-card-header">
        <h1 className="form-title">Risk Assessment</h1>
        <p className="form-subtitle">
          Enter your health metrics below. Fields marked{" "}
          <span className="req-star">*</span> are required.
        </p>
      </div>

      <div className="form-body">
        <div className="fields-grid">
          {FIELDS.map(f => (
            <div key={f.key} className={`field-group ${isInvalid(f) ? "field-error" : ""}`}>
              <label className="field-label">
                {f.label}
                {f.required && <span className="req-star"> *</span>}
              </label>
              <div className="input-wrap">
                <input
                  type="number"
                  className="field-input"
                  value={form[f.key]}
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  placeholder={`e.g. ${f.min + Math.round((f.max - f.min) / 3)}`}
                  onChange={e => set(f.key, e.target.value)}
                  onBlur={() => touch(f.key)}
                />
                <span className="input-unit">{f.unit}</span>
              </div>
              <p className="field-hint">
                {isInvalid(f) ? "This field is required." : f.hint}
              </p>
            </div>
          ))}
        </div>

        <div className="selects-grid">
          {SELECT_FIELDS.map(f => (
            <div key={f.key} className="field-group">
              <label className="field-label">{f.label}</label>
              <select
                className="field-select"
                value={form[f.key]}
                onChange={e => set(f.key, e.target.value)}
              >
                {f.options.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          ))}
        </div>

        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠</span>
            <span>{error}</span>
          </div>
        )}

        <button
          className={`submit-btn ${loading ? "loading" : ""}`}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? (
            <span className="btn-loading">
              <span className="spinner" />
              Analysing...
            </span>
          ) : (
            "Assess My Risk →"
          )}
        </button>
      </div>
    </div>
  );
}