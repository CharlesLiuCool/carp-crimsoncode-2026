export const EXPECTED_COLUMNS = [
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
  "Outcome",
];

export const DIAGNOSIS_COLUMNS = [
  "Pregnancies",
  "Glucose",
  "BloodPressure",
  "Insulin",
  "BMI",
  "DiabetesPedigreeFunction",
  "Age",
];

export function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map((h) => h.trim());
  const rows = lines
    .slice(1)
    .map((line) => line.split(",").map((v) => v.trim()));
  return { headers, rows };
}

export function validateColumns(headers, expected) {
  const missing = expected.filter((col) => !headers.includes(col));
  const extra = headers.filter((col) => !expected.includes(col));
  return { missing, extra };
}
