# Explainable AI for Diabetic Retinopathy Screening

An AI-powered screening tool for Diabetic Retinopathy (DR) that doesn't just predict — it explains. Built to address the gap in automated eye-care access in rural India, where ophthalmologist availability is low and manual grading is slow.

**Live Demo:** https://9cahuxqggtpla6h5zjtsv7.streamlit.app/
**Repo:** https://github.com/Gambhiryadav1902/dr_screening_app-/tree/master

---

## Problem

Diabetic retinopathy is a leading cause of preventable blindness. In rural India, access to ophthalmologists for timely screening is limited, and manual grading of retinal images is slow and expertise-dependent. This project builds an automated, interpretable screening pipeline that can run on a lightweight web app.

## Dataset

- **APTOS 2019 Blindness Detection** (Kaggle competition)
- Retinal fundus images labeled into 5 severity classes:
  - 0 — No DR
  - 1 — Mild
  - 2 — Moderate
  - 3 — Severe
  - 4 — Proliferative DR
- Known challenge: significant class imbalance across severity levels.

## Approach

**1. Baseline model**
- Transfer learning with **EfficientNetB0** (frozen backbone), custom classification head for 5-class output
- Framework: TensorFlow/Keras, trained on Kaggle Notebooks (GPU)
- Result: **68% validation accuracy**

**2. Identified gap**
Standard categorical loss functions treat all misclassifications equally — but DR severity is ordinal. Misclassifying Mild as Moderate is a far smaller error than misclassifying Mild as Proliferative. Most literature on DR classification doesn't account for this.

**3. Custom distance-weighted loss function**
Designed a loss function that penalizes misclassifications in proportion to how far apart the predicted and true class are on the severity scale.

**4. Result after fix**
- Validation accuracy: 68% → **74%**
- Moderate-class recall: 0.30 → **0.63**
- Trade-off: Proliferative-class recall dropped (0.59 → 0.27) — flagged as a direction for future work (class-weighted sampling, focal loss combination)

**5. Explainability**
Integrated **Grad-CAM** to visualize which regions of the retinal image drove each prediction — critical for clinician trust and real-world medical deployment, not just raw accuracy.

**6. Deployment**
Built and deployed a **Streamlit** demo app: upload a retinal image → get a severity prediction + Grad-CAM heatmap overlay.

## Tech Stack

- Python, TensorFlow/Keras
- EfficientNetB0 (transfer learning)
- Grad-CAM (explainability)
- Streamlit (demo/deployment)
- Kaggle Notebooks (training environment)

## Future Work

- Improve Proliferative-class recall without losing Moderate-class gains
- Validate on external/held-out datasets beyond APTOS
- Explore mobile-friendly deployment for field screening use

## Running Locally

```bash
git clone https://github.com/Gambhiryadav1902/dr_screening_app-.git
cd dr_screening_app-
pip install -r requirements.txt
streamlit run app.py
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
