# The AI pay penalty that isn't

**Question:** In the 2025 Stack Overflow survey, daily AI users report *lower* pay than non-users. Is that real?

**Answer:** No. It's a composition effect. Daily AI use is concentrated in lower-paying countries (r = −0.73
between a country's median pay and its share of daily users) and among junior developers.

| Model (log pay, robust SE) | Daily AI users vs non-users |
|---|---|
| Raw, all countries pooled | **−22.0%** (95% CI −25.1 to −18.8) |
| + country fixed effects | **+6.1%** (+3.2 to +9.0) |
| + experience | +7.4% (+4.6 to +10.2) |
| + company size, remote, education, IC/manager | **+6.8%** (+4.1 to +9.5) |

- LightGBM (5-fold CV × 3 seeds): adding the four AI columns lowers RMSE from 0.6373 to 0.6341, a small gain. Country (64%) and experience (16%) dominate.
- India (n = 739): +23.6% for daily users after controls, but with a wide CI (+6% to +44%). Median pay goes $7k → $14k → $28k → $44k across 0–2 / 3–5 / 6–10 / 11–15 years of experience.

This is association, not causation (self-selection, survey bias, cross-sectional data).

- `build_notebook.py` generates `dev-pay-2025.ipynb` (cells kept as reviewable Python)
- `dev-pay-2025.ipynb` is executed, with outputs
- Data: [Stack Overflow Developer Survey 2025](https://survey.stackoverflow.co/2025) (ODbL 1.0), mirrored on Kaggle as `aliaslam25/stack-overflow-developer-survey-2025`. It isn't committed here.

Reproduce:
```
python -m venv .venv && .venv/Scripts/pip install pandas numpy scikit-learn statsmodels lightgbm matplotlib nbformat nbconvert ipykernel
kaggle datasets download aliaslam25/stack-overflow-developer-survey-2025 -p data --unzip
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace dev-pay-2025.ipynb
```
