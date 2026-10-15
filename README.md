# Recruiter Feedback Intelligence (prototype)

Independent, synthetic-data demo: recruiter rejection feedback → evidence-tiered hypotheses → recruiter-approved changes → testing on a separate holdout batch.

This project now includes a FastAPI web entry point for hosting environments that expect a top-level application object in `app.py`:

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

The app exports both `app` and `application`. The browser dashboard is available at `/`, with `/health`, `/api/observations`, `/api/evaluate`, and `/api/candidates` endpoints.

The original Streamlit dashboard is preserved in `streamlit_app.py` and can be run locally with:

```bash
streamlit run streamlit_app.py
```

## Data and safeguards
- `baseline_batch.csv`: 200 synthetic candidate records with recruiter decisions.
- `holdout_batch.csv`: 120 separate synthetic records with simulated relevance labels.
- `generate_data.py`: regenerate CSVs (seed 7; run from this folder).
- No real candidate data is included. Planted patterns and simulated labels demonstrate the method, not real-world performance.
- Rejection labels are not ground truth. Keep recruiter approval in the loop, avoid protected traits, audit for disparate impact, and follow data-protection requirements before real use.
