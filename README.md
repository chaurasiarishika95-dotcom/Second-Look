# Recruiter Feedback Intelligence (prototype)
Independent, synthetic-data demo: rejection feedback → evidence-tiered hypotheses → recruiter-approved changes → test on a separate holdout batch.
Not affiliated with Naukri/AI REX. No real candidate data.

    pip install -r requirements.txt
    python generate_data.py   # optional: regenerates the CSVs
    streamlit run app.py

Files: `app.py`, `generate_data.py`, `baseline_batch.csv` (200 candidates with decisions), `holdout_batch.csv` (120 candidates with simulated relevance labels).
Limits: planted patterns and simulated labels demonstrate the method, not real-world performance.
