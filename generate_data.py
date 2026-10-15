"""Generates synthetic data: a baseline campaign batch (with recruiter decisions)
and a separate holdout batch (with simulated recruiter-reviewed relevance labels)."""
import numpy as np, pandas as pd
DOM = ["Banking","Healthcare","IT Services","E-commerce","Developer Tools","SaaS","FinTech"]
DW = [.22,.06,.24,.14,.12,.12,.10]  # Healthcare deliberately rare -> small-sample trap
OTHER = ["SQL","Microservices","AWS","Docker","Kafka","Kubernetes","System Design","REST APIs"]
LOC = ["Remote","Delhi NCR","Bengaluru","Hyderabad","Pune"]
JR = {"R1":3.0,"R2":3.5,"R3":4.5}  # recruiters apply different "too junior" cutoffs

def profiles(n, rng, prefix):
    rows = []
    for i in range(n):
        dom = rng.choice(DOM, p=DW)
        java = rng.random() < .85; spring = java and rng.random() < .8
        sk = (["Java"] if java else []) + (["Spring Boot"] if spring else []) + list(rng.choice(OTHER, rng.integers(2,5), replace=False))
        rows.append(dict(candidate_id=f"{prefix}-{i+1:03d}", recruiter=rng.choice(list(JR)),
            experience_years=round(float(np.clip(rng.normal(5,2.2),.5,10)),1), domain=dom,
            skills=", ".join(sk), location=rng.choice(LOC),
            notice_period_days=int(rng.choice([15,30,45,60,90], p=[.1,.15,.2,.3,.25] if dom!="IT Services" else [.05,.1,.15,.3,.4]))))
    return pd.DataFrame(rows)

def decide(r, rng):
    if r.experience_years < JR[r.recruiter]: return "Too junior"
    if "Java" not in r.skills or "Spring Boot" not in r.skills: return "Missing critical skills"
    if r.notice_period_days >= 90 and rng.random() < .8: return "Notice period too long"
    if r.domain == "Healthcare" and rng.random() < .6: return "Wrong domain"
    if 4 <= r.experience_years < 7 and rng.random() < .12: return "Insufficient production ownership"
    if rng.random() < .03: return "Salary mismatch"
    if rng.random() < .04: return "Location mismatch"
    return None

rng = np.random.default_rng(7)
b = profiles(200, rng, "SYN-B")
b["rejection_reason"] = [decide(r, rng) for r in b.itertuples()]
b["decision"] = np.where(b.rejection_reason.isna(), "Shortlisted", "Rejected")
b["rejection_reason"] = b.rejection_reason.fillna("")
b.to_csv("baseline_batch.csv", index=False)

h = profiles(120, rng, "SYN-H")
rel = (h.experience_years >= 4) & h.skills.str.contains("Java") & h.skills.str.contains("Spring Boot") & (h.notice_period_days <= 60)
flip = rng.random(len(h)) < .08  # label noise: reviewers are not perfect
h["recruiter_relevant"] = (rel ^ flip).astype(int)
h.drop(columns="recruiter").to_csv("holdout_batch.csv", index=False)
print(b.decision.value_counts().to_dict(), b.rejection_reason.value_counts().to_dict(), h.recruiter_relevant.mean().round(2))
