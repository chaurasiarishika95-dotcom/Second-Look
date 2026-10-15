import streamlit as st, pandas as pd, numpy as np
from pathlib import Path
D = Path(__file__).parent
st.set_page_config(page_title="Recruiter Feedback Intelligence", page_icon="🔎", layout="wide")
base = pd.read_csv(D/"baseline_batch.csv").fillna(""); hold = pd.read_csv(D/"holdout_batch.csv")
rej = base[base.decision == "Rejected"]
st.title("🔎 Recruiter Feedback Intelligence")
st.caption("Independent prototype · 100% synthetic data · Not affiliated with Naukri or AI REX")
st.markdown("Turns recruiter rejections into **evidence-backed sourcing experiments**, and keeps what the data *shows* separate from what the AI *guesses*.")

def wilson(k, n, z=1.96):
    if n == 0: return 0, 0
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); m = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c-m)/d, (c+m)/d
def tier(n, lo, hi):
    if n < 8: return "🔴 Too few records to act on"
    return "🟡 Early signal" if hi-lo > .25 else "🟢 Consistent pattern"

TXT = {  # reason -> (hypothesis, suggested change, how to check)
 "Too junior": ("The seniority criteria may not capture the level recruiters expect.", "Raise the experience floor and/or add explicit ownership requirements.", "Share of 'too junior' rejections in the next fresh batch."),
 "Missing critical skills": ("Must-have skills may not be enforced in the search.", "Require Java and Spring Boot evidence in the profile.", "Share of 'missing skills' rejections in the next fresh batch."),
 "Notice period too long": ("Availability may not be filtered although it is a real constraint.", "Cap notice period (only if it is a true business constraint).", "Notice-period rejections, and whether strong candidates are lost."),
 "Wrong domain": ("Domain may matter less than assumed, or the sample is simply small.", "Clarify whether domain is essential or learnable before filtering.", "Collect more feedback first."),
}
st.header("1 · Observations → hypotheses (recruiter approves each)")
st.write(f"**{len(base)} candidates** in the baseline batch: **{len(rej)} rejected**, {len(base)-len(rej)} shortlisted ({len(rej)/len(base):.0%} rejection rate).")
accepted = {}
for reason, k in rej.rejection_reason.value_counts().items():
    lo, hi = wilson(k, len(rej)); t = tier(k, lo, hi); sub = rej[rej.rejection_reason == reason]
    with st.container(border=True):
        st.markdown(f"**{reason}** · {t}")
        by = sub.recruiter.value_counts().to_dict(); tot = rej.recruiter.value_counts()
        rates = {r: (by.get(r, 0)/tot[r]) for r in tot.index if tot[r] >= 10}
        st.markdown(f"**Observation:** {k} of {len(rej)} rejections ({k/len(rej):.0%}; 95% CI {lo:.0%}–{hi:.0%}). "
                    f"Most common domain: {sub.domain.mode()[0]} ({(sub.domain == sub.domain.mode()[0]).sum()} of {k}).")
        if rates and max(rates.values()) - min(rates.values()) > .15:
            st.warning("Recruiters disagree: " + ", ".join(f"{r} {v:.0%}" for r, v in sorted(rates.items())) + " of their rejections use this reason. The criteria may be ambiguous, or recruiters may apply different standards.")
        if reason in TXT:
            h, s, c = TXT[reason]
            st.markdown(f"**Hypothesis (unproven):** {h}  \n**Suggested change:** {s}  \n**How to check:** {c}")
            if t.startswith("🔴"): st.caption("Not enough records to suggest a change. Gather more feedback.")
            else: accepted[reason] = st.checkbox("Approve this change for the simulation", key=reason)
        else:
            st.caption("Too rare to suggest a change." if k < 8 else "No template yet for this reason.")
st.caption("Rejection tags are one recruiter's label, not ground truth, and can't prove the job description or search caused a rejection.")

st.header("2 · Test approved changes on a fresh holdout batch")
c = st.columns(3)
exp_floor = c[0].slider("Experience floor (if approved)", 3.0, 7.0, 4.0, .5)
notice_cap = c[1].slider("Notice cap, days (if approved)", 30, 90, 60, 15)
dom_filter = c[2].selectbox("Domain filter (illustrative, not suggested above)", ["None"] + sorted(hold.domain.unique()))
def apply(df, on):
    m = df.skills.str.contains("Java")
    if on.get("Too junior"): m &= df.experience_years >= exp_floor
    if on.get("Missing critical skills"): m &= df.skills.str.contains("Spring Boot")
    if on.get("Notice period too long"): m &= df.notice_period_days <= notice_cap
    if dom_filter != "None": m &= df.domain == dom_filter
    return df[m]
def score(df): return df.experience_years.clip(upper=10)*4 + df.skills.str.contains("Spring Boot")*30 - df.notice_period_days/6
def evaluate(on, label):
    kept = apply(hold, on); top = kept.assign(s=score(kept)).sort_values("s", ascending=False).head(20)
    rel = hold.recruiter_relevant.sum(); lost = hold[(hold.recruiter_relevant == 1) & ~hold.candidate_id.isin(kept.candidate_id)]
    return {"Setup": label, "Profiles passing": len(kept), "Precision@20": f"{top.recruiter_relevant.mean():.0%}" if len(top) else "—",
            "Relevant candidates excluded": f"{len(lost)} of {rel}"}
res = pd.DataFrame([evaluate({}, "Baseline (Java only)"), evaluate(accepted, "With approved changes")])
st.dataframe(res, hide_index=True, width="stretch")
st.info("The holdout batch was never used to find the patterns. 'Relevant' is a simulated recruiter-reviewed label with ~8% noise, so these numbers only demonstrate the method. A tighter filter can raise precision while excluding good candidates; watch both columns.")

with st.expander("Records and safeguards"):
    st.dataframe(base, hide_index=True, width="stretch")
    st.markdown("- All data is synthetic and generated by `generate_data.py` (seed 7), with planted patterns: one strong (seniority), one recruiter disagreement, one small-sample trap (Healthcare), one rare reason.\n- Evidence tiers use counts and Wilson intervals, not a validated model.\n- Never use protected traits. Keep recruiter approval in the loop, audit for disparate impact and follow data-protection rules before real use.")
