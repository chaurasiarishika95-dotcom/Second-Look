"""Recruiter Feedback Intelligence web app.

This FastAPI entry point exports ``app`` (and ``application``) for hosts that
expect a top-level ASGI application. The original Streamlit version is kept in
streamlit_app.py and can be run with ``streamlit run streamlit_app.py``.
"""
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
base = pd.read_csv(BASE_DIR / "baseline_batch.csv").fillna("")
hold = pd.read_csv(BASE_DIR / "holdout_batch.csv").fillna("")
rej = base[base["decision"] == "Rejected"].copy()

app = FastAPI(title="Recruiter Feedback Intelligence", version="2.0.0")
# Some deployment adapters look for the WSGI/ASGI alias `application`.
application = app

TXT = {
    "Too junior": {
        "hypothesis": "The seniority criteria may not capture the level recruiters expect.",
        "change": "Raise the experience floor and/or add explicit ownership requirements.",
        "check": "Share of 'too junior' rejections in the next fresh batch.",
        "key": "experience_floor",
    },
    "Missing critical skills": {
        "hypothesis": "Must-have skills may not be enforced consistently in the search.",
        "change": "Require Java and Spring Boot evidence in the profile.",
        "check": "Share of missing-skills rejections in the next fresh batch.",
        "key": "spring_boot",
    },
    "Notice period too long": {
        "hypothesis": "Availability may not be filtered although it is a real constraint.",
        "change": "Set a notice-period cap only if it is a genuine business constraint.",
        "check": "Notice-period rejections and whether strong candidates are lost.",
        "key": "notice_cap",
    },
    "Wrong domain": {
        "hypothesis": "Domain may matter less than assumed, or the sample may be too small.",
        "change": "Clarify whether domain is essential or learnable before filtering.",
        "check": "Collect more feedback and validate with another batch.",
        "key": None,
    },
}


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / d), min(1.0, (center + margin) / d)


def evidence_tier(n: int, lo: float, hi: float) -> str:
    if n < 8:
        return "Too few records to act on"
    return "Early signal" if hi - lo > 0.25 else "More consistent pattern"


def clean_text(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def observations():
    total_rej = len(rej)
    data = []
    counts = rej["rejection_reason"].value_counts()
    for reason, count in counts.items():
        lo, hi = wilson(int(count), total_rej)
        subset = rej[rej["rejection_reason"] == reason]
        mode = subset["domain"].mode()
        dom = mode.iloc[0] if not mode.empty else "Unknown"
        dom_count = int((subset["domain"] == dom).sum())
        data.append({
            "reason": str(reason), "count": int(count),
            "share": float(count / total_rej) if total_rej else 0,
            "ci_low": lo, "ci_high": hi,
            "tier": evidence_tier(int(count), lo, hi),
            "domain": str(dom), "domain_count": dom_count,
            "template": TXT.get(str(reason)),
            "approvable": str(reason) in TXT and int(count) >= 8 and TXT[str(reason)]["key"] is not None,
        })
    return data


class EvaluationRequest(BaseModel):
    approved: Dict[str, bool] = Field(default_factory=dict)
    experience_floor: float = Field(default=4.0, ge=3.0, le=7.0)
    notice_cap: int = Field(default=60, ge=30, le=90)
    domain_filter: Optional[str] = "None"


def apply_filters(df: pd.DataFrame, approved: Dict[str, bool], exp_floor: float, notice_cap: int, domain_filter: str):
    skills = df["skills"].fillna("")
    mask = skills.str.contains("Java", regex=False)
    if approved.get("Too junior", False):
        mask &= df["experience_years"] >= exp_floor
    if approved.get("Missing critical skills", False):
        mask &= skills.str.contains("Spring Boot", regex=False)
    if approved.get("Notice period too long", False):
        mask &= df["notice_period_days"] <= notice_cap
    if domain_filter and domain_filter != "None":
        mask &= df["domain"] == domain_filter
    return df[mask].copy()


def score(df: pd.DataFrame):
    skills = df["skills"].fillna("")
    return df["experience_years"].clip(upper=10) * 4 + skills.str.contains("Spring Boot", regex=False).astype(int) * 30 - df["notice_period_days"] / 6


def evaluate_one(approved: Dict[str, bool], label: str, exp_floor: float, notice_cap: int, domain_filter: str):
    kept = apply_filters(hold, approved, exp_floor, notice_cap, domain_filter)
    top = kept.assign(_score=score(kept)).sort_values("_score", ascending=False).head(20)
    relevant_total = int((hold["recruiter_relevant"] == 1).sum())
    lost = hold[(hold["recruiter_relevant"] == 1) & (~hold["candidate_id"].isin(kept["candidate_id"]))]
    precision = float(top["recruiter_relevant"].mean()) if len(top) else None
    return {
        "setup": label,
        "profiles_passing": int(len(kept)),
        "precision_at_20": precision,
        "relevant_excluded": int(len(lost)),
        "relevant_total": relevant_total,
        "kept_relevant": int((kept["recruiter_relevant"] == 1).sum()),
    }


@app.get("/health")
def health():
    return {"status": "ok", "app": "recruiter-feedback-intelligence"}


@app.get("/api/observations")
def api_observations():
    return {
        "baseline_total": int(len(base)),
        "rejected": int(len(rej)),
        "shortlisted": int(len(base) - len(rej)),
        "rejection_rate": float(len(rej) / len(base)) if len(base) else 0,
        "observations": observations(),
        "domains": ["None"] + sorted(str(v) for v in hold["domain"].dropna().unique()),
    }


@app.post("/api/evaluate")
def api_evaluate(payload: EvaluationRequest):
    approved = {k: bool(v) for k, v in payload.approved.items()}
    baseline = evaluate_one({}, "Baseline (Java only)", payload.experience_floor, payload.notice_cap, "None")
    changed = evaluate_one(approved, "With approved changes", payload.experience_floor, payload.notice_cap, payload.domain_filter or "None")
    return {"results": [baseline, changed], "note": "All records and relevance labels are synthetic. Filtering can improve precision while excluding relevant candidates; review both outcomes."}


@app.get("/api/candidates")
def api_candidates():
    display = base.copy().fillna("")
    return JSONResponse(display.to_dict(orient="records"))


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recruiter Feedback Intelligence</title>
<style>
:root{--navy:#13243a;--blue:#315bea;--bg:#f4f6fa;--muted:#68758a;--line:#e2e7ef;--green:#147d64;--amber:#9a6700;--red:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#192435;font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Arial,sans-serif}.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 52px}.hero{background:linear-gradient(125deg,#14263d,#243f69);color:white;border-radius:20px;padding:28px 30px;margin-bottom:22px;box-shadow:0 12px 28px #13243a18}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:11px;font-weight:750;color:#c3d4ff}.hero h1{font-size:clamp(26px,4vw,36px);line-height:1.1;margin:9px 0}.hero p{color:#e0e8f6;max-width:800px;margin:10px 0 0}.pill{display:inline-block;background:#ffffff1a;border:1px solid #ffffff26;border-radius:999px;padding:5px 10px;font-size:12px;margin-top:17px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0 26px}.stat,.panel,.reason{background:#fff;border:1px solid var(--line);border-radius:15px}.stat{padding:18px}.stat .label{font-size:12px;color:var(--muted);font-weight:650}.stat .value{font-size:29px;font-weight:780;letter-spacing:-.03em;margin-top:3px;color:var(--navy)}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:28px 0 13px}.section-head h2{margin:0;font-size:21px;color:var(--navy)}.section-head p{margin:4px 0 0;color:var(--muted);font-size:13px}.reason-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.reason{padding:19px;display:flex;flex-direction:column;gap:12px}.reason-top{display:flex;align-items:flex-start;justify-content:space-between;gap:10px}.reason h3{margin:0;font-size:17px}.count{white-space:nowrap;background:#eff3fa;border-radius:9px;padding:5px 9px;font-weight:750;color:var(--navy)}.tier{font-size:11px;text-transform:uppercase;letter-spacing:.06em;font-weight:750;color:var(--amber)}.tier.low{color:var(--red)}.tier.good{color:var(--green)}.data-line{font-size:13px;color:var(--muted)}.reason p{margin:0;font-size:13px}.reason strong{color:#26374f}.hypothesis{background:#f7f9fd;border-radius:10px;padding:12px;display:grid;gap:7px}.approve{display:flex;align-items:flex-start;gap:9px;padding-top:8px;border-top:1px solid var(--line);font-weight:650;font-size:13px}.approve input{accent-color:var(--blue);margin-top:4px;transform:scale(1.1)}.not-enough{font-size:12px;color:var(--red);font-weight:600}.panel{padding:20px;margin-top:16px}.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.control label{display:flex;justify-content:space-between;font-size:13px;font-weight:700;margin-bottom:7px}.control input[type=range]{width:100%;accent-color:var(--blue)}.control select{width:100%;padding:10px;border:1px solid #d6deea;border-radius:9px;background:white;font:inherit}.actions{display:flex;flex-wrap:wrap;align-items:center;gap:12px;margin-top:18px}.btn{border:0;border-radius:10px;padding:11px 17px;font:inherit;font-weight:750;cursor:pointer}.btn-primary{background:var(--blue);color:white}.btn-ghost{background:#edf1f8;color:var(--navy)}.btn:disabled{opacity:.55;cursor:wait}.results{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:16px}.result-card{border:1px solid var(--line);border-radius:12px;padding:16px}.result-card h3{font-size:14px;margin:0 0 12px}.result-metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.mini{background:#f6f8fc;border-radius:9px;padding:10px}.mini span{display:block;font-size:11px;color:var(--muted)}.mini b{font-size:21px;color:var(--navy)}.note{padding:12px 14px;border-radius:10px;background:#fff8e8;color:#715113;font-size:12px;margin-top:14px}.safeguards{font-size:13px;color:var(--muted)}details{margin-top:22px;background:white;border:1px solid var(--line);border-radius:12px;padding:15px}summary{font-weight:750;cursor:pointer}.tablewrap{overflow:auto;margin-top:12px}table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}th{background:#f5f7fb;color:#44516a}.error{color:var(--red);font-size:13px}@media(max-width:760px){.wrap{padding:16px 12px 35px}.hero{padding:23px}.stats{grid-template-columns:repeat(2,minmax(0,1fr))}.reason-grid,.results{grid-template-columns:1fr}.controls{grid-template-columns:1fr}.section-head{align-items:flex-start;flex-direction:column}}
</style></head><body><main class="wrap">
<header class="hero"><div class="eyebrow">Evidence-led sourcing · Independent prototype</div><h1>Recruiter Feedback Intelligence</h1><p>Turn recruiter rejection patterns into testable sourcing experiments. Separate what the data shows from what the team hypothesises, require recruiter approval, then test changes on a fresh holdout batch.</p><span class="pill">100% synthetic data · Not affiliated with Naukri or AI REX</span></header>
<section class="stats" aria-label="Baseline batch summary"><div class="stat"><div class="label">Baseline candidates</div><div class="value" id="total">—</div></div><div class="stat"><div class="label">Rejected</div><div class="value" id="rejected">—</div></div><div class="stat"><div class="label">Shortlisted</div><div class="value" id="shortlisted">—</div></div><div class="stat"><div class="label">Rejection rate</div><div class="value" id="rate">—</div></div></section>
<div class="section-head"><div><h2>1. Observations → hypotheses</h2><p>Every proposed change requires recruiter approval. Confidence intervals indicate uncertainty, not causation.</p></div></div>
<div id="reasonGrid" class="reason-grid"><div class="panel">Loading observations…</div></div>
<div class="section-head"><div><h2>2. Test approved changes on a holdout batch</h2><p>The holdout is separate from the baseline so the same records are not used for both discovery and evaluation.</p></div></div>
<section class="panel"><div class="controls"><div class="control"><label for="expFloor">Experience floor <span id="expVal">4.0 years</span></label><input id="expFloor" type="range" min="3" max="7" value="4" step="0.5"><div class="data-line">Applied only when “Too junior” is approved.</div></div><div class="control"><label for="noticeCap">Notice-period cap <span id="noticeVal">60 days</span></label><input id="noticeCap" type="range" min="30" max="90" value="60" step="15"><div class="data-line">Applied only when “Notice period too long” is approved.</div></div><div class="control"><label for="domainFilter">Illustrative domain filter</label><select id="domainFilter"><option>None</option></select><div class="data-line">Optional filter for exploring trade-offs; not a recommended change.</div></div></div><div class="actions"><button id="runBtn" class="btn btn-primary">Evaluate holdout batch</button><button id="resetBtn" class="btn btn-ghost">Reset approvals</button><span id="status" class="data-line" role="status"></span></div><div id="results" class="results" aria-live="polite"></div><div id="evalNote" class="note" hidden></div></section>
<details><summary>Records and safeguards</summary><p class="safeguards">The included baseline and holdout records are synthetic. Rejection tags are one recruiter's label, not ground truth, and cannot prove that a job description or search caused a rejection. The sample includes planted patterns to demonstrate the method, not real-world performance. Never use protected traits; keep a human recruiter in the loop, audit for disparate impact, and follow data-protection rules before real use.</p><div class="tablewrap"><table id="candidateTable"><thead></thead><tbody></tbody></table></div></details>
</main><script>
const pct=n=>`${Math.round(n*100)}%`; let obs=[];
async function init(){try{const r=await fetch('/api/observations');const d=await r.json();obs=d.observations;document.getElementById('total').textContent=d.baseline_total;document.getElementById('rejected').textContent=d.rejected;document.getElementById('shortlisted').textContent=d.shortlisted;document.getElementById('rate').textContent=pct(d.rejection_rate);const grid=document.getElementById('reasonGrid');grid.innerHTML='';obs.forEach((o,i)=>{const t=o.tier==='Too few records to act on'?'low':(o.tier==='More consistent pattern'?'good':'');const tpl=o.template;const card=document.createElement('article');card.className='reason';let body=`<div class="reason-top"><div><div class="tier ${t}">${o.tier}</div><h3>${esc(o.reason)}</h3></div><div class="count">${o.count} records</div></div><div class="data-line">${o.count} of ${d.rejected} rejections (${pct(o.share)}); 95% Wilson interval ${pct(o.ci_low)}–${pct(o.ci_high)}.</div><p><strong>Observation:</strong> Most common domain was ${esc(o.domain)} (${o.domain_count} of ${o.count}).</p>`;if(tpl){body+=`<div class="hypothesis"><p><strong>Hypothesis (unproven):</strong> ${esc(tpl.hypothesis)}</p><p><strong>Suggested change:</strong> ${esc(tpl.change)}</p><p><strong>How to check:</strong> ${esc(tpl.check)}</p></div>`;}else{body+=`<p class="data-line">${o.count<8?'Too few records to suggest a change.':'No tested recommendation template for this reason.'}</p>`;}if(o.approvable){body+=`<label class="approve"><input type="checkbox" data-reason="${esc(o.reason)}"><span>Approve this change for the simulation</span></label>`;}else if(tpl&&o.count<8){body+=`<div class="not-enough">Gather more feedback before changing criteria.</div>`;}card.innerHTML=body;grid.appendChild(card);});const sel=document.getElementById('domainFilter');d.domains.forEach(v=>{if(v!=='None'){const o=document.createElement('option');o.value=v;o.textContent=v;sel.appendChild(o);}});await loadCandidates();}catch(e){document.getElementById('reasonGrid').innerHTML='<p class="error">Could not load the data. Check that the CSV files are included in the project.</p>';}}
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function loadCandidates(){const r=await fetch('/api/candidates');const rows=await r.json();if(!rows.length)return;const keys=Object.keys(rows[0]);document.querySelector('#candidateTable thead').innerHTML='<tr>'+keys.map(k=>'<th>'+esc(k.replaceAll('_',' '))+'</th>').join('')+'</tr>';document.querySelector('#candidateTable tbody').innerHTML=rows.map(row=>'<tr>'+keys.map(k=>'<td>'+esc(row[k])+'</td>').join('')+'</tr>').join('');}
document.getElementById('expFloor').addEventListener('input',e=>document.getElementById('expVal').textContent=Number(e.target.value).toFixed(1)+' years');document.getElementById('noticeCap').addEventListener('input',e=>document.getElementById('noticeVal').textContent=e.target.value+' days');
document.getElementById('resetBtn').addEventListener('click',()=>{document.querySelectorAll('[data-reason]').forEach(c=>c.checked=false);document.getElementById('expFloor').value=4;document.getElementById('expVal').textContent='4.0 years';document.getElementById('noticeCap').value=60;document.getElementById('noticeVal').textContent='60 days';document.getElementById('domainFilter').value='None';document.getElementById('results').innerHTML='';document.getElementById('evalNote').hidden=true;document.getElementById('status').textContent='Approvals and filters reset.';});
document.getElementById('runBtn').addEventListener('click',async()=>{const btn=document.getElementById('runBtn'),status=document.getElementById('status');btn.disabled=true;status.textContent='Evaluating…';try{const approved={};document.querySelectorAll('[data-reason]').forEach(c=>approved[c.dataset.reason]=c.checked);const payload={approved,experience_floor:Number(document.getElementById('expFloor').value),notice_cap:Number(document.getElementById('noticeCap').value),domain_filter:document.getElementById('domainFilter').value};const resp=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!resp.ok)throw new Error('Evaluation request failed');const data=await resp.json();document.getElementById('results').innerHTML=data.results.map(r=>`<article class="result-card"><h3>${esc(r.setup)}</h3><div class="result-metrics"><div class="mini"><span>Profiles passing</span><b>${r.profiles_passing}</b></div><div class="mini"><span>Precision@20</span><b>${r.precision_at_20===null?'—':pct(r.precision_at_20)}</b></div><div class="mini"><span>Relevant excluded</span><b>${r.relevant_excluded} / ${r.relevant_total}</b></div><div class="mini"><span>Relevant retained</span><b>${r.kept_relevant}</b></div></div></article>`).join('');const note=document.getElementById('evalNote');note.textContent=data.note;note.hidden=false;status.textContent='Evaluation complete.';}catch(e){status.textContent='Could not evaluate. Please retry.';}finally{btn.disabled=false;}});
init();
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(PAGE)
