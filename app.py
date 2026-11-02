"""JusticeLens - Explainable AI in criminal-justice *documentation* workflows.

SCOPE: this educational prototype does NOT predict guilt, criminality,
recidivism, bail eligibility or sentencing. It flags documentation gaps in
fictional, synthetic case files that may warrant human review.
"""
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

SEED = 23

# key, short label (results), long label (checklist), description, icon name
FEATURES = [
    ("missing_evidence", "Evidence attachment missing", "Evidence attachment missing",
     "Referenced attachment is absent", "paperclip"),
    ("timeline_conflict", "Timeline inconsistency", "Timeline inconsistency flagged",
     "Dates/times appear inconsistent", "clock"),
    ("chain_gap", "Chain-of-custody gap", "Chain-of-custody documentation gap",
     "Custody transfer entry missing", "link"),
    ("unreadable_report", "Report unreadable or incomplete", "Report is unreadable or incomplete",
     "Report marked unreadable or pages missing", "file"),
    ("statement_missing", "Referenced statement missing", "Referenced statement missing",
     "Witness statement absent", "clipboard"),
]
KEYS = [f[0] for f in FEATURES]
META = {f[0]: {"short": f[1], "label": f[2], "desc": f[3], "icon": f[4]} for f in FEATURES}


def build_demo_model():
    """Train a logistic regression on 1,800 fully synthetic, reproducible rows."""
    rng = np.random.default_rng(SEED)
    X = rng.integers(0, 2, size=(1800, 5))
    weights = np.array([1.25, 1.05, 1.35, 0.95, 0.85])
    logits = -1.65 + X @ weights + rng.normal(0, 0.55, size=len(X))
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(len(X)) < probs).astype(int)
    model = LogisticRegression(random_state=SEED, max_iter=1000)
    model.fit(X, y)
    return model


# Built once at import time: no pickle files, no load latency.
MODEL = build_demo_model()
COEFS = {k: float(c) for k, c in zip(KEYS, MODEL.coef_[0])}
INTERCEPT = float(MODEL.intercept_[0])


def signal_level(pct: float):
    if pct >= 65:
        return "Higher review signal", "HIGH SIGNAL", "high"
    if pct >= 35:
        return "Some review signal", "MED SIGNAL", "med"
    return "Lower review signal", "LOW SIGNAL", "low"


def analyze(selected):
    """Score a set of selected flags and return a fully explained result."""
    selected = [k for k in KEYS if k in set(selected)]
    x = np.array([[1 if k in selected else 0 for k in KEYS]])
    prob = float(MODEL.predict_proba(x)[0][1])
    pct = round(prob * 100)
    level, badge, tone = signal_level(prob * 100)

    contribs = [
        {"key": k, "label": META[k]["short"], "value": round(COEFS[k], 3)}
        for k in selected
    ]
    contribs.sort(key=lambda c: c["value"], reverse=True)
    logit = INTERCEPT + sum(COEFS[k] for k in selected)
    return {
        "probability": round(prob, 4),
        "percent": pct,
        "level": level,
        "badge": badge,
        "tone": tone,
        "selected": selected,
        "contributions": contribs,
        "intercept": round(INTERCEPT, 3),
        "logit": round(logit, 3),
        "coefficients": {k: round(v, 3) for k, v in COEFS.items()},
        "max_coef": round(max(COEFS.values()), 3),
    }


def _selected_from_request():
    if request.is_json:
        data = request.get_json(silent=True) or {}
        flags = data.get("flags", data)
        if isinstance(flags, list):
            return [k for k in flags if k in KEYS]
        return [k for k in KEYS if flags.get(k) in (True, "on", 1, "1", "true")]
    return [k for k in KEYS if request.form.get(k) == "on"]


@app.route("/", methods=["GET", "POST"])
def index():
    result = analyze(_selected_from_request()) if request.method == "POST" else None
    return render_template(
        "index.html",
        features=FEATURES,
        coefs=COEFS,
        max_coef=max(COEFS.values()),
        result=result,
        selected=result["selected"] if result else [],
    )


@app.route("/analyze", methods=["POST"])
def analyze_api():
    return jsonify(analyze(_selected_from_request()))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
