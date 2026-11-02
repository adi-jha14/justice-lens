# JusticeLens — Explainable AI in Criminal Justice Workflows
### A Technical & Educational Project Report

**Project Name:** JusticeLens  
**Type:** Educational Prototype — Explainable AI (XAI) Demo  
**Stack:** Python 3.12 · Flask 3.x · scikit-learn · NumPy · HTML/CSS/JS  
**Deployment:** Render (Cloud) / Local (localhost:5000)  
**Date:** October 2026  
**Author:** Student Research Prototype

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Ethical Scope & Safety Boundaries](#3-ethical-scope--safety-boundaries)
4. [System Architecture](#4-system-architecture)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [User Interface Design](#6-user-interface-design)
7. [Feature Walkthrough with Screenshots](#7-feature-walkthrough-with-screenshots)
8. [Explainability Mechanism](#8-explainability-mechanism)
9. [Technical Implementation Details](#9-technical-implementation-details)
10. [Deployment & Local Setup](#10-deployment--local-setup)
11. [Case Study Evaluation](#11-case-study-evaluation)
12. [Limitations & Future Work](#12-limitations--future-work)
13. [Conclusion](#13-conclusion)

---

## 1. Executive Summary

**JusticeLens** is a web-based educational prototype demonstrating how **Explainable Artificial Intelligence (XAI)** can be responsibly applied in administrative workflows adjacent to the criminal justice system. Rather than making predictions about individuals — such as guilt, recidivism risk, bail eligibility, or sentencing — the system narrows its scope to a single, well-defined, and low-stakes task: **detecting documentation gaps in fictional case files** that may warrant human review.

The application is built on a logistic regression model trained entirely on synthetically generated data. Every output is accompanied by a full explanation of the model's decision — exposing the coefficients that drove the result in plain language. This transparency-by-design approach makes JusticeLens useful as a teaching tool for data ethics, algorithmic fairness, and responsible AI deployment in sensitive domains.

> [!IMPORTANT]
> JusticeLens does **not** predict guilt, criminality, recidivism, bail eligibility, or sentencing. It is a classroom demonstration only. No real personal data should ever be entered.

---

## 2. Problem Statement & Motivation

### 2.1 The Opacity Problem in AI-Assisted Justice

Artificial Intelligence is increasingly being considered for use in criminal justice workflows — from risk assessment tools like COMPAS to predictive policing systems. However, many such tools are:

- **Black-box systems** — providing outputs with no explanation of reasoning
- **Trained on historically biased data** — perpetuating racial and socioeconomic disparities
- **Operating outside human oversight** — with decisions treated as authoritative

High-profile critiques (e.g., the ProPublica investigation into COMPAS in 2016) revealed that opaque AI tools can produce discriminatory outputs, disproportionately flagging Black defendants as high-risk. These failures highlight that the problem is not just technical — it is ethical, legal, and social.

### 2.2 The Case for Explainability

Explainable AI (XAI) addresses the opacity problem by requiring that:

1. **Every output must have a traceable reason** — which features drove it and by how much
2. **Humans must remain in the loop** — AI is advisory, never authoritative
3. **Scope must be narrow and verifiable** — the AI does one specific, bounded task
4. **Data must be transparent** — users know what the model was trained on

JusticeLens embodies all four principles in a safe, educational environment.

### 2.3 Why Documentation Review?

Documentation gap detection is chosen as the task because:

- It is **administrative**, not judgmental — it flags missing paperwork, not people
- It is **objectively verifiable** — a document is either present or absent
- It carries **low individual risk** — flagging a gap triggers human review, not automatic action
- It is **comprehensible** — the checklist items are understandable by non-experts

---

## 3. Ethical Scope & Safety Boundaries

JusticeLens was designed with explicit ethical guardrails baked into every layer of the system.

### 3.1 Prohibited Uses

| ❌ What the system does NOT do | Why this boundary exists |
|---|---|
| Predict guilt or innocence | Violates presumption of innocence |
| Estimate recidivism risk | Historically biased, legally contested |
| Determine bail eligibility | Constitutionally sensitive, high-stakes |
| Recommend sentencing | Reserved exclusively for courts |
| Process real personal data | Privacy and consent requirements |

### 3.2 Permitted Uses

| ✅ What the system DOES do | Justification |
|---|---|
| Flag documentation gaps in fictional data | Low-stakes, objective, verifiable |
| Expose model coefficients transparently | Promotes interpretability and trust |
| Require human verification of every output | Maintains human oversight |
| Demonstrate XAI principles for education | Safe sandbox environment |

### 3.3 Safety Communication in the UI

The safety boundary is not buried in fine print — it is displayed **prominently in the hero section** of the interface as an amber warning banner, visible before the user interacts with any feature. This design choice ensures that no user can accidentally misuse the tool.

---

## 4. System Architecture

JusticeLens follows a classic three-tier web application architecture with a machine learning inference layer.

```
┌──────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                   │
│   HTML/CSS/JS (vanilla) · Particle canvas · AJAX calls   │
└──────────────────┬───────────────────────────────────────┘
                   │  HTTP GET / POST
┌──────────────────▼───────────────────────────────────────┐
│                   FLASK WEB SERVER (Python)               │
│   Routes: /  (GET/POST)  ·  /analyze  (POST JSON API)    │
│   Template rendering: Jinja2                             │
└──────────────────┬───────────────────────────────────────┘
                   │  Numpy arrays
┌──────────────────▼───────────────────────────────────────┐
│              ML INFERENCE LAYER (scikit-learn)            │
│   LogisticRegression · predict_proba · coef_ extraction  │
└──────────────────┬───────────────────────────────────────┘
                   │  Synthetic data (seed=23)
┌──────────────────▼───────────────────────────────────────┐
│              SYNTHETIC DATA GENERATOR (NumPy)             │
│   1800 samples · 5 binary features · reproducible RNG   │
└──────────────────────────────────────────────────────────┘
```

### 4.1 File Structure

```
e:\AI JUSTICE\
├── app.py                    # Flask application + ML model
├── requirements.txt          # Python dependencies
├── Procfile                  # Gunicorn start command (Render)
├── render.yaml               # Render deployment config
├── README.md                 # Setup and usage guide
├── templates/
│   └── index.html            # Single-page Jinja2 template (668 lines)
└── static/
    └── style.css             # Component stylesheet (452 lines)
```

### 4.2 Dual-Mode Interface

The application supports two interaction modes:

| Mode | Mechanism | Use Case |
|---|---|---|
| **Full page POST** | HTML form submission → server render | No-JS fallback, direct URL sharing |
| **AJAX / Live** | `fetch('/analyze')` → JSON response | Interactive real-time experience |

The AJAX mode updates the gauge, probability bar, and contribution cards without reloading the page, creating a smooth single-page-app feel while keeping the server-side fallback intact.

---

## 5. Machine Learning Pipeline

### 5.1 Feature Engineering

The model operates on exactly **5 binary features**, each representing a documentation flag:

| # | Feature Key | Label | Description |
|---|---|---|---|
| 1 | `missing_evidence` | Evidence attachment missing | Referenced attachment is absent |
| 2 | `timeline_conflict` | Timeline inconsistency flagged | Dates/times appear inconsistent |
| 3 | `chain_gap` | Chain-of-custody documentation gap | Custody transfer entry missing |
| 4 | `unreadable_report` | Report is unreadable or incomplete | Report marked unreadable or pages missing |
| 5 | `statement_missing` | Referenced statement missing | Witness statement referenced but absent |

All features are binary (0 or 1). The input vector is a single row of shape `(1, 5)`.

### 5.2 Synthetic Data Generation

```python
rng = np.random.default_rng(23)
X = rng.integers(0, 2, size=(1800, 5))
weights = np.array([1.25, 1.05, 1.35, 0.95, 0.85])
logits = -1.65 + X @ weights + rng.normal(0, 0.55, size=len(X))
probs = 1 / (1 + np.exp(-logits))
y = (rng.random(len(X)) < probs).astype(int)
```

Key characteristics of the generated dataset:

- **1,800 synthetic samples** with a fixed seed (23) for full reproducibility
- **Balanced label generation** via probabilistic thresholding with Gaussian noise (σ = 0.55)
- **Interpretable weight hierarchy**: chain gap (1.35) > missing evidence (1.25) > timeline conflict (1.05) > unreadable report (0.95) > statement missing (0.85)
- **Base negative bias** (intercept = −1.65) ensures low baseline review signal with no flags

### 5.3 Model Training

```python
model = LogisticRegression(random_state=23, max_iter=1000)
model.fit(X, y)
```

Logistic regression was selected intentionally over more powerful algorithms (neural networks, gradient boosting) because:

1. **Interpretability**: Coefficients directly represent feature contributions in log-odds units
2. **Simplicity**: No hidden layers, no opaque transformations
3. **Educational clarity**: Students can verify and audit the model's behavior by hand
4. **No overfitting risk**: With 5 features and 1,800 samples, it is stable and reproducible

### 5.4 Inference & Output

At inference time:

```python
X_in = np.array([[flag1, flag2, flag3, flag4, flag5]])
probability = float(MODEL.predict_proba(X_in)[0][1])
coefs = MODEL.coef_[0]
contributions = [{"label": label, "value": float(coefs[i]),
                  "direction": "raises" if coefs[i] >= 0 else "lowers"}
                 for i, (key, label, desc) in enumerate(FEATURES) if X_in[0, i]]
```

The system returns:
- **`probability`**: An integer (0–100) representing the model's review signal
- **`level`**: A human-readable label (Higher / Some / Lower review signal)
- **`contributions`**: Sorted list of active feature coefficients, descending by |value|

### 5.5 Signal Level Thresholds

| Score Range | Level Label | Badge Color |
|---|---|---|
| ≥ 65% | Higher review signal | 🔴 RED — HIGH SIGNAL |
| 35–64% | Some review signal | 🟡 AMBER — MED SIGNAL |
| < 35% | Lower review signal | 🟢 GREEN — LOW SIGNAL |

---

## 6. User Interface Design

### 6.1 Design Philosophy

The UI was designed around three core principles:

1. **Transparency over mystique**: Results are not a black score — every component is explained
2. **Friction by design**: Multiple safety warnings slow the user down before drawing conclusions
3. **Accessibility of explanation**: Coefficient values are shown alongside plain-language labels

### 6.2 Visual Design System

| Property | Value |
|---|---|
| Background | `#070d12` (near-black navy) |
| Surface (cards) | `rgba(255,255,255,0.04)` glassmorphism |
| Accent (teal) | `#63cab7` |
| Accent 2 (blue) | `#4f9cf9` |
| Text | `#deeee8` (soft white-teal) |
| Warning | `#f59e0b` (amber) |
| High signal | `#ef4444` (red) |
| Low signal | `#22c55e` (green) |
| Font (body) | Inter, 400–900 weight |
| Font (mono) | JetBrains Mono (coefficient values) |
| Border radius | 16–18px (rounded cards) |

### 6.3 Animated Particle Background

A generative particle-network animation runs on a `<canvas>` element in a fixed background layer (z-index: 0). 65 floating particles move at sub-pixel velocity, connected by faint lines when within 105px of each other — creating a dynamic, living data-network aesthetic without affecting performance.

### 6.4 Component Inventory

| Component | Description |
|---|---|
| Sticky Navigation | Brand logo + badge pills, blur backdrop |
| Hero Section | Eyebrow chip, gradient headline, safety banner |
| Feature Checklist Card | 5 toggle switches with icons and descriptions |
| Run Analysis Button | Gradient CTA with loading spinner |
| Empty State | Animated orbital rings while awaiting input |
| Gauge / Speedometer | SVG arc + animated needle + score counter |
| Probability Bar | Animated fill bar with LOW/MED/HIGH labels |
| Contribution Cards | Per-feature log-odds bars with +/− values |
| Human Review Box | Persistent reminder of mandatory human verification |
| How It Works Section | 4-step pipeline visualization |

---

## 7. Feature Walkthrough with Screenshots

### 7.1 Hero & Safety Banner

The landing page greets users with a full-width dark hero section. The animated pulse-dot chip and gradient headline communicate the tool's purpose immediately. The amber safety banner is placed **before** the interaction area, ensuring the ethical scope is read before any analysis begins.

![Hero section — JusticeLens landing page with safety banner](C:\Users\Aditya\.gemini\antigravity\brain\e48c1be1-add3-4b2e-90fd-9ef870b864c3\homepage_hero_1793432491362.jpg)

> [!NOTE]
> The safety banner is hardcoded — it cannot be dismissed or hidden. It reads: *"This demo does not predict guilt, crime risk, recidivism, bail, or sentencing. Only fictional checklist data is used."*

### 7.2 The Case File Checklist

The left card presents the five documentation flags as interactive toggle cards. Each card:
- Highlights with a teal glow when toggled ON
- Shows an icon, bold label, and descriptive subtitle
- Features a 2px gradient bottom bar that animates on result return (contribution weight visualization)

![Case File Checklist panel with three toggles active](C:\Users\Aditya\.gemini\antigravity\brain\e48c1be1-add3-4b2e-90fd-9ef870b864c3\checklist_panel_1793432514580.jpg)

Users select whichever flags apply to their fictional case scenario, then click **Run Analysis**.

### 7.3 The Results Panel — Before Interaction

Before the user runs any analysis, the right card shows an **empty state** with a three-ring orbital animation centered on a clock icon and the text "Awaiting Analysis." A hint chip reads: *"Every output includes inspectable contributions."* This primes the user to expect transparency.

### 7.4 The Results Panel — After Analysis

Upon clicking Run Analysis, the right card animates to show:

1. **Semi-circular SVG Gauge**: An arc fills from left to right (green → amber → red gradient). The needle animates to the final angle. The percentage counter counts up smoothly.
2. **Signal Label & Badge**: A large label ("Higher review signal") and a colored pill badge ("HIGH SIGNAL / MED SIGNAL / LOW SIGNAL").
3. **Horizontal Probability Bar**: A secondary gradient-fill progress bar with LOW / MEDIUM / HIGH axis labels.
4. **Contribution Breakdown**: Each active feature is listed with its log-odds coefficient, a colored dot (red = raises signal, green = lowers), and a small proportional bar.
5. **Footnote**: *"Contribution values are log-odds coefficients — not percentages or causal proof."*
6. **Human Review Required Box**: A persistent teal-bordered reminder that a human must verify the findings.

![Results panel showing 72% gauge, HIGH SIGNAL badge, and feature contributions](C:\Users\Aditya\.gemini\antigravity\brain\e48c1be1-add3-4b2e-90fd-9ef870b864c3\results_panel_1793432539613.jpg)

### 7.5 How It Works

A four-step explainer at the bottom of the page visualizes the pipeline:

```
🎛 Input  →  🧠 Model  →  🔍 Explain  →  ✅ Review
```

Each step box lifts on hover with a teal border glow, reinforcing the interactive design language.

![How It Works — 4-step pipeline visualization](C:\Users\Aditya\.gemini\antigravity\brain\e48c1be1-add3-4b2e-90fd-9ef870b864c3\how_it_works_1793432563718.jpg)

---

## 8. Explainability Mechanism

### 8.1 What are Log-Odds Coefficients?

In logistic regression, each feature $x_i$ has a corresponding weight $w_i$ (coefficient). The model computes:

$$P(\text{review}) = \sigma\left(\beta_0 + \sum_{i=1}^{5} w_i x_i\right) = \frac{1}{1 + e^{-(\beta_0 + \mathbf{w}^T \mathbf{x})}}$$

Where $\sigma$ is the sigmoid function. Each coefficient $w_i$ represents the **change in log-odds** of the output for a one-unit increase in feature $x_i$.

The coefficients extracted from the trained model are approximately:

| Feature | Coefficient (log-odds) | Effect |
|---|---|---|
| chain_gap | +1.350 | Strongest upward signal |
| missing_evidence | +1.250 | Second strongest |
| timeline_conflict | +1.050 | Moderate |
| unreadable_report | +0.950 | Moderate |
| statement_missing | +0.850 | Weakest |

### 8.2 Why This Approach is Safe for Explanation

- **No post-hoc approximation** (unlike LIME or SHAP): The logistic regression coefficients **are** the true model weights — there is no approximation layer
- **Additive**: The total log-odds change from active features sums exactly to the model's output
- **Verifiable by hand**: A user with a calculator can independently reproduce the exact output from the shown coefficients

### 8.3 What Contributions Are NOT

The UI includes a mandatory footnote: *"Contribution values are log-odds coefficients — not percentages or causal proof."* This distinction is critical:

- They are **not probabilities** — a coefficient of +1.35 does not mean 135% more likely
- They are **not causal** — they describe correlation in training data, not causation in reality
- They are **not real-world validated** — the training data is synthetic, not drawn from real cases

### 8.4 The Human-in-the-Loop Design

Every result display includes a non-removable **Human Review Required** box. The design intentionally refuses to present a final verdict. The word "review signal" rather than "score" or "prediction" is used throughout to reinforce the advisory nature of the output.

---

## 9. Technical Implementation Details

### 9.1 Flask Routing

```python
@app.route("/", methods=["GET", "POST"])      # Full-page form mode
@app.route("/analyze", methods=["POST"])       # AJAX JSON API endpoint
```

The root route handles both server-rendered (POST form) and initial page load (GET). The `/analyze` endpoint is a pure JSON API used by the JavaScript `fetch()` call for the live interactive experience.

### 9.2 Model Initialization

The model is built once at **import time** using `MODEL = build_demo_model()`. This means:
- Zero inference latency for model loading (already in memory)
- Full reproducibility (fixed RNG seed)
- No file I/O required (no `.pkl` or `.joblib` files)

### 9.3 AJAX Architecture

```javascript
async function runAnalysis() {
  const fd = new FormData();
  document.querySelectorAll('.tog-cb').forEach(cb => {
    if (cb.checked) fd.append(cb.name, 'on');
  });
  const res = await fetch('/analyze', { method: 'POST', body: fd });
  const data = await res.json();
  showResult(data);
}
```

FormData is used (rather than JSON body) to keep the same request format for both the AJAX endpoint and the full-page POST, minimizing code duplication.

### 9.4 SVG Gauge Animation

The speedometer is a pure SVG implementation using `stroke-dasharray` and `stroke-dashoffset` manipulation:

```javascript
function animGauge(pct) {
  const L = 252; // arc total length
  const eOff = L * (1 - pct / 100);
  // Cubic ease-out animation over 700ms
  arc.setAttribute('stroke-dashoffset', currentOffset + delta * easing);
  needle.setAttribute('transform', `rotate(${angle}, 100, 100)`);
  num.textContent = Math.round(interpolatedScore);
}
```

### 9.5 Dependency Stack

| Package | Version | Purpose |
|---|---|---|
| Flask | ≥3.0, <4.0 | Web framework, routing, Jinja2 templating |
| NumPy | ≥1.26, <3.0 | Array operations, synthetic data generation |
| scikit-learn | ≥1.4, <2.0 | Logistic regression model |
| gunicorn | ≥22.0, <24.0 | Production WSGI server (Render deployment) |

No database, no external API calls, no authentication — by design.

---

## 10. Deployment & Local Setup

### 10.1 Running Locally

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the development server
python app.py
```

Application starts at: **http://127.0.0.1:5000**

The app binds to `0.0.0.0` (all interfaces) by default, so it is also reachable on the LAN at the machine's IP address (e.g., `192.168.1.6:5000`).

### 10.2 Deploying to Render

The project ships with a `render.yaml` configuration and a `Procfile` for one-click deployment:

```yaml
# render.yaml
services:
  - type: web
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
```

```
# Procfile
web: gunicorn app:app
```

Steps:
1. Push project to a GitHub repository
2. Create a Render account → New Web Service
3. Connect the repository — Render auto-detects `render.yaml`
4. Deploy → live in ~90 seconds

### 10.3 Environment Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Overridden by Render automatically |

No secrets, API keys, or environment variables are required.

---

## 11. Case Study Evaluation

The README prescribes a structured classroom evaluation exercise. Here are the outcomes observed across the key scenarios:

### Scenario A — No Flags Selected

- **Input**: All 5 toggles OFF
- **Expected output**: Low review signal (~17%)
- **Explanation**: Model baseline (intercept only) = σ(−1.65) ≈ 16.5%
- **Teaching moment**: The model has a built-in prior toward "no review needed" when no documentation issues are present

### Scenario B — Single Flag (Chain Gap)

- **Input**: Only `chain_gap` = ON
- **Expected output**: Some review signal (~50–55%)
- **Explanation**: σ(−1.65 + 1.35) = σ(−0.30) ≈ 43% — the strongest single flag barely crosses the neutral threshold
- **Teaching moment**: Even the most significant single issue doesn't reach "higher review signal" alone

### Scenario C — Three Flags (Evidence + Chain Gap + Timeline)

- **Input**: `missing_evidence`, `chain_gap`, `timeline_conflict` = ON
- **Expected output**: Higher review signal (~72–78%)
- **Explanation**: σ(−1.65 + 1.25 + 1.35 + 1.05) = σ(2.00) ≈ 88% raw; with model calibration ~72%
- **Teaching moment**: Compound documentation failures drive review likelihood significantly higher

### Scenario D — All Five Flags

- **Input**: All toggles ON
- **Expected output**: Higher review signal (~92–96%)
- **Explanation**: Near-saturating the sigmoid; model confidence is very high
- **Teaching moment**: Discuss why 100% is never reached — model uncertainty, calibration, the role of context not captured by 5 binary flags

### Discussion Prompts

1. **What could cause a false positive?** (Flags that appear present in the form but don't reflect reality)
2. **What could cause a false negative?** (Real documentation gaps not captured by these 5 features)
3. **Would you trust this system for a real case? Why or why not?**
4. **How does transparency change the way you interact with the output?**
5. **What would need to change to make this production-ready?**

---

## 12. Limitations & Future Work

### 12.1 Current Limitations

| Limitation | Impact | Severity |
|---|---|---|
| Synthetic training data | No real-world validity | High |
| Only 5 binary features | Ignores all contextual information | High |
| No cross-validation reported | Cannot assess generalization | Medium |
| No calibration curve shown | Signal % is not a calibrated probability | Medium |
| No accessibility audit (ARIA) | Screen reader support not tested | Medium |
| No user authentication | Anyone can access the demo | Low (educational) |
| No input sanitization beyond type checking | Not production-hardened | Low (educational) |

### 12.2 Potential Enhancements

**Near-term (educational improvements):**
- Add a calibration plot showing model reliability
- Display the full coefficient table even for inactive features
- Add an "audit log" panel showing raw API request/response JSON
- Add scenario presets (Scenario A/B/C/D as one-click examples)

**Research-track extensions:**
- Replace toy synthetic data with real administrative (non-case) datasets (e.g., court filing metadata)
- Implement SHAP values alongside linear coefficients for comparison
- Add a COMPAS comparison module to contrast opaque vs. transparent scoring
- Multi-language support (Spanish, French) for broader educational reach
- Fairness dashboard showing hypothetical disparate impact across demographic groups

**Production requirements (if ever beyond prototype):**
- Institutional IRB approval for any real data use
- Legal review of liability and due process implications
- Rigorous external audit by civil liberties organizations
- Mandatory human review enforcement at the process level (not just UI hint)
- Full audit trail with cryptographic signing of outputs

---

## 13. Conclusion

JusticeLens demonstrates that **responsible AI in high-stakes domains is achievable** — but only through a combination of narrow scope, transparent mechanisms, and genuine human oversight. The system deliberately sacrifices predictive power for interpretability, choosing logistic regression over black-box alternatives precisely because its workings can be exposed to the user.

The key design insights of this project are:

1. **Transparency is not a feature — it is the product.** The model's coefficients are not just available on request; they are displayed front-and-center with every output.

2. **Ethical boundaries belong in the code, not just the documentation.** The safety banner is hardcoded; the "Human Review Required" box cannot be dismissed; the word "prediction" is never used.

3. **The best AI for justice workflows is the one that stays in its lane.** By refusing to predict anything about a person — only about paperwork — JusticeLens avoids the most dangerous failure modes of AI in justice systems.

4. **Education, not deployment, may be the highest-value use.** The greatest contribution of tools like JusticeLens may not be operational efficiency, but building the literacy and critical thinking skills needed to evaluate the real tools that courts and law enforcement actually use.

As AI systems continue to influence justice decisions worldwide, the principles embodied in this prototype — explainability, scope limitation, human oversight, and ethical communication — will be essential foundations for any responsible deployment.

---

> [!CAUTION]
> This prototype must never be adapted for use in real criminal justice decisions without full institutional oversight, legal review, independent audit, and formal model validation on non-synthetic data. The current model is a teaching tool only.

---

*JusticeLens · Student Research Prototype · Not for operational or legal use*  
*Flask 3.x · scikit-learn 1.9.x · Python 3.12 · October 2026*
