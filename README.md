# Explainable AI in Criminal Justice — Educational Prototype

A deployable Flask demonstration of explainability in a justice-adjacent workflow. It highlights **case-file documentation gaps** that may warrant a human review.

## Important scope and safeguards
- This is a classroom demonstration, not a real justice tool.
- The model uses **synthetic generated data only**.
- It does **not** predict guilt, criminality, recidivism, dangerousness, bail eligibility, or sentencing.
- Do not enter real names, case numbers, personal data, or real case details.
- The displayed percentage is a toy model's review signal, not a calibrated real-world probability.
- Feature contributions describe the toy model's behavior; they are not causal explanations.
- A human reviewer must independently verify any documentation issue.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\\Scripts\\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Deploy on Render
1. Create a GitHub repository and upload these project files.
2. Sign in to Render and choose **New + → Web Service**.
3. Connect the repository. Render should detect `render.yaml`; if prompted, use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Deploy, then open the provided `onrender.com` URL.

## How it works
1. The user selects documentation flags in the form.
2. A logistic regression model trained on generated toy examples estimates whether the file may need documentation review.
3. The page shows the model's review signal and the active feature contributions.
4. The output is explicitly advisory and not a legal conclusion.

## Suggested case-study evaluation
- Test no flags, one flag, and several flags.
- Observe how the review signal changes.
- Discuss transparency, false positives/negatives, data quality, bias, calibration, human oversight, and why this prototype must not be used for real decisions.
