# Job Salary Prediction — Backend (Flask)

Simple Flask API that returns a salary prediction given job title, skills and experience.
This backend loads a pre-trained model (`model.pkl`) and exposes two endpoints:

- `GET /` — health check
- `POST /predict` — predict salary

---

## Quick Links
- Repo: https://github.com/Mayaaids/job-salary-backend

---

## Files in this folder
- `app.py` — Flask application (health + predict endpoints)
- `model.pkl` — trained ML model used for prediction
- `requirements.txt` — Python dependencies
- `Procfile` — start command for Render (`web: gunicorn app:app`)
- `README.md` — this file

---

## Run locally (quick)
1. Open terminal in this folder.
2. (Optional) create & activate a Python virtual environment.
3. Install dependencies:
```bash
pip install -r requirements.txt
