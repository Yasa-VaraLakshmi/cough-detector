# Cough + Speaker Detection

This project provides a simple cough detection system with a browser mic frontend.

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Generate starter data

```powershell
python data_gen.py
```

## Run

```powershell
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser.

## Notes
- Data is stored in `data/` and models in `models/`.
- The model uses log-mel + MFCC embeddings; restart the server to retrain after adding data.
- Synthetic coughs include dry, wet, and wheeze variants to improve detection diversity.
