# News Category Classifier: Web App

A deployed inference service for the [BBC News classifier](https://github.com/Mubydeji/bbc-news-classifier)
fine-tuned in a companion notebook. FastAPI backend, static HTML/CSS/JS frontend,
containerized and deployed to Render. Model weights are hosted on Hugging Face Hub
and downloaded at container startup.

**Status:** not deployed yet. Run it locally with the steps below.

**Note:** in the companion notebook's four-way comparison, a simple TF-IDF baseline
slightly outperformed this fine-tuned DistilBERT model on the held-out test set
(98.80% vs 97.90%). This app deploys DistilBERT specifically because it demonstrates
the fine-tuning and deployment pipeline; see the [training repo](https://github.com/Mubydeji/bbc-news-classifier)
for the full comparison and that honest result.

## Architecture

```
Browser (static frontend)
        │  POST /classify { text }
        ▼
FastAPI app (Render, Docker web service)
        │  downloads weights from Hugging Face Hub at startup
        ▼
DistilBERT (CPU inference)
        │
        ▼
{ predicted_category, probabilities }
```

## Running locally

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

export MODEL_REPO_ID="muby26/bbc-news-distilbert"
export MODEL_FILENAME="distilbert_news_classifier.pt"

uvicorn app.main:app --reload
```

Visit `http://localhost:8000`.

## Deploying to Render

1. Push this repo to GitHub.
2. On Render: **New +** → **Web Service** → connect this repo.
3. Render auto-detects the `Dockerfile`. Select the **Free** instance type.
4. Add environment variables:
   - `MODEL_REPO_ID` = `muby26/bbc-news-distilbert`
   - `MODEL_FILENAME` = `distilbert_news_classifier.pt`
5. Click **Create Web Service**.

## Limitations

Research prototype trained on a single, relatively clean news dataset (BBC News
Archive, 2,225 articles). Predictions on text very different in style or topic from
BBC news reporting (informal text, non-English, highly technical jargon, etc.) are
untested and may be unreliable. See the training repo for full methodology and
stated limitations.
