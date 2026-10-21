# CropSense

Open-source crop disease advisor for smallholder farmers in Nigeria (MIT license).

A farmer opens the site on a phone, photographs a sick leaf, picks a language (English, Hausa, Yoruba, Igbo, or Nigerian Pidgin), and gets the most likely disease, a confidence score, the top three alternatives, and short advice. Treatment text comes only from retrieved knowledge-base passages — the language model is not allowed to invent pesticide names, doses, or schedules.

## How it works

1. **Vision.** A fine-tuned MobileNetV3 (`timm`, PyTorch) predicts a class from `models/labels.json`.
2. **Retrieval.** Markdown notes in `app/knowledge/` are embedded with Ollama (`nomic-embed-text`) and stored in ChromaDB.
3. **Advice.** Gemma 4 (model name from `LLM_MODEL`, default `gemma4:e4b`) writes in the selected language using only those passages.
4. **Safety.** If top-1 confidence is below `CONFIDENCE_THRESHOLD` (default 0.60), CropSense does not diagnose. It asks for a clearer photo and points the farmer to a local extension officer. Every response includes a decision-support disclaimer.

Starter crops: tomato, maize, cassava, pepper.

## Setup

Python 3.10+ and a local [Ollama](https://ollama.com) server.

```powershell
cd cropsense
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On macOS/Linux, use `source .venv/bin/activate` and `cp .env.example .env`.

Pull the LLM and embedding models (names must match `.env`):

```powershell
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

Index the knowledge base (needs Ollama running):

```powershell
python scripts/build_knowledge_base.py
```

Train the classifier after you have images in `data/processed/train/<class_id>/`:

```powershell
python scripts/download_data.py
python scripts/train.py
```

`download_data.py` creates the class folders and will try Kaggle if the Kaggle CLI is installed. Otherwise copy PlantVillage tomato/maize/pepper images and iCassava images into those folders. Class ids are listed in `models/labels.json`.

Until `models/classifier.pt` exists, `/api/diagnose` returns a clear error. The UI and `/health` still run.

Start the app:

```powershell
python run.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) on your phone (same Wi‑Fi) using the machine’s LAN IP if needed (`HOST=0.0.0.0` in `.env`).

## API

- `GET /health` — classifier weights, Ollama, and knowledge-base status.
- `POST /api/diagnose` — multipart form: `image`, `language` (`en`, `ha`, `yo`, `ig`, `pcm`).

JSON shape:

```json
{
  "status": "ok",
  "predictions": [{"label": "Tomato Early Blight", "confidence": 0.87}],
  "advice": "...",
  "language": "en",
  "disclaimer": "This is decision support..."
}
```

`status` is `ok`, `low_confidence`, or `error`.

## Add a disease

1. Add a class object to `models/labels.json` (`id`, `display`, `crop`, `kb_file`).
2. Drop `app/knowledge/<kb_file>` with symptoms, treatment options, and prevention. Do not write unregistered product doses.
3. Run `python scripts/build_knowledge_base.py`.
4. Add images under `data/processed/train/<id>/` and retrain.

## Tests

```powershell
pytest
```

## Project layout

```
app/            Flask app, classifier, advisor, knowledge markdown, UI
scripts/        download_data.py, train.py, build_knowledge_base.py
models/         labels.json; classifier.pt after training
data/           raw images, processed splits, Chroma store
tests/          pytest
```

Weights (`*.pt`) are gitignored. Do not commit secrets; keep them in `.env`.
