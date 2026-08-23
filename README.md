# Aletheia — AI Digital Deception & Abuse Forensics Platform

Aletheia is a forensic investigation platform for images, video, audio, screenshots, documents and abuse-related text. It is designed around **evidence integrity, real ML inference, explainability and human verification**. It never fabricates a model result: if a required trained detector checkpoint is unavailable, that analysis is reported as unavailable rather than converted into a fake probability.

## Architecture
- **Frontend:** React + Vite + Tailwind CSS
- **API:** FastAPI + SQLAlchemy + PostgreSQL
- **Async jobs:** Celery + Redis
- **ML tracking:** MLflow
- **Evidence graph:** NetworkX (portable; Neo4j can be added later)
- **Forensics:** SHA-256, ExifTool/Pillow, PyMuPDF, C2PA adapter, FFmpeg
- **OCR:** PaddleOCR adapter
- **Reports:** ReportLab
- **ML:** PyTorch, torchvision, Transformers, torchaudio

## Important model-integrity rule
The repository contains real model architectures and training/inference pipelines. Large datasets and trained checkpoints are intentionally not bundled. The API refuses to invent detector scores when a task-specific checkpoint is absent.

## Quick start

### Docker
```bash
docker compose up --build
```
Frontend: http://localhost:5173  
API: http://localhost:8000/docs  
MLflow: http://localhost:5000

### Local backend
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Configuration
Copy `.env.example` to `.env` and set database/storage locations. Model paths are configured under `configs/models.yaml`.

## Evidence workflow
1. Create incident.
2. Upload original evidence.
3. Aletheia hashes the exact bytes with SHA-256 and records custody metadata.
4. Select analysis modules.
5. Run real model inference or forensic extraction.
6. Review evidence, explanations, graph and timeline.
7. Human investigator verifies/rejects findings.
8. Generate a forensic report.

## Dataset setup
The project deliberately does not redistribute restricted datasets. Use the official dataset providers and comply with their terms. The included scripts prepare local datasets into a stable format.

Suggested research datasets:
- FaceForensics++
- Celeb-DF / Celeb-DF++
- DFDC
- ASVspoof 2021 DF
- task-appropriate multilingual abuse/toxicity datasets for English/Hindi/Hinglish

Put source data under the configured `DATASET_DIR`, then run:
```bash
python scripts/prepare_datasets.py --task image
python scripts/verify_datasets.py
```

## Training
Development runs are deliberately small but still train the actual model architectures.
```bash
python training/train_image.py --config configs/image_train.yaml
python training/train_audio.py --config configs/audio_train.yaml
python training/train_video.py --config configs/video_train.yaml
python training/train_nlp.py --config configs/nlp_train.yaml
python training/train_all.py
```

After training, checkpoints are placed under `models/`. Update `configs/models.yaml` if needed. Run evaluation:
```bash
python evaluation/evaluate_all.py
```

## API examples
```bash
curl -X POST http://localhost:8000/incidents -H 'Content-Type: application/json' -d '{"title":"Example investigation","description":"Synthetic demonstration"}'
```

Upload evidence with multipart form data to `/evidence/upload` and then call an analysis endpoint using the returned evidence ID.

## Model evaluation and leakage prevention
Image/video splits are source-aware. Video frames from the same source are never independently randomized into train/test. Classification reports include accuracy, precision, recall, F1, ROC-AUC and PR-AUC where applicable. Speaker verification reports EER/FAR/FRR when the relevant evaluation set exists.

## Limitations
- Detector performance is dataset/model dependent and should not be treated as legal certainty.
- No C2PA credential is **not** evidence that content is fake.
- Speaker similarity is not proof of identity.
- Generator attribution is expressed as a possible family, not a certainty.
- AI findings are separate from human-verified findings.

## Ethics
Use only with lawful authority and appropriate consent. Protect victim data, minimize retention, restrict access, and preserve original evidence. Never use an automated score as the sole basis for a legal or disciplinary decision.

## License
MIT for the original Aletheia source code. Third-party model/dataset licenses remain applicable to their respective assets.

## Real media models

Aletheia no longer requires a manually supplied `best.pt` checkpoint for its core image/audio media lab. The application lazily downloads and caches two real research checkpoints on first AI analysis:

- `buildborderless/CommunityForensics-DeepfakeDet-ViT` for AI-generated image screening.
- `Vansh180/deepfake-audio-wav2vec2` for speech spoof/deepfake screening.

The cache is persisted in `models/hf_cache` through Docker. If you want to prefetch them before using the UI, run from the project root:

```bash
python scripts/download_models.py
```

The first download requires internet access and can take time because the audio checkpoint is large. Model outputs are screening findings, not legal certainty.

Video uses the real image detector across multiple sampled frames and reports frame-level probabilities and suspicious timestamps; it is intentionally described as a multi-frame ensemble rather than being mislabeled as a temporal neural network.

## GitHub-safe model caching

Aletheia does **not** store AI model weights in Git. Model weights are downloaded lazily on first inference and cached in a persistent host directory.

### Recommended Windows setup

For a cache shared across project clones, create a local `.env` and set:

```env
ALETHEIA_MODEL_CACHE=C:/AletheiaData/models
ALETHEIA_STORAGE_DIR=C:/AletheiaData/storage
```

These directories are intentionally outside the Git repository. Docker mounts the model cache at `/app/models`, and Hugging Face/Transformers reuse the same cache across container rebuilds and `docker compose down` / `up` cycles.

If you do not set the variables, the default is `./.local/model-cache` and `./storage`, both excluded from Git.

### Start from Git

```powershell
git clone <YOUR_REPOSITORY_URL>
cd Aletheia
git pull
Copy-Item .env.example .env
# Optionally edit .env to use C:/AletheiaData/models

docker compose up --build
```

**Do not run `docker compose down -v` unless you intentionally want to delete Docker-managed persistent data.**

### Check model cache size

From the project root:

```powershell
python scripts/model_cache_status.py
```

Inside the backend container:

```powershell
docker compose exec backend python scripts/model_cache_status.py
```

### Clear downloaded models when you need the disk space

```powershell
python scripts/clear_model_cache.py
```

Clearing the cache does not delete your source code. Models will be downloaded again only when an analysis requires them.

### Git rule

Never commit `.env`, datasets, generated evidence, `node_modules`, or model weights (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.onnx`). The included `.gitignore` protects these paths.

## Docker dependency integrity

The backend Docker image explicitly disables pip's `--require-hashes` mode for this un-hashed requirements file. Package versions/ranges remain controlled by `requirements.txt`; the build does not silently replace or rewrite package hashes. This avoids stale hash failures for transitive dependencies while preserving normal HTTPS package verification.

## Persistent AI model cache

Set `ALETHEIA_MODEL_CACHE` to a host directory such as `C:/AletheiaData/models` on Windows. Hugging Face model files are downloaded once and reused across container rebuilds/restarts. Keep this directory outside Git.
