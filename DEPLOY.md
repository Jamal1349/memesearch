# Docker deploy

## Files on the server

Create these directories near the project:

```bash
mkdir -p data artifacts
cp .env.example .env
```

Put persistent bot state into `./data` automatically via Docker volume.

If you already have local artifacts, place them in `./artifacts`:

- `clip.index`
- `clip_meta.npy`
- `clip_finetuned/` or another model path
- `meme_enriched/` if you use a local dataset copy

If `meme_enriched/` is absent, the bot can load the dataset from Hugging Face using `DATASET_NAME` and `DATASET_SPLIT`.

## Run

```bash
docker compose up -d --build
```

## Logs

```bash
docker compose logs -f memebot
```

## Update

```bash
git pull
docker compose up -d --build
```
