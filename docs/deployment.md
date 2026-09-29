# Deployment guide

## Local / demo

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker (single container)

```bash
docker build -t nexashop .
docker run -d -p 8000:8000 --name nexashop nexashop
```

The image ships the processed dataset + trained artifacts, so startup is
seconds. Persist live data by mounting a volume for `app.db`:
`-v nexadata:/srv/app` keeps SQLite between restarts.

## Render.com / Railway / Fly.io (free tiers)

1. Push this repo to GitHub.
2. Create a Web Service from the repo — it auto-detects the Dockerfile.
3. Port `8000`, health check `/api/products` (requires login; use `/` instead
   which serves the UI unconditionally).
4. Attach a persistent disk for `/srv/app/app.db` if the platform offers one.

## AWS EC2 (cheap real-environment demo)

```bash
sudo apt install docker.io -y
git clone <your repo> && cd nextgen-recsys
sudo docker build -t nexashop .
sudo docker run -d -p 80:8000 --restart always nexashop
```

Open the instance's public IP on port 80 (allow inbound TCP 80 in the
security group).

## Scaling notes (for the paper's "scalability" section)

- Model artifacts are read-only → run N API replicas behind a load
  balancer; only SQLite writes are stateful (swap for Postgres in prod).
- Item-item neighbor lists and rules are precomputed → per-request cost
  is O(user items × neighbors), a few ms.
- Retrain nightly via `scripts/train.py` as a cron job.
