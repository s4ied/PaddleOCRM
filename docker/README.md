# Mobile OCR (detection + recognition only)

Runs [run_mobile_ocr.py](run_mobile_ocr.py) in a container with the lightweight
`PP-OCRv5_mobile` detection and recognition models. Document orientation,
unwarping and textline orientation are disabled, so it is text detection plus
recognition only.

Files in this folder:

| File | Purpose |
| --- | --- |
| `Dockerfile.mobile-ocr` | CPU image with `paddlepaddle` + `paddleocr` in a venv at `/opt/venv` |
| `docker-compose.mobile-ocr.yml` | Dev setup: mounts this folder at `/app`, `../images` at `/data`, caches models in a volume |
| `run_mobile_ocr.py` | The CLI entry point |

The build context is this `docker/` folder, so run every command from here:

```bash
cd docker
```

## Run with Docker Compose

```bash
docker compose -f docker-compose.mobile-ocr.yml build
docker compose -f docker-compose.mobile-ocr.yml run --rm ocr \
    python run_mobile_ocr.py /data/en_1.png -l en -o /data/out
```

The compose service overrides the entrypoint with `/bin/bash`, so you can also
open a shell and iterate:

```bash
docker compose -f docker-compose.mobile-ocr.yml run --rm ocr
# inside the container
python run_mobile_ocr.py /data/book.jpg -l de -o /data/out --save-images

# Fine-tuned German recognition model (../train/versions is mounted at /models)
python run_mobile_ocr.py /data/book.jpg -l de -o /data/out \
    --rec-model-dir /models/latin_de_rec_model_1/latin_de_rec_infer
```

This folder is bind mounted at `/app`, so edits to `run_mobile_ocr.py` on the
host take effect immediately — no rebuild needed.

Clean up the model cache volume when you no longer need it:

```bash
docker compose -f docker-compose.mobile-ocr.yml down -v
```

## Run with plain Docker

```bash
docker build -f Dockerfile.mobile-ocr -t mobile-ocr .
docker run --rm \
    -v ocr-models:/root/.paddlex \
    -v "$(pwd)/../images:/data" \
    mobile-ocr /data/en_1.png -l en -o /data/out
```

The image entrypoint is the script itself, so arguments are passed directly.

## CLI options

```
run_mobile_ocr.py INPUT [-l LANG] [-o OUTPUT] [--device DEVICE]
                  [--save-images] [--mkldnn | --no-mkldnn]
                  [--cpu-threads N]
```

| Option | Default | Notes |
| --- | --- | --- |
| `INPUT` | — | Image file, PDF, or directory of images |
| `-l, --lang` | `de` | One of `ch`, `de`, `en`, `latin` — selects the recognition model |
| `-o, --output` | `./output` | JSON results are written here, one file per input |
| `--device` | auto | e.g. `cpu`, `gpu:0` |
| `--save-images` | off | Also write box visualisations |
| `--mkldnn / --no-mkldnn` | off on ARM, on elsewhere | oneDNN segfaults in the aarch64 CPU build of Paddle |
| `--cpu-threads` | `2` | Threads used by the Paddle CPU runtime |

Per line, the script prints the confidence score, the top-left corner of the
box and the recognised text; the full result is saved as JSON.

## Notes

- The image is CPU-only. For GPU you need a CUDA base image and the matching
  `paddlepaddle-gpu` wheel.
- Model weights are downloaded on first run into `/root/.paddlex`. Keeping the
  `ocr-models` volume avoids re-downloading them.
- On Apple Silicon the container runs as `linux/arm64`; oneDNN stays off by
  default for that reason.
- Write outputs to `/data/...` (or another mount) so they land on the host.
