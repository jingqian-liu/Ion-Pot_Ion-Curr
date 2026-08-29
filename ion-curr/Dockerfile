# CPU-only torch: GlobalContextPredictor is a tiny 3D CNN (~170KB checkpoint),
# so CPU inference is fast and this avoids torch's multi-GB CUDA wheel stack.
# For GPU inference instead, drop the --index-url below and use `docker run --gpus all`.
FROM python:3.11-slim

WORKDIR /app

# Only the packages inference.py actually imports
# (train.py's extra deps - scikit-learn, tqdm, psutil - are intentionally left out).
RUN pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    pandas==3.0.5 \
    MDAnalysis==2.9.0 \
    scipy==1.14.1

COPY inference.py IonCurr.py ./
COPY param/ ./param/

# checkpoint/ (model_best.pth) and any --input data are intentionally NOT
# baked into the image. Mount them at runtime, e.g.:
#
#   docker run --gpus all \
#     -v /path/to/checkpoint:/app/checkpoint \
#     -v /path/to/input_dir:/app/input \
#     -v /path/to/output_dir:/app/output \
#     ion-curr-inference \
#     --input /app/input --output /app/output/preds.csv --voltage 0.4

ENTRYPOINT ["python", "inference.py"]
CMD ["--help"]
