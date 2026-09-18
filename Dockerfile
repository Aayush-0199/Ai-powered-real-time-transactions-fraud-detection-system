# ─────────────────────────────────────────────────────────────
# FraudGuard AI — Production Dockerfile
# Build: docker build -t fraudguard .
# ─────────────────────────────────────────────────────────────

# Use the official slim Python image to keep the image small
FROM python:3.11-slim

# System-level dependencies for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# ── Step 1: Install PyTorch CPU-only first (significantly smaller than GPU) ──
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install PyTorch Geometric wheels for torch 2.1.0 + CPU ──
RUN pip install --no-cache-dir \
    torch_geometric==2.4.0 \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html || \
    pip install --no-cache-dir torch_geometric==2.4.0

# ── Step 3: Install the rest of the application dependencies ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# ── Step 4: Copy the full application source ──
COPY . .

# Render injects PORT env var; expose it
EXPOSE 5000

# ── Production start command ──
# gunicorn with 1 worker (models are heavy; 1 worker per dyno to save RAM)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "app:app"]
