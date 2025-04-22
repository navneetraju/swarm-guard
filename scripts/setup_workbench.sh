# First clone the repo
git clone https://github.com/navneetraju/swarm-guard.git
cd swarm-guard

# Create a conda virtual environment
conda create -n swarm_guard python=3.9
conda activate swarm_guard

# Install the required packages
pip install -r requirements.txt

# Create a data directory
mkdir ~/data

# Copy the data from GCS bucket, preserving directory structure
gsutil -m rsync -r gs://swarm_guard_dataset/ ~/data/

# Create a models directory
mkdir -p ~/models

# Mount the models bucket
fusermount -u ~/models || true

gcsfuse \
  --implicit-dirs \                     # allow “directory” semantics in a flat namespace
  --file-mode 0777 \                    # make files world‑writable (adjust as needed) :contentReference[oaicite:0]{index=0}
  --dir-mode 0777 \                     # same for directories :contentReference[oaicite:1]{index=1}
  --metadata-cache-ttl-secs 0 \         # disable metadata caching so stat/type changes propagate immediately :contentReference[oaicite:2]{index=2}
  --kernel-list-cache-ttl-secs 0 \      # disable directory‐listing cache for up‑to‑date listings :contentReference[oaicite:3]{index=3}
  swarm_guard_models ~/models