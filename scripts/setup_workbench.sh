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

# Copy the data from GCS bucket
gsutil -m cp -r gs://swarm-guard-dataset/* ~/data/

# Create a models directory
mkdir -p ~/models

# Mount the models bucket
fusermount -u ~/models
gcsfuse \
  --stat-cache-ttl 0s \
  --type-cache-ttl 0s \
  --implicit-dirs \
  swarm_guard_models ~/models