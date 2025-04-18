# ~/models exists delete it
if [ -d ~/models ]; then
  rm -rf ~/models
fi
mkdir -p ~/models
fusermount -u ~/models
gcsfuse \
  --stat-cache-ttl 0s \
  --type-cache-ttl 0s \
  --implicit-dirs \
  swarm_guard_models ~/models