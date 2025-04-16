if [ ! -d ~/swarm_guard_models ]; then
  rm -r ~/swarm_guard_models
mkdir -p ~/swarm_guard_models
fi
fusermount -u ~/swarm_guard_models
gcsfuse \
  --stat-cache-ttl 0s \
  --type-cache-ttl 0s \
  --implicit-dirs \
  models ~/swarm_guard_models
