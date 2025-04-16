rm -r ~/swarm_guard_models
mkdir -p ~/swarm_guard_models
fusermount -u ~/swarm_guard_models
gcsfuse \
  --stat-cache-ttl 0s \
  --type-cache-ttl 0s \
  --implicit-dirs \
  swarm_guard_models ~/swarm_guard_models