rm -r ~/swarm_guard_dataset
mkdir -p ~/swarm_guard_dataset
fusermount -u ~/swarm_guard_dataset
gcsfuse \
  --stat-cache-ttl 0s \
  --type-cache-ttl 0s \
  --implicit-dirs \
  swarm_guard_dataset ~/swarm_guard_dataset