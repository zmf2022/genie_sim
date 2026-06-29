#!/usr/bin/env bash
# Drop into a shell inside the running GenieSim container.
set -eo pipefail
NAME="${GENIESIM_CONTAINER:-geniesim}"
exec docker exec -it \
    -u "$(id -u):$(id -g)" \
    -e HOME=/home/isaac-sim \
    -w /workspace \
    "${NAME}" bash -l
