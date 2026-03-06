
## How things are modified
* `Dockerfile`
    - Commented out GPU related commands.
    - As we only focus on transformers, also commented out commands for other workloads.
* `compose.yaml`
    - Deleted GPU related fields.
    - Mount the `step_cpu` repo instead (this will mirror any changes made to `step_cpu` to the container).
    
## How to build

```bash
# Move this to the darpa-mocha repo (replace the prior Dockerfile)
cd /path/to/darpa-mocha/
mv Dockerfile Dockerfile_org
mv compose.yaml compose_org.yaml
cp /path/to/step_cpu/Docker/Dockerfile /path/to/darpa-mocha
cp /path/to/step_cpu/Docker/compose.yaml /path/to/darpa-mocha

# This will save the build log to compose-build.log in detail.
docker compose build --progress=plain --no-cache 2>&1 | tee compose-build.log
```

## How to run
### Foreground
```
# runs it in foreground and deletes the container once exited.
docker compose run --rm -e UID=$(id -u) -e GID=$(id -g) mocha
```

### Background
```bash
# To run it in background and keep it running:
cd /path/to/darpa-mocha/
docker compose run -d \
  --name mocha-bg \
  -e UID=$(id -u) -e GID=$(id -g) \
  mocha
# Enter it later
docker exec -it mocha-bg bash
# Stop it
docker stop mocha-bg
# Remove it
docker rm mocha-bg
```

## Once in the container
```
source mochaenv/bin/activate
```

## Content mirrored
As we used `--it` and the `compose.yaml`,
```
volumes:
  - ./step_cpu:/home/ginasohn/step_cpu
```
the contents in these locations are mirrored.

Edits on the host in ./step_cpu show up immediately in the container.
Edits in the container under /home/ginasohn/step_cpu write back to the host