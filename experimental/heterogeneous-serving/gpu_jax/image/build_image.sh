#!/bin/bash
# Builds the JAX-on-GPU serving image without docker: install the locked
# environment into /opt/venv on top of the python base image, then append it as
# one layer with crane and push to Artifact Registry.
# Runs inside a CPU pod on the cluster's default-pool (pod `image-build`, namespace gpu-jax) with:
#   /work/requirements.lock  (uv pip compile output, index URLs embedded)
#   $AR_TOKEN                 OAuth access token with Artifact Registry write
#   $BASE $TARGET             base image (digest) and target image reference
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
mark() { echo "MARK $1 $(date +%s.%N)"; }

mark build_start
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null
curl -sSfL https://astral.sh/uv/0.10.2/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh >/dev/null
curl -sSfL https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C /usr/local/bin crane
mark tools_ready

uv venv /opt/venv --python /usr/local/bin/python3.12 --seed -q
VIRTUAL_ENV=/opt/venv uv pip install -q --no-cache --index-strategy unsafe-best-match \
  --prerelease=if-necessary-or-explicit -r /work/requirements.lock
mark venv_installed

echo "== uv pip check"; VIRTUAL_ENV=/opt/venv uv pip check
echo "== key versions"
VIRTUAL_ENV=/opt/venv uv pip freeze | grep -E '^(vllm-tpu|tpu-inference|jax|jaxlib|jax-cuda13-[a-z]+|jax-triton|triton|tokamax|flax|torch|torchax|libtpu|ray)=='
echo "== venv size"; du -sh /opt/venv

tar -C / -cf - opt/venv | gzip -1 > /tmp/venv-layer.tar.gz
ls -l /tmp/venv-layer.tar.gz
mark layer_packed

crane auth login asia-southeast1-docker.pkg.dev -u oauth2accesstoken -p "$AR_TOKEN"
crane append --platform linux/amd64 -b "$BASE" -f /tmp/venv-layer.tar.gz -t "$TARGET.tmp"
crane mutate "$TARGET.tmp" -t "$TARGET" \
  --env PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
  --env VIRTUAL_ENV=/opt/venv \
  --label org.opencontainers.image.description="vllm-tpu 0.32.0 + tpu-inference 0.32.0 + jax[cuda13] 0.11.2 (hetero-research research/gpu-jax)"
echo "== pushed $TARGET @ $(crane digest "$TARGET")"
mark build_done
