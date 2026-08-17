#!/usr/bin/env bash
# Build the full site into _deploy/ from scratch and serve it locally.
# _deploy/ is a throwaway build artifact (gitignored) — never edit or serve it
# directly; CI rebuilds it the same way in .github/workflows/deploy.yml.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf _deploy

(cd blog && bundle exec jekyll build \
  --config _config.yml,_config_local.yml \
  --destination ../_deploy/blog)

cp index.html config.js _deploy/
cp -r images reimagenet _deploy/

PORT="${1:-8000}"
echo "Hub  → http://localhost:${PORT}/"
echo "Blog → http://localhost:${PORT}/blog/"
python3 -m http.server "$PORT" --directory _deploy
