#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
wasm_version=$(python3 - <<'PY'
import tomllib
with open('Cargo.lock', 'rb') as f:
    lock = tomllib.load(f)
print(next(p['version'] for p in lock['package'] if p['name'] == 'wasm-bindgen'))
PY
)
if ! command -v wasm-bindgen >/dev/null || [[ "$(wasm-bindgen --version)" != "wasm-bindgen $wasm_version" ]]; then
  echo "Install the matching tool: cargo install wasm-bindgen-cli --version $wasm_version --locked"
  exit 1
fi
cargo build --locked --profile web-release --target wasm32-unknown-unknown --features web --bin map_generation_lab
mkdir -p dist/pkg dist/assets/shaders dist/assets/models
wasm-bindgen --target web --no-typescript --out-dir dist/pkg "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/web-release/map_generation_lab.wasm"
cp web/index.html web/app.js web/style.css dist/
python3 scripts/web_notices.py
cp assets/shaders/vegetation_wind.wgsl dist/assets/shaders/
for model in Horse Horse_White Deer Alpaca Wolf; do
  cp "assets/models/$model.gltf" dist/assets/models/
done
touch dist/.nojekyll
echo "Built dist/. Preview: python3 -m http.server 8080 --directory dist"
