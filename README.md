# Procedural World Lab

A study project for procedural terrain, rivers, and vegetation using Rust and Bevy. Explore the map as a horse, with peaceful animals roaming nearby.

## Setup

Install Rust and Cargo, then clone the repository:

```sh
git clone https://github.com/masamori0083/procedural-world-lab.git
cd procedural-world-lab
```

Animated horse, deer, alpaca and wolf models are included for the browser demo and desktop app. Vegetation shapes and textures are generated in Rust. See [asset details](assets/README.md).

## Browser demo

[Try the map in your browser](https://masamori0083.github.io/procedural-world-lab/). Click **Explore**, then click the map to capture the mouse. A desktop browser with WebGPU and hardware acceleration is required. Use the seed field to generate another world.

The browser keeps nine chunks nearby, compared with 49 on desktop. Generation and LOD jobs yield to the browser between small batches, aiming for 2 ms of work per slice. The 3D view renders at CSS-pixel resolution, capped at 1280×720; the HUD retains the canvas resolution. Animal models and animations are shared with the desktop app.

The status line shows FPS, generated mesh estimates, the last two seconds' frame p95, and the largest generation slice since the previous status update. These are elapsed times, not GPU timings or CPU utilization. Process memory and GPU usage are unavailable in the browser.

To build and preview locally:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.126 --locked
scripts/build_web.sh
python3 -m http.server 8080 --directory dist
```

Open `http://localhost:8080`. The GitHub Actions workflow builds and deploys `dist/` to GitHub Pages after a push to `main`. The CLI version must match the wasm-bindgen version in `Cargo.lock`. Third-party notices for the distributed browser binary are collected during the build.

## Run

Run commands from the project root. The first build may take a while.

```sh
# Generate new chunks as you explore.
cargo run --locked -- --streaming --seed 42

# Explore the finite map.
cargo run --locked -- --seed 42
```

Click inside the window to control the horse. Close the window to quit.

Streaming mode keeps up to 49 terrain chunks around the horse. Rivers widen downstream and flow into lakes; shallow fords have small stone markers on their banks. Dry sand and gravel are walkable, while deep water blocks movement. Tree detail changes per tree, and grass detail changes in 16m patches. Neighboring detail levels fade gradually, with geometry cached ahead of time. Grass, broadleaf leaves, and shrubs sway gently; conifers stay still. Vegetation shapes and shared textures are generated in Rust.

The resource dashboard shows CPU usage, process memory, frame rate and resident mesh estimates. Mesh estimates exclude ECS and GPU allocations; unavailable GPU measurements display `N/A`.

## Controls

| Action | Input |
|---|---|
| Move forward / backward | W / S |
| Turn | A / D, mouse left / right |
| Look up / down | Mouse up / down |
| Run | Shift + W |
| Switch first-person / follow camera | V |
| Switch overhead / follow camera | B |
| Adjust follow distance / overhead height | Mouse wheel |
| Return to the starting point | R |
| Move to a riverbank | L |
| Move to a ford entrance (streaming only) | G |
| Cycle exploration checkpoints (finite map only) | N |
| Toggle resource dashboard | F3 |
| Cycle natural / moisture / tree density / rockiness views | F4 |
| Pause and release the mouse | Esc |
| Resume | Click inside the window |

## Settings

```sh
cargo run --locked -- --streaming --seed 42 \
  --mountain-strength 0.85 --tree-variation 0.65 \
  --forest-uniformity 0.90 --landmark-strength 1.0
```

| Option | Purpose |
|---|---|
| `--seed` | Choose the map seed (default: 20261003). |
| `--mountain-strength` | Adjust mountain height and steepness. |
| `--tree-variation` | Adjust differences between individual trees. |
| `--forest-uniformity` | Adjust similarity among nearby trees. |
| `--landmark-strength` | Emphasize isolated trees in streaming mode. |
| `--no-vegetation-wind` | Disable foliage and grass movement. |
| `--no-vegetation-lod` | Keep full vegetation detail for comparison. |
| `--no-terrain-lod` | Keep full ground detail for comparison. |
| `--soft-grass-shadows` | Use softer, more expensive grass shadow filtering. |

All numeric settings except the seed accept values from 0 to 1.

## Checks and reports

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings

# Generation reports need no models or window.
cargo run --locked -- --seed 42 --generation-report
cargo run --locked -- --streaming --seed 42 --generation-report

# Exploration checks need the animal models.
cargo run --locked -- --seed 42 --smoke-test
cargo run --locked -- --seed 42 --stream-smoke-test

# Hold the actual map and measure a stationary camera rotation.
cargo run --locked -- --streaming --seed 42 --map-profile \
  --profile-pose grass --profile-case all --profile-rotate --profile-label grass
```

Map profiles use a settled map, a fixed animation clock and a GPU completion wait. Their timings measure serialized rendering cost, not interactive FPS or GPU pass timestamps. Run comparisons sequentially. `--profile-case pbr-grass` restores standard grass lighting for comparison; `no-grass` hides grass. `--profile-pose` also accepts `leaves`, `far` and `grass-shadow`.

## File structure

```text
src/main.rs                  Application setup and command-line options
src/terrain.rs               Finite terrain and water boundaries
src/streaming.rs              Seeded chunks, scheduling, installation and eviction
src/watershed.rs              Shared river and lake drainage plans
src/environment.rs           Habitats and vegetation placement
src/trees.rs                 Procedural tree shapes and detail levels
src/grassfield.rs             Seeded meadow patches
src/plant_shapes.rs           Shared generated leaf and grass geometry
src/vegetation.rs             Tree and grass LOD caches
src/wind.rs                   Vegetation materials and wind settings
src/web.rs                    Browser seed and loading status
src/work_budget.rs            Cooperative browser work and render pixel budget
web/                          Browser page and controls
scripts/build_web.sh          Build and package the browser demo
scripts/web_notices.py        Collect bundled third-party notices
.github/workflows/web.yml     GitHub Pages build and deployment
src/map_profile.rs            Actual-map rendering comparisons
src/player.rs                 Horse movement and camera controls
src/animals.rs                Peaceful wildlife
assets/shaders/               Vegetation rendering shader
assets/models/                Shared animated animal models
assets/README.md              Asset setup and source details
Cargo.toml                    Rust package and dependencies
Cargo.lock                    Locked dependency versions
```

## Output directories

| Path | Contents |
|---|---|
| `reports/` | Generation, terrain/vegetation LOD, exploration and rendering CSV reports. |
| `captures/seed-<seed>/` | Finite-map screenshots. |
| `captures/stream-seed-<seed>/` | Streaming screenshots. |
| `captures/map-profile/` | Rendering comparison images. |
| `target/` | Cargo build output. |
| `dist/` | Browser build, assets and third-party notices; excluded from source control. |
