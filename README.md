# Procedural World Lab

A study project for procedural terrain, rivers, and vegetation using Rust and Bevy. Explore the map as a horse, with peaceful animals roaming nearby.

## Setup and assets

Install Rust and Cargo, then clone the repository:

```sh
git clone https://github.com/masamori0083/procedural-world-lab.git
cd procedural-world-lab
```

Animal models are not included. Place these files in `assets/models/`:

- `Horse.gltf`
- `Horse_White.gltf`
- `Deer.gltf`
- `Alpaca.gltf`
- `Wolf.gltf`

Use self-contained glTF files with embedded buffers and textures. See [asset details](assets/README.md) for setup and source information.

## Run

Run these commands from the project root. The first build may take a while.

```sh
# Explore a finite map.
cargo run --locked

# Generate new chunks as you explore.
cargo run --locked -- --streaming --seed 42
```

Click inside the window to control the horse. Close the window to quit.

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

All settings except the seed accept values from 0 to 1.

## Reports and captures

```sh
# Generate CSV reports without models or a window.
cargo run --locked -- --seed 42 --generation-report
cargo run --locked -- --streaming --seed 42 --generation-report

# Run exploration checks and save screenshots. Requires the animal models.
cargo run --locked -- --seed 42 --smoke-test
cargo run --locked -- --seed 42 --stream-smoke-test

```

## Project structure

```text
src/                         Terrain, rivers, vegetation, horse controls, and rendering
assets/models/               Locally supplied animal models
assets/README.md             Asset setup and source details
Cargo.toml                   Rust package and dependencies
Cargo.lock                   Locked dependency versions
```

## Output directories

| Path | Contents |
|---|---|
| `reports/seed-<seed>.csv` | Finite-map generation report. |
| `reports/stream-seed-<seed>.csv` | Streaming generation report. |
| `captures/seed-<seed>/` | Finite-map screenshots. |
| `captures/stream-seed-<seed>/` | Streaming screenshots. |
| `target/` | Cargo build output. |
