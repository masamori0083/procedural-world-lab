# Assets

Trees, grass, shrubs and ford stones are generated in Rust. `src/plant_shapes.rs` provides the leaf mask, leaf-card layout and curved grass ribbons used by the map. One shared 256×256 leaf texture and a 128×64 grass coverage mask are generated in memory; no external vegetation models or image files are needed.

Broadleaf trees and riverside bushes use alpha-masked leaf cards. Conifers use feathered green boughs, a dense crown interior and an exposed lower stem. Seeded shapes and placement stay consistent across detail levels.

`shaders/vegetation_wind.wgsl` applies gentle wind using Bevy's GPU clock in drawing and shadow passes. Broadleaf foliage, shrubs and tall grass sway; conifers, trunks and roots stay fixed. Grass uses six curved blades per Near tuft and three blades on every second Middle tuft in streaming mode. Independent 16m patches gradually switch detail; distant blades disappear through coverage masking. Simple shadow proxies fade over 6–10m.

Grass uses diffuse sunlight, ambient light and modest backlighting, sharing shadow results between its two sides. Local lights or light probes restore standard PBR lighting. Other vegetation retains standard PBR lighting. Camera movement does not rebuild stable grass geometry or rebind materials.

## Animal models

Place the following files in `assets/models/` before starting the exploration window:

- `Horse.gltf`
- `Horse_White.gltf`
- `Deer.gltf`
- `Alpaca.gltf`
- `Wolf.gltf`

Use self-contained glTF files with embedded buffers and textures. Models are kept locally and are not included in this repository or the source archive.

A possible source is [Quaternius Ultimate Animated Animal Pack](https://quaternius.com/packs/ultimateanimatedanimals.html), listed as CC0 on its official page. The local files have not been matched against the original download package; verify the source of the models you obtain.
