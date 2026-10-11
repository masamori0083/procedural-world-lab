# Assets

Trees, grass, shrubs and ford stones are generated in Rust. `src/plant_shapes.rs` provides the leaf mask, leaf-card layout and curved grass ribbons used by the map. One shared 256×256 leaf texture and a 128×64 grass coverage mask are generated in memory; no external vegetation models or image files are needed.

Broadleaf trees and riverside bushes use alpha-masked leaf cards. Conifers use feathered green boughs, a dense crown interior and an exposed lower stem. Seeded shapes and placement stay consistent across detail levels.

`shaders/vegetation_wind.wgsl` applies gentle wind using Bevy's GPU clock in drawing and shadow passes. Broadleaf foliage, shrubs and tall grass sway; conifers, trunks and roots stay fixed. Grass uses six curved blades per Near tuft and three blades on every second Middle tuft in streaming mode. Independent 16m patches gradually switch detail; distant blades disappear through coverage masking. Simple shadow proxies fade over 6–10m.

Grass uses diffuse sunlight, ambient light and modest backlighting, sharing shadow results between its two sides. Local lights or light probes restore standard PBR lighting. Other vegetation retains standard PBR lighting. Camera movement does not rebuild stable grass geometry or rebind materials.

## Animal models

The self-contained glTF models in `models/` are shared by the desktop application and browser demo:

- `Horse.gltf`
- `Horse_White.gltf`
- `Deer.gltf`
- `Alpaca.gltf`
- `Wolf.gltf`

They were supplied for this study project, and the owner confirmed their use for the browser distribution. The source pack is [Quaternius — Ultimate Animated Animal Pack](https://quaternius.com/packs/ultimateanimatedanimals.html), published under [CC0](https://creativecommons.org/publicdomain/zero/1.0/). See `models/ATTRIBUTION.txt`.

The browser build copies only these five models and the map shader. Standalone tree/forest/grass experiments, private working notes and local captures are excluded. Dependency and embedded-font redistribution notices are included in the browser output as `third-party-notices.txt`.
