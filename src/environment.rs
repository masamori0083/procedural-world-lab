//! Deterministic habitat estimates on the terrain grid; no weather simulation.
use crate::terrain::Meadow;
use bevy::prelude::*;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvironmentSample {
    pub elevation: f32,
    pub slope: f32,
    pub water_distance: f32,
    pub moisture: f32,
    pub tree_density: f32,
    pub grass_density: f32,
    pub rockiness: f32,
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl EnvironmentSample {
    pub(crate) fn evaluate(
        elevation: f32,
        relative_height: f32,
        slope: f32,
        water_distance: f32,
        soil: f32,
        forest_patch: f32,
    ) -> Self {
        let altitude = smooth((relative_height - 8.0) / 40.0);
        let steepness = smooth((slope - 0.25) / 0.65);
        let moisture = (0.24 + 0.30 * soil + 0.44 * (-water_distance / 24.0).exp()
            - 0.16 * altitude
            - 0.10 * steepness)
            .clamp(0.0, 1.0);
        let rockiness =
            smooth((slope - 0.40) / 0.70).max(smooth((relative_height - 20.0) / 35.0) * 0.85);
        // Flooded/eroded edge stays open; trees form a band on the moist bank.
        let riparian = (-((water_distance - 12.) / 16.).powi(2)).exp();
        let canopy = smooth((forest_patch - 0.22) / 0.55).max(riparian * 0.65);
        let treeline = smooth((relative_height - 45.) / 25.);
        let tree_density = (0.25 + 0.75 * moisture)
            * canopy
            * (1.0 - rockiness).powi(2)
            * (1.0 - altitude * 0.45)
            * (1. - treeline)
            * (1.0 - smooth((slope - 0.40) / 0.25))
            * smooth((water_distance - 2.) / 5.);
        let grass_density = (0.28 + 0.72 * moisture)
            * (1.0 - rockiness).powi(2)
            * (1.0 - altitude * 0.4)
            * (1.0 - smooth((slope - 0.50) / 0.25))
            * smooth((water_distance - 0.3) / 2.);
        Self {
            elevation,
            slope,
            water_distance,
            moisture,
            tree_density,
            grass_density,
            rockiness,
        }
    }

    pub fn at(world: &Meadow, p: Vec2) -> Self {
        let elevation = world.ground(p);
        let mut sample = Self::evaluate(
            elevation,
            elevation - world.ground(Vec2::ZERO),
            world.slope(p),
            world.water_distance(p),
            world.noise(p * 0.018 + Vec2::new(219.0, -53.0)),
            world.noise(p * 0.035 + Vec2::splat(31.0)),
        );
        // Keep the starting clearing open; forest patches follow regional habitat.
        sample.tree_density *= smooth((p.length() - 17.0) / 10.0)
            * (1.0 - smooth((p.length() - crate::terrain::PLAY_RADIUS) / 8.0));
        sample
    }

    pub fn color(self, layer: MapLayer, patch: f32) -> [f32; 4] {
        let color = match layer {
            MapLayer::Natural => {
                let dry = Vec3::new(0.32, 0.36, 0.13);
                let wet = Vec3::new(0.12, 0.32, 0.09);
                let grass = dry.lerp(wet, self.moisture) * (0.88 + patch * 0.24);
                let rock = grass.lerp(Vec3::new(0.42, 0.43, 0.38), self.rockiness * 0.9);
                rock.lerp(
                    Vec3::new(0.48, 0.39, 0.22),
                    1.0 - smooth(self.water_distance / 2.5),
                )
            }
            MapLayer::Moisture => {
                Vec3::new(0.47, 0.25, 0.07).lerp(Vec3::new(0.02, 0.48, 0.78), self.moisture)
            }
            MapLayer::ForestDensity => {
                Vec3::new(0.40, 0.29, 0.16).lerp(Vec3::new(0.07, 0.65, 0.13), self.tree_density)
            }
            MapLayer::Rockiness => {
                Vec3::new(0.13, 0.35, 0.10).lerp(Vec3::new(0.70, 0.70, 0.70), self.rockiness)
            }
        };
        [color.x, color.y, color.z, 1.0]
    }
}

#[derive(Resource, Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum MapLayer {
    #[default]
    Natural,
    Moisture,
    ForestDensity,
    Rockiness,
}

impl MapLayer {
    pub fn next(self) -> Self {
        match self {
            Self::Natural => Self::Moisture,
            Self::Moisture => Self::ForestDensity,
            Self::ForestDensity => Self::Rockiness,
            Self::Rockiness => Self::Natural,
        }
    }

    pub fn legend(self) -> &'static str {
        match self {
            Self::Natural => "NATURAL | terrain-based vegetation",
            Self::Moisture => "MOISTURE ESTIMATE | Brown: dry -> Blue: wet",
            Self::ForestDensity => "TREE DENSITY POTENTIAL | Brown: sparse -> Green: dense",
            Self::Rockiness => "ROCKINESS | Green: soil -> Gray: exposed rock",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_and_terrain_change_habitat_in_the_expected_direction() {
        let near = EnvironmentSample::evaluate(0.0, 0.0, 0.1, 8.0, 0.5, 0.7);
        let dry = EnvironmentSample::evaluate(0.0, 0.0, 0.1, 100.0, 0.5, 0.7);
        let steep = EnvironmentSample::evaluate(0.0, 0.0, 1.1, 8.0, 0.5, 0.7);
        let high = EnvironmentSample::evaluate(50.0, 50.0, 0.1, 8.0, 0.5, 0.7);
        assert!(near.moisture > dry.moisture + 0.2);
        assert!(near.tree_density > dry.tree_density);
        assert!(near.grass_density > dry.grass_density);
        assert!(high.rockiness > near.rockiness);
        assert!(high.tree_density < near.tree_density);
        assert_eq!(steep.tree_density, 0.0);
        assert_eq!(steep.grass_density, 0.0);
        let water = EnvironmentSample::evaluate(-2.0, 0.0, 0.1, 0.0, 0.5, 0.7);
        assert_eq!(water.tree_density, 0.0);
        assert_eq!(water.grass_density, 0.0);
    }

    #[test]
    fn riparian_band_increases_vegetation_but_leaves_the_edge_and_treeline_open() {
        let sample = |distance, height| {
            EnvironmentSample::evaluate(height, height, 0.1, distance, 0.5, 0.15)
        };
        let edge = sample(0.2, 0.);
        let grass_bank = sample(1.5, 0.);
        let wooded_bank = sample(12., 0.);
        let inland = sample(100., 0.);
        assert_eq!(edge.tree_density, 0.);
        assert_eq!(edge.grass_density, 0.);
        assert_eq!(grass_bank.tree_density, 0.);
        assert!(grass_bank.grass_density > 0.);
        assert!(wooded_bank.tree_density > inland.tree_density + 0.2);
        assert!(wooded_bank.grass_density > inland.grass_density);
        assert_eq!(sample(12., 75.).tree_density, 0.);
    }

    #[test]
    fn cached_habitats_and_tree_placement_are_repeatable_safe_and_bounded() {
        use crate::terrain::{SIDE, TerrainSettings};
        for seed in [20261003, 42, 314159] {
            for mountain_strength in [0.0, 0.65, 1.0] {
                let settings = TerrainSettings { mountain_strength };
                let world = Meadow::with_settings(seed, settings);
                let again = Meadow::with_settings(seed, settings);
                assert_eq!(world.environment, again.environment);
                assert_eq!(world.trees, again.trees);
                assert_eq!(world.environment.len(), SIDE * SIDE);
                for (index, sample) in world.environment.iter().enumerate() {
                    assert!(sample.elevation.is_finite());
                    assert!(sample.slope.is_finite() && sample.slope >= 0.0);
                    assert!(sample.water_distance.is_finite() && sample.water_distance >= 0.0);
                    for value in [
                        sample.moisture,
                        sample.tree_density,
                        sample.grass_density,
                        sample.rockiness,
                    ] {
                        assert!((0.0..=1.0).contains(&value));
                    }
                    let p = crate::river::position(index);
                    if sample.water_distance == 0.0 {
                        assert_eq!(sample.tree_density, 0.0);
                        assert_eq!(sample.grass_density, 0.0);
                    }
                    if p.length() <= 17.0 {
                        assert_eq!(sample.tree_density, 0.0);
                    }
                }
                assert!(
                    world.trees.iter().any(|(p, _)| p.length() > 160.0),
                    "Expanded region must have vegetation, seed {seed}"
                );
                for (p, size) in &world.trees {
                    assert!(world.environment_at(*p).tree_density > 0.0);
                    assert!(!world.near_water(*p, 3.0));
                    assert!(world.slope(*p) <= 0.65);
                    assert!(size.is_finite() && *size > 0.0);
                }
            }
        }
    }
}
