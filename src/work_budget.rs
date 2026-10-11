//! Cooperative CPU work. Native workers run uninterrupted; Web jobs give the
//! browser event loop time to paint and handle input between small batches.
use bevy::platform::time::Instant;

#[cfg(any(target_arch = "wasm32", test))]
pub fn render_size(width: u32, height: u32) -> bevy::math::UVec2 {
    let width = width.max(1);
    let height = height.max(1);
    let scale = (1280. / width as f64).min(720. / height as f64).min(1.);
    bevy::math::UVec2::new(
        (width as f64 * scale).floor().max(1.) as u32,
        (height as f64 * scale).floor().max(1.) as u32,
    )
}

pub struct WorkBudget {
    start: Instant,
    enabled: bool,
    #[cfg(test)]
    force: bool,
}

impl WorkBudget {
    pub fn unlimited() -> Self {
        Self {
            start: Instant::now(),
            enabled: false,
            #[cfg(test)]
            force: false,
        }
    }
    pub fn runtime() -> Self {
        Self {
            enabled: cfg!(target_arch = "wasm32"),
            ..Self::unlimited()
        }
    }
    #[cfg(test)]
    pub fn forced() -> Self {
        Self {
            enabled: true,
            force: true,
            ..Self::unlimited()
        }
    }
    pub async fn checkpoint(&mut self) {
        if !self.enabled {
            return;
        }
        #[cfg(test)]
        if self.force {
            bevy::tasks::futures_lite::future::yield_now().await;
            return;
        }
        if self.start.elapsed().as_secs_f64() < 0.002 {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            crate::web::record_slice(self.start.elapsed().as_secs_f64() * 1000.);
            crate::web::yield_to_browser().await;
        }
        self.start = Instant::now();
    }
}

impl Drop for WorkBudget {
    fn drop(&mut self) {
        #[cfg(target_arch = "wasm32")]
        if self.enabled {
            crate::web::record_slice(self.start.elapsed().as_secs_f64() * 1000.);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn render_budget_caps_pixels_without_changing_aspect_or_upscaling() {
        for (width, height) in [(496, 552), (1274, 778), (3840, 2160), (390, 844)] {
            let size = render_size(width, height);
            assert!(size.x <= 1280 && size.y <= 720);
            assert!(size.x <= width && size.y <= height);
            assert!((size.x as f64 / size.y as f64 - width as f64 / height as f64).abs() < 0.005);
        }
        assert_eq!(render_size(496, 552), bevy::math::UVec2::new(496, 552));
        assert_eq!(render_size(0, 0), bevy::math::UVec2::ONE);
    }
}
