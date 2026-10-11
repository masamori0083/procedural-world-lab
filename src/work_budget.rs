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

// None means continuous interactive rendering. A paused preview keeps its image,
// but should not compete with another window's active WebGPU game.
#[cfg(any(target_arch = "wasm32", test))]
pub fn browser_frame_wait(
    loaded: bool,
    active: bool,
    foreground: bool,
) -> Option<std::time::Duration> {
    if !loaded || (active && foreground) {
        None
    } else if foreground {
        Some(std::time::Duration::from_secs_f64(1. / 15.))
    } else {
        Some(std::time::Duration::from_millis(500))
    }
}

// Wait for settled geometry and pipeline warmup before reusing a paused view.
#[cfg(any(target_arch = "wasm32", test))]
pub fn browser_draw_scene(loaded: bool, playing: bool, unchanged: bool, settled_secs: f64) -> bool {
    !loaded || playing || !unchanged || settled_secs < 2.
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
    fn paused_web_view_reuses_the_frame_but_loading_input_and_scene_changes_refresh_it() {
        assert!(!browser_draw_scene(true, false, true, 3.));
        assert!(browser_draw_scene(false, false, true, 3.));
        assert!(browser_draw_scene(true, true, true, 3.));
        assert!(browser_draw_scene(true, false, false, 3.));
        assert!(browser_draw_scene(true, false, true, 0.));
        assert!(browser_draw_scene(true, false, true, 1.9));
        assert!(!browser_draw_scene(true, false, true, 2.));
    }

    #[test]
    fn browser_paused_rendering_yields_gpu_budget_and_resumes_without_limiting_play() {
        use std::time::Duration;
        assert_eq!(browser_frame_wait(false, false, true), None);
        assert_eq!(browser_frame_wait(true, true, true), None);
        assert_eq!(
            browser_frame_wait(true, false, true),
            Some(Duration::from_secs_f64(1. / 15.))
        );
        assert_eq!(
            browser_frame_wait(true, true, false),
            Some(Duration::from_millis(500))
        );
        assert_eq!(
            browser_frame_wait(true, false, false),
            Some(Duration::from_millis(500))
        );
        // Loading continues promptly, and a streamed ground wait must not
        // reset the sticky loaded flag or remove the idle/background budget.
        assert_eq!(browser_frame_wait(false, false, false), None);
        assert_eq!(browser_frame_wait(true, true, true), None);
    }

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
