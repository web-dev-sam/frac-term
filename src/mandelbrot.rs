//! Perturbation-based Mandelbrot renderer.
//!
//! One *reference* orbit is iterated on the CPU in arbitrary precision. Every
//! pixel then iterates only its f32 *delta* from that orbit on the GPU
//! (`δ' = 2Zδ + δ² + Δc`), which stays accurate as long as the delta remains
//! small relative to the reference. Pixels for which that breaks down (the
//! reference escaped first, or a Pauldelbrot glitch) are marked unresolved; a
//! new reference is chosen among them and the kernel re-runs on just those.
//! Whatever is still unresolved after a few passes is iterated in full
//! precision on the CPU.

use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::cube;
use cubecl::prelude::*;
use image::{DynamicImage, RgbImage};
use ratatui::layout::Size;

use crate::View;
use crate::fix::Fix;

type Rt = cubecl::wgpu::WgpuRuntime;

const CUBE_DIM: u32 = 256;
const MAX_REFERENCE_PASSES: usize = 8;

pub struct Renderer {
    client: ComputeClient<Rt>,
}

/// Maps pixel coordinates onto the complex plane in full precision.
struct Grid<'a> {
    cx: &'a Fix,
    cy: &'a Fix,
    pixel_size: Fix,
    width: usize,
    height: usize,
}

impl Grid<'_> {
    fn coordinate(&self, px: usize, py: usize) -> (Fix, Fix) {
        let dre = self.pixel_size.scale(px as i64 - (self.width / 2) as i64);
        let dim = self.pixel_size.scale((self.height / 2) as i64 - py as i64);
        (self.cx.add(&dre), self.cy.add(&dim))
    }
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            client: Rt::client(&Default::default()),
        }
    }

    pub fn render(&self, view: &View, max_iterations: u32, size: Size) -> DynamicImage {
        let (width, height) = (size.width as usize, size.height as usize);
        let iterations = self.iterations(view, max_iterations as u32, width, height);

        // Colors depend only on the iteration count, so build the palette once.
        let palette: Vec<[u8; 3]> = (0..=max_iterations as u32)
            .map(|n| {
                let (r, g, b) = iterations_to_rgb(n as f64, max_iterations);
                [r, g, b]
            })
            .collect();
        let mut rgb = vec![0u8; 3 * width * height];
        for (px, &n) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(&iterations) {
            *px = palette[n as usize];
        }
        let img_buf = RgbImage::from_raw(width as u32, height as u32, rgb)
            .expect("buffer size must match width * height * 3");
        DynamicImage::ImageRgb8(img_buf)
    }

    /// Escape iteration per pixel (row-major), `max_iter` for interior points.
    fn iterations(&self, view: &View, max_iter: u32, width: usize, height: usize) -> Vec<u32> {
        let n_pixels = width * height;
        // Iteration counts are `0..=max_iter`; one past that marks "unresolved".
        let unresolved_mark = max_iter + 1;

        let (cx, cy) = view.center();
        let pixel_size = view.span() / width as f64;
        let grid = Grid {
            cx,
            cy,
            pixel_size: Fix::from_f64(pixel_size, cx.prec()),
            width,
            height,
        };

        let out_handle = self
            .client
            .create(Bytes::from_elems(vec![unresolved_mark; n_pixels]));
        let cube_count = (n_pixels as u32).div_ceil(CUBE_DIM);

        let mut reference = (width / 2, height / 2);
        let mut iterations: Vec<u32> = Vec::new();
        let mut unresolved: Vec<usize> = Vec::new();
        for _ in 0..MAX_REFERENCE_PASSES {
            let (rx, ry) = grid.coordinate(reference.0, reference.1);
            let orbit = reference_orbit(&rx, &ry, max_iter);
            let ref_len = (orbit.len() / 2) as u32;
            let orbit_len = orbit.len();
            let orbit_handle = self.client.create(Bytes::from_elems(orbit));

            unsafe {
                mandel_delta::launch_unchecked::<Rt>(
                    &self.client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(orbit_handle, orbit_len),
                    ArrayArg::from_raw_parts(out_handle.clone(), n_pixels),
                    width as u32,
                    height as u32,
                    ref_len,
                    max_iter,
                    pixel_size as f32,
                    reference.0 as f32,
                    reference.1 as f32,
                );
            }

            let bytes = self
                .client
                .read_one(out_handle.clone())
                .expect("GPU readback of iteration counts");
            iterations = u32::from_bytes(&bytes).to_vec();

            unresolved = iterations
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n == unresolved_mark)
                .map(|(i, _)| i)
                .collect();
            if unresolved.is_empty() {
                break;
            }
            // The median unresolved pixel tends to sit inside the largest
            // glitched region rather than on its edge.
            let i = unresolved[unresolved.len() / 2];
            reference = (i % width, i / width);
        }

        if !unresolved.is_empty() {
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            let per_thread = unresolved.len().div_ceil(threads).max(1);
            let resolved: Vec<(usize, u32)> = std::thread::scope(|scope| {
                let workers: Vec<_> = unresolved
                    .chunks(per_thread)
                    .map(|indices| {
                        let grid = &grid;
                        scope.spawn(move || {
                            indices
                                .iter()
                                .map(|&i| {
                                    let (cx, cy) = grid.coordinate(i % width, i / width);
                                    (i, iterate_full(&cx, &cy, max_iter))
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                workers
                    .into_iter()
                    .flat_map(|w| w.join().expect("fallback worker panicked"))
                    .collect()
            });
            for (i, n) in resolved {
                iterations[i] = n;
            }
        }
        iterations
    }
}

/// Reference orbit `z_0, z_1, ...` as interleaved f32 `(re, im)` pairs.
/// Stops early (shorter than `max_iter`) if the reference point escapes.
fn reference_orbit(cx: &Fix, cy: &Fix, max_iter: u32) -> Vec<f32> {
    let prec = cx.prec();
    let mut zx = Fix::zero(prec);
    let mut zy = Fix::zero(prec);
    let four = Fix::from_f64(4.0, prec);

    let mut orbit = Vec::with_capacity(2 * max_iter as usize);
    for _ in 0..max_iter {
        orbit.push(zx.to_f64() as f32);
        orbit.push(zy.to_f64() as f32);

        let xx = zx.mul(&zx);
        let yy = zy.mul(&zy);
        if xx.add(&yy).gt(&four) {
            break;
        }
        let new_zx = xx.sub(&yy).add(cx);
        let new_zy = zx.mul(&zy).double().add(cy);
        zx = new_zx;
        zy = new_zy;
    }
    orbit
}

/// Full-precision escape-time iteration for a single point.
fn iterate_full(cx: &Fix, cy: &Fix, max_iter: u32) -> u32 {
    let prec = cx.prec();
    let mut zx = Fix::zero(prec);
    let mut zy = Fix::zero(prec);
    let four = Fix::from_f64(4.0, prec);
    for n in 0..max_iter {
        let xx = zx.mul(&zx);
        let yy = zy.mul(&zy);
        if xx.add(&yy).gt(&four) {
            return n;
        }
        let new_zx = xx.sub(&yy).add(cx);
        let new_zy = zx.mul(&zy).double().add(cy);
        zx = new_zx;
        zy = new_zy;
    }
    max_iter
}

/// Iterates the delta of each still-unresolved pixel against the reference
/// orbit. Writes the escape iteration, `max_iter` for interior points, or
/// `max_iter + 1` when the reference is insufficient for this pixel.
#[cube(launch_unchecked)]
fn mandel_delta(
    ref_orbit: &Array<f32>,
    out: &mut Array<u32>,
    width: u32,
    height: u32,
    ref_len: u32,
    max_iter: u32,
    pixel_size: f32,
    ref_px: f32,
    ref_py: f32,
) {
    let i = ABSOLUTE_POS;
    let unresolved = max_iter + 1;

    if i < (width * height) as usize && out[i] == unresolved {
        let px = (i as u32) % width;
        let py = (i as u32) / width;

        let dcx = (px as f32 - ref_px) * pixel_size;
        let dcy = (ref_py - py as f32) * pixel_size;

        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        let mut result = max_iter;
        let mut n = 0u32;

        while n < max_iter {
            if n >= ref_len {
                result = unresolved;
                break;
            }

            let zx = ref_orbit[(2 * n) as usize];
            let zy = ref_orbit[(2 * n + 1) as usize];

            let fx = zx + dx;
            let fy = zy + dy;
            let mag = fx * fx + fy * fy;

            if mag > 4.0f32 {
                result = n;
                break;
            }

            // Pauldelbrot criterion: |Z + δ| << |Z| means the delta has
            // absorbed all the precision and the result is untrustworthy.
            let ref_mag = zx * zx + zy * zy;
            if mag < ref_mag * 1e-6f32 {
                result = unresolved;
                break;
            }

            let new_dx = 2.0f32 * (zx * dx - zy * dy) + (dx * dx - dy * dy) + dcx;
            let new_dy = 2.0f32 * (zx * dy + zy * dx) + 2.0f32 * dx * dy + dcy;
            dx = new_dx;
            dy = new_dy;
            n += 1;
        }

        out[i] = result;
    }
}

/// Map the amount of recursive iterations into the corresponding pixel color
fn iterations_to_rgb(current_iterations: f64, max_iterations: u32) -> (u8, u8, u8) {
    if current_iterations >= max_iterations as f64 {
        return (0, 0, 0);
    }

    let hue = current_iterations / 64.0;
    let saturation = 1.0;
    let value = current_iterations / (current_iterations + 8.0);

    hsv_to_rgb(hue, saturation, value)
}

/// Converts hsv color into an rgb color
fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (u8, u8, u8) {
    let h = h.rem_euclid(1.0) * 6.0; // h in [0,1) -> [0,6)
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;

    let (r1, g1, b1) = if h < 1.0 {
        (c, x, 0.0)
    } else if h < 2.0 {
        (x, c, 0.0)
    } else if h < 3.0 {
        (0.0, c, x)
    } else if h < 4.0 {
        (0.0, x, c)
    } else if h < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r1 + m) * 255.0).round() as u8,
        ((g1 + m) * 255.0).round() as u8,
        ((b1 + m) * 255.0).round() as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(re: f64, im: f64, zoom: f64) -> View {
        let prec = View::precision_for(zoom);
        View {
            re: Fix::from_f64(re, prec),
            im: Fix::from_f64(im, prec),
            zoom,
        }
    }

    fn direct_f64(cx: f64, cy: f64, max_iter: u32) -> u32 {
        let (mut zx, mut zy) = (0.0f64, 0.0f64);
        for n in 0..max_iter {
            let (xx, yy) = (zx * zx, zy * zy);
            if xx + yy > 4.0 {
                return n;
            }
            (zx, zy) = (xx - yy + cx, 2.0 * zx * zy + cy);
        }
        max_iter
    }

    /// Fraction of pixels whose GPU iteration count differs from `expected`.
    fn mismatch_rate(actual: &[u32], expected: impl Fn(usize) -> u32) -> f64 {
        let mismatches = (0..actual.len())
            .filter(|&i| actual[i] != expected(i))
            .count();
        mismatches as f64 / actual.len() as f64
    }

    fn measure_low_zoom(re: f64, im: f64, zoom: f64) -> f64 {
        let (w, h, max_iter) = (256usize, 192usize, 100u32);
        let v = view(re, im, zoom);
        let got = Renderer::new().iterations(&v, max_iter, w, h);
        let pixel_size = v.span() / w as f64;
        mismatch_rate(&got, |i| {
            let cx = re + (i % w) as f64 * pixel_size - (w / 2) as f64 * pixel_size;
            let cy = im + (h / 2) as f64 * pixel_size - (i / w) as f64 * pixel_size;
            direct_f64(cx, cy, max_iter)
        })
    }

    #[test]
    fn matches_direct_computation_with_interior_reference() {
        let rate = measure_low_zoom(0.0, 0.0, 1.0);
        assert!(rate < 0.02, "mismatch rate {rate}");
    }

    #[test]
    fn matches_direct_computation_with_escaping_reference() {
        let rate = measure_low_zoom(-2.2, 0.8, 1.5);
        assert!(rate < 0.02, "mismatch rate {rate}");
    }

    fn measure_deep(re: f64, im: f64, zoom: f64, max_iter: u32) -> (f64, usize) {
        let (w, h) = (64usize, 48usize);
        let v = view(re, im, zoom);
        let got = Renderer::new().iterations(&v, max_iter, w, h);
        let (cx, cy) = v.center();
        let grid = Grid {
            cx,
            cy,
            pixel_size: Fix::from_f64(v.span() / w as f64, cx.prec()),
            width: w,
            height: h,
        };
        let distinct = got.iter().collect::<std::collections::BTreeSet<_>>().len();
        let rate = mismatch_rate(&got, |i| {
            let (px, py) = grid.coordinate(i % w, i / w);
            iterate_full(&px, &py, max_iter)
        });
        (rate, distinct)
    }

    /// Near the antenna tip escape times vary row by row at pixel scale, so a
    /// perturbation error would show up as a band shift. Zoom 1e30 needs
    /// ~164 fixed-point bits, far past what f64 could address.
    #[test]
    fn matches_full_precision_at_deep_zoom() {
        for zoom in [1e20, 1e30] {
            let (rate, distinct) = measure_deep(-1.999_999_999_9, 0.0, zoom, 300);
            assert!(distinct > 1, "degenerate view at zoom {zoom}");
            assert!(rate < 0.02, "mismatch rate {rate} at zoom {zoom}");
        }
    }
}
