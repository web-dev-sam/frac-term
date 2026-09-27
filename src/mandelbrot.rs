//! Perturbation-based Mandelbrot renderer.
//!
//! One *reference* orbit is iterated on the CPU in arbitrary precision. Every
//! pixel then iterates only its *delta* from that orbit on the GPU
//! (`δ' = 2Zδ + δ² + Δc`), as an f32 mantissa pair with a shared integer
//! exponent so it survives far below f32's range. This stays accurate as long
//! as the delta remains small relative to the reference. Pixels for which that
//! breaks down (the reference escaped first, or a Pauldelbrot glitch) are
//! marked unresolved; a new reference is chosen among them and the kernel
//! re-runs on just those.
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
/// Squared escape radius. Far beyond 2 so the fractional escape count is
/// continuous across iteration boundaries; costs ~3 extra iterations per
/// escaping pixel (|z| squares each step).
const BAILOUT_SQ: f32 = 65536.0;
/// Marks a pixel the current reference could not resolve.
const UNRESOLVED: f32 = -1.0;

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
        let escapes = self.iterations(view, max_iterations, width, height);

        let mut rgb = vec![0u8; 3 * width * height];
        for (px, &nu) in rgb.as_chunks_mut::<3>().0.iter_mut().zip(&escapes) {
            let (r, g, b) = escape_to_rgb(nu, max_iterations);
            *px = [r, g, b];
        }
        let img_buf = RgbImage::from_raw(width as u32, height as u32, rgb)
            .expect("buffer size must match width * height * 3");
        DynamicImage::ImageRgb8(img_buf)
    }

    /// Fractional escape count per pixel (row-major), `max_iter` for interior points.
    fn iterations(&self, view: &View, max_iter: u32, width: usize, height: usize) -> Vec<f32> {
        let n_pixels = width * height;

        let (cx, cy) = view.center();
        let pixel_size = view.span() / width as f64;
        // pixel_size = pixel_mant · 2^pixel_exp with pixel_mant in [1, 2). f64
        // stays normal down to 2^-1022, i.e. zoom ~1e305 for a 1000px image.
        let (pixel_mant, pixel_exp) = frexp_normal(pixel_size);
        let grid = Grid {
            cx,
            cy,
            pixel_size: Fix::from_f64(pixel_size, cx.prec()),
            width,
            height,
        };

        let out_handle = self
            .client
            .create(Bytes::from_elems(vec![UNRESOLVED; n_pixels]));
        let cube_count = (n_pixels as u32).div_ceil(CUBE_DIM);

        let mut reference = (width / 2, height / 2);
        let mut iterations: Vec<f32> = Vec::new();
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
                    pixel_mant,
                    pixel_exp,
                    reference.0 as f32,
                    reference.1 as f32,
                );
            }

            let bytes = self
                .client
                .read_one(out_handle.clone())
                .expect("GPU readback of iteration counts");
            iterations = f32::from_bytes(&bytes).to_vec();

            unresolved = iterations
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n == UNRESOLVED)
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
            let resolved: Vec<(usize, f32)> = std::thread::scope(|scope| {
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

/// Continuous escape count: `n` minus how far past the bailout `|z|² = mag`
/// overshot, in doublings. Exactly `n` at the bailout, approaching `n - 1`
/// one full squaring beyond it.
fn fractional_escape(n: u32, mag: f64) -> f32 {
    (n as f64 - (mag.log2() / f64::from(BAILOUT_SQ).log2()).log2()) as f32
}

/// Split a positive normal f64 into `(m, e)` with `x = m · 2^e`, `m ∈ [1, 2)`.
fn frexp_normal(x: f64) -> (f32, i32) {
    debug_assert!(x.is_normal() && x > 0.0, "{x}");
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
    (m as f32, e)
}

/// Reference orbit `z_0, z_1, ...` as interleaved f32 `(re, im)` pairs.
/// Stops early (shorter than `max_iter`) if the reference point escapes.
fn reference_orbit(cx: &Fix, cy: &Fix, max_iter: u32) -> Vec<f32> {
    let prec = cx.prec();
    let mut zx = Fix::zero(prec);
    let mut zy = Fix::zero(prec);
    let bailout = Fix::from_f64(f64::from(BAILOUT_SQ), prec);

    let mut orbit = Vec::with_capacity(2 * max_iter as usize);
    for _ in 0..max_iter {
        orbit.push(zx.to_f64() as f32);
        orbit.push(zy.to_f64() as f32);

        let xx = zx.mul(&zx);
        let yy = zy.mul(&zy);
        if xx.add(&yy).gt(&bailout) {
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
fn iterate_full(cx: &Fix, cy: &Fix, max_iter: u32) -> f32 {
    let prec = cx.prec();
    let mut zx = Fix::zero(prec);
    let mut zy = Fix::zero(prec);
    let bailout = Fix::from_f64(f64::from(BAILOUT_SQ), prec);
    for n in 0..max_iter {
        let xx = zx.mul(&zx);
        let yy = zy.mul(&zy);
        let mag = xx.add(&yy);
        if mag.gt(&bailout) {
            return fractional_escape(n, mag.to_f64());
        }
        let new_zx = xx.sub(&yy).add(cx);
        let new_zy = zx.mul(&zy).double().add(cy);
        zx = new_zx;
        zy = new_zy;
    }
    max_iter as f32
}

/// `2^e` as f32; zero below the normal range (`e < -126`). Callers only pass
/// exponents whose value fits, so `e > 127` never occurs.
#[cube]
fn pow2(e: i32) -> f32 {
    let mut p = 0.0f32;
    if e >= -126 {
        p = f32::reinterpret::<u32>(u32::cast_from(e + 127) << 23u32);
    }
    p
}

/// `floor(log2(m))` for a positive normal f32.
#[cube]
fn exponent(m: f32) -> i32 {
    i32::cast_from((u32::reinterpret::<f32>(m) >> 23u32) & 0xffu32) - 127
}

/// Iterates the delta of each still-unresolved pixel against the reference
/// orbit. Writes the fractional escape count, `max_iter` for interior points,
/// or [`UNRESOLVED`] when the reference is insufficient for this pixel.
///
/// The delta is held as `(dx, dy) · 2^e`: an f32 mantissa pair with a shared
/// integer exponent, so it stays representable long after `pixel_size` itself
/// has left f32's range (zoom > 1e38). With `Δc = (cx, cy) · 2^pixel_exp`,
///
/// ```text
/// δ' = 2Zδ + δ² + Δc
///    = 2^e · [ 2Z(dx,dy) + 2^e (dx,dy)² + 2^(pixel_exp - e) (cx,cy) ]
/// ```
///
/// Only the two bracketed scale factors depend on `e`; they are cached and
/// refreshed when the mantissa drifts out of `[2^-32, 2^32]`. A factor that
/// underflows f32 is dropped as zero, which is exact to f32 precision since
/// the term is then below the rounding error of `2Zδ`.
#[cube(launch_unchecked)]
fn mandel_delta(
    ref_orbit: &Array<f32>,
    out: &mut Array<f32>,
    width: u32,
    height: u32,
    ref_len: u32,
    max_iter: u32,
    pixel_mant: f32,
    pixel_exp: i32,
    ref_px: f32,
    ref_py: f32,
) {
    let i = ABSOLUTE_POS;

    if i < (width * height) as usize && out[i] == UNRESOLVED {
        let px = (i as u32) % width;
        let py = (i as u32) / width;

        let cx = (px as f32 - ref_px) * pixel_mant;
        let cy = (ref_py - py as f32) * pixel_mant;

        let mut dx = 0.0f32;
        let mut dy = 0.0f32;
        // Starting at Δc's exponent makes the first `+ Δc` land unscaled.
        let mut e = pixel_exp;
        let mut scale = pow2(e);
        let mut ccx = cx;
        let mut ccy = cy;
        let mut result = max_iter as f32;
        let mut n = 0u32;

        while n < max_iter {
            if n >= ref_len {
                result = UNRESOLVED;
                break;
            }

            let zx = ref_orbit[(2 * n) as usize];
            let zy = ref_orbit[(2 * n + 1) as usize];

            let fx = zx + dx * scale;
            let fy = zy + dy * scale;
            let mag = fx * fx + fy * fy;

            if mag > BAILOUT_SQ {
                // Same formula as `fractional_escape`; no log2 on device, so
                // ln with the constants folded: log2(log2(mag) / 16).
                let ln_bailout = f32::ln(BAILOUT_SQ);
                let ln2 = f32::ln(2.0f32);
                result = n as f32 - f32::ln(f32::ln(mag) / ln_bailout) / ln2;
                break;
            }

            // Pauldelbrot criterion: |Z + δ| << |Z| means the delta has
            // absorbed all the precision and the result is untrustworthy.
            let ref_mag = zx * zx + zy * zy;
            if mag < ref_mag * 1e-6f32 {
                result = UNRESOLVED;
                break;
            }

            let new_dx = 2.0f32 * (zx * dx - zy * dy) + scale * (dx * dx - dy * dy) + ccx;
            let new_dy = 2.0f32 * (zx * dy + zy * dx) + scale * (2.0f32 * dx * dy) + ccy;
            dx = new_dx;
            dy = new_dy;
            n += 1;

            let m = f32::max(f32::abs(dx), f32::abs(dy));
            if m > 4294967296.0f32 || (m < 2.3283064e-10f32 && m > 0.0f32) {
                // |δ| never exceeds the bailout radius, so k <= 40 and the
                // rescale 2^-k is always representable.
                let k = exponent(m);
                let s = pow2(-k);
                dx *= s;
                dy *= s;
                e += k;
                scale = pow2(e);
                // Δc relative to the new scale. Cancellation can leave δ
                // briefly below Δc (positive shift); the clamp keeps the
                // product finite in the pathological case.
                let shift = pixel_exp - e;
                let t = pow2(i32::min(shift, 64));
                ccx = cx * t;
                ccy = cy * t;
            }
        }

        out[i] = result;
    }
}

/// Map the fractional escape count into the corresponding pixel color
fn escape_to_rgb(nu: f32, max_iterations: u32) -> (u8, u8, u8) {
    if nu >= max_iterations as f32 {
        return (0, 0, 0);
    }
    let nu = f64::from(nu);

    let hue = nu / 64.0;
    let saturation = 1.0;
    let value = nu / (nu + 8.0);

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

    fn direct_f64(cx: f64, cy: f64, max_iter: u32) -> f32 {
        let (mut zx, mut zy) = (0.0f64, 0.0f64);
        for n in 0..max_iter {
            let (xx, yy) = (zx * zx, zy * zy);
            if xx + yy > f64::from(BAILOUT_SQ) {
                return fractional_escape(n, xx + yy);
            }
            (zx, zy) = (xx - yy + cx, 2.0 * zx * zy + cy);
        }
        max_iter as f32
    }

    /// Fraction of pixels whose GPU escape count differs from `expected` by
    /// more than f32 perturbation noise in the fractional part.
    fn mismatch_rate(actual: &[f32], expected: impl Fn(usize) -> f32) -> f64 {
        let mismatches = (0..actual.len())
            .filter(|&i| (actual[i] - expected(i)).abs() > 0.01)
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
        let distinct = got
            .iter()
            .map(|&n| n as u32)
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let rate = mismatch_rate(&got, |i| {
            let (px, py) = grid.coordinate(i % w, i / w);
            iterate_full(&px, &py, max_iter)
        });
        (rate, distinct)
    }

    /// Near the antenna tip escape times vary row by row at pixel scale, so a
    /// perturbation error would show up as a band shift. Zoom 1e30 needs
    /// ~164 fixed-point bits, far past what f64 could address; from 1e38 the
    /// pixel deltas are below f32's exponent range.
    #[test]
    fn matches_full_precision_at_deep_zoom() {
        for (zoom, max_iter) in [
            (1e20, 300),
            (1e30, 300),
            (1e40, 300),
            (1e100, 600),
            (1e300, 1500),
        ] {
            let (rate, distinct) = measure_deep(-1.999_999_999_9, 0.0, zoom, max_iter);
            assert!(distinct > 1, "degenerate view at zoom {zoom}");
            assert!(rate < 0.02, "mismatch rate {rate} at zoom {zoom}");
        }
    }
}
