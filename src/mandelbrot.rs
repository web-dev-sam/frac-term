use image::{DynamicImage, RgbImage};
use num_complex::{Complex, Complex64};
use ratatui::layout::Size;

use crate::App;

pub fn render(app: &App, size: &Size) -> DynamicImage {
    let (width, height) = (size.width as usize, size.height as usize);
    let (min_x, max_x, min_y, max_y) =
        get_mathematical_coordinate_frame(width, height, &app.position, app.zoom);

    let mut frame = vec![0u8; width * height * 3];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let rows_per_chunk = height.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_idx, chunk) in frame.chunks_mut(rows_per_chunk * width * 3).enumerate() {
            let y0 = chunk_idx * rows_per_chunk;
            scope.spawn(move || {
                for (row_idx, row) in chunk.chunks_exact_mut(width * 3).enumerate() {
                    let im = map_range((y0 + row_idx) as f64, 0.0, height as f64, min_y, max_y);
                    for (x, px) in row.chunks_exact_mut(3).enumerate() {
                        let re = map_range(x as f64, 0.0, width as f64, min_x, max_x);
                        let (r, g, b) = mandelbrot(&Complex64 { re, im }, app.max_iterations);
                        px[0] = r;
                        px[1] = g;
                        px[2] = b;
                    }
                }
            });
        }
    });
    let img_buf = RgbImage::from_raw(width as u32, height as u32, frame)
        .expect("buffer size must match width * height * 3");
    DynamicImage::ImageRgb8(img_buf)
}

/// Calculates the mathematical bounds of the mandelbrot view frame
fn get_mathematical_coordinate_frame(
    width: usize,
    height: usize,
    position: &Complex<f64>,
    zoom: f64,
) -> (f64, f64, f64, f64) {
    let mathematical_half_width = 2.0 / zoom; // 2.0 is the zoom=1 left bound
    let mathematical_half_height = mathematical_half_width * height as f64 / width as f64;
    let min_x = position.re - mathematical_half_width;
    let max_x = position.re + mathematical_half_width;
    let min_y = position.im + mathematical_half_height;
    let max_y = position.im - mathematical_half_height;
    (min_x, max_x, min_y, max_y)
}

/// Returns the color for a coordinate in the mandelbrot set
fn mandelbrot(coordinate: &Complex64, max_iterations: u16) -> (u8, u8, u8) {
    let mut z = Complex64::new(0.0, 0.0);
    let mut i = 0;
    while i < max_iterations && z.norm_sqr() <= 4.0 {
        z = z * z + coordinate;
        i += 1;
    }
    iterations_to_rgb(i as f64, max_iterations)
}

/// Map the amount of recursive iterations into the corresponding pixel color
fn iterations_to_rgb(current_iterations: f64, max_iterations: u16) -> (u8, u8, u8) {
    if current_iterations >= max_iterations as f64 {
        return (0, 0, 0);
    }

    let hue = current_iterations / 64.0;
    let saturation = 1.0;
    let value = current_iterations / (current_iterations + 8.0);

    hsv_to_rgb(hue, saturation, value)
}

/// Maps a range to another range linearly
fn map_range(val: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    to_min + (val - from_min) * (to_max - to_min) / (from_max - from_min)
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
