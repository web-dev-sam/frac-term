mod event;
mod fix;
mod kitty;
mod mandelbrot;

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crossterm::terminal::{BeginSynchronizedUpdate, EndSynchronizedUpdate};
use crossterm::{execute, queue};

use ratatui::{
    DefaultTerminal, Frame,
    layout::{Constraint, Layout, Size},
    widgets::{Block, Borders, Paragraph},
};
use ratatui_image::{
    FontSize, Image, Resize,
    picker::{Picker, ProtocolType, cap_parser},
    protocol::Protocol,
};

use crate::event::AppEvent;
use crate::fix::Fix;

/// The visible region of the complex plane.
///
/// At `zoom == 1` the view spans `[-2, 2]` horizontally. The center is kept in
/// fixed-point with `64 + log2(zoom)` bits so panning stays exact far beyond
/// what f64 can address.
#[derive(Clone)]
pub struct View {
    re: Fix,
    im: Fix,
    zoom: f64,
}

impl View {
    fn precision_for(zoom: f64) -> u64 {
        64 + zoom.log2().max(0.0).ceil() as u64
    }

    fn new() -> Self {
        let prec = Self::precision_for(1.0);
        Self {
            re: Fix::zero(prec),
            im: Fix::zero(prec),
            zoom: 1.0,
        }
    }

    fn zoom_by(&mut self, factor: f64) {
        self.zoom *= factor;
        let prec = Self::precision_for(self.zoom);
        if prec != self.re.prec() {
            self.re = self.re.with_prec(prec);
            self.im = self.im.with_prec(prec);
        }
    }

    /// Pan by a fraction of the half-width (2 / zoom) along each axis.
    fn pan(&mut self, dre: f64, dim: f64) {
        let half_width = 2.0 / self.zoom;
        let prec = self.re.prec();
        self.re = self.re.add(&Fix::from_f64(dre * half_width, prec));
        self.im = self.im.add(&Fix::from_f64(dim * half_width, prec));
    }

    /// Width of the view in complex-plane units.
    pub fn span(&self) -> f64 {
        4.0 / self.zoom
    }

    pub fn center(&self) -> (&Fix, &Fix) {
        (&self.re, &self.im)
    }

    /// Human-readable position, with enough decimals to distinguish pixels.
    pub fn status_line(&self) -> String {
        // ~log10(zoom) digits locate the view; +3 resolves within it.
        let digits = self.zoom.log10().max(0.0) as usize + 3;
        format!(
            "zoom {:.3e}  |  re {}  im {}  |  {} bits",
            self.zoom,
            self.re.to_decimal(digits),
            self.im.to_decimal(digits),
            self.re.prec(),
        )
    }
}

struct RenderRequest {
    id: u64,
    view: View,
    max_iterations: u32,
    /// Image size in pixels.
    pixels: Size,
    /// Image area in terminal cells.
    cells: Size,
}

/// A frame encoded for the terminal, ready to be rendered as a widget.
enum Picture {
    /// ratatui-image's encoders (sixel, iTerm2, halfblocks).
    Protocol(Protocol),
    /// Our PNG-payload Kitty widget; ratatui-image's Kitty path sends raw RGBA.
    Kitty(kitty::KittyImage),
}

struct RenderResult {
    id: u64,
    picture: Picture,
}

/// Runs the GPU renderer off the UI thread so the last finished frame stays
/// on screen while the next one computes. Queued requests are coalesced:
/// holding a key produces one render of the final view, not one per press.
fn spawn_renderer(picker: Picker) -> (Sender<RenderRequest>, Receiver<RenderResult>) {
    let (request_tx, request_rx) = mpsc::channel::<RenderRequest>();
    let (result_tx, result_rx) = mpsc::channel::<RenderResult>();
    thread::spawn(move || {
        let renderer = mandelbrot::Renderer::new();
        // Kitty image ids must be unique within the terminal session; other
        // programs may have left images behind, so start from a clock-derived
        // value rather than 1.
        let mut kitty_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(1, |d| d.subsec_nanos())
            | 1;
        let mut previous_kitty_id = None;
        while let Ok(mut request) = request_rx.recv() {
            while let Ok(newer) = request_rx.try_recv() {
                request = newer;
            }
            let img = renderer.render(&request.view, request.max_iterations, request.pixels);
            let picture = if picker.protocol_type() == ProtocolType::Kitty {
                kitty_id = kitty_id.wrapping_add(2);
                let image =
                    kitty::KittyImage::new(&img, request.cells, kitty_id, previous_kitty_id);
                previous_kitty_id = Some(kitty_id);
                Picture::Kitty(image)
            } else {
                Picture::Protocol(
                    picker
                        .new_protocol(img, request.cells, Resize::Fit(None))
                        .expect("ratatui_image::picker::Picker::new_protocol() succeeds"),
                )
            };
            if result_tx
                .send(RenderResult {
                    id: request.id,
                    picture,
                })
                .is_err()
            {
                return; // UI thread is gone
            }
        }
    });
    (request_tx, result_rx)
}

struct App {
    ratatui_image_picker: Picker,
    view: View,
    max_iterations: u32,
    requests: Sender<RenderRequest>,
    results: Receiver<RenderResult>,
    /// Last finished frame; shown until the next one arrives.
    picture: Option<Picture>,
    /// Pixel size of the image area as of the last draw.
    image_size: Size,
    /// Cell size of the image area as of the last draw.
    image_cells: Size,
    /// Size the newest request was rendered for; differs from `image_size` after a resize.
    requested_size: Size,
    view_changed: bool,
    latest_request: u64,
    latest_result: u64,
}

impl App {
    /// Initialize app state
    pub fn new(picker: Picker) -> Self {
        let (requests, results) = spawn_renderer(picker.clone());
        Self {
            view: View::new(),
            max_iterations: 100,
            ratatui_image_picker: picker,
            requests,
            results,
            picture: None,
            image_size: Size::default(),
            image_cells: Size::default(),
            requested_size: Size::default(),
            view_changed: true,
            latest_request: 0,
            latest_result: 0,
        }
    }

    fn apply(&mut self, event: AppEvent) {
        const PAN_STEP: f64 = 1.0 / 8.0; // 1/8 of half width
        match event {
            AppEvent::ZoomIncrease => self.view.zoom_by(1.2),
            AppEvent::ZoomDecrease => self.view.zoom_by(1.0 / 1.2),
            AppEvent::IterationIncrease => {
                self.max_iterations = (1.05 * self.max_iterations as f32).round().max(1.0) as u32
            }
            AppEvent::IterationDecrease => {
                self.max_iterations = (0.95 * self.max_iterations as f32).round().max(1.0) as u32
            }
            AppEvent::MoveLeft => self.view.pan(-PAN_STEP, 0.0),
            AppEvent::MoveRight => self.view.pan(PAN_STEP, 0.0),
            AppEvent::MoveDown => self.view.pan(0.0, -PAN_STEP),
            AppEvent::MoveUp => self.view.pan(0.0, PAN_STEP),
            AppEvent::Resize => return, // picked up via `image_size` on the next draw
            AppEvent::Quit => unreachable!("handled by the event loop"),
        }
        self.view_changed = true;
    }

    fn is_rendering(&self) -> bool {
        self.latest_request != self.latest_result
    }

    /// Queue a render if the view or the image area changed since the last request.
    fn request_render_if_needed(&mut self) {
        if !self.view_changed && self.image_size == self.requested_size {
            return;
        }
        if self.image_size.width == 0 || self.image_size.height == 0 {
            return;
        }
        self.latest_request += 1;
        self.requested_size = self.image_size;
        self.view_changed = false;
        self.requests
            .send(RenderRequest {
                id: self.latest_request,
                view: self.view.clone(),
                max_iterations: self.max_iterations,
                pixels: self.image_size,
                cells: self.image_cells,
            })
            .expect("render thread alive");
    }

    /// Adopt a finished frame. Returns whether the display changed.
    fn poll_results(&mut self) -> bool {
        let mut changed = false;
        while let Ok(result) = self.results.try_recv() {
            if result.id > self.latest_result {
                self.latest_result = result.id;
                self.picture = Some(result.picture);
                changed = true;
            }
        }
        changed
    }

    fn draw(&mut self, frame: &mut Frame) {
        let [stats_area, image_area] =
            Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(frame.area());

        let mut status = format!(
            "{}  |  {} iterations",
            self.view.status_line(),
            self.max_iterations
        );
        if self.is_rendering() {
            status.push_str("  |  rendering...");
        }
        let stats = Paragraph::new(status).block(Block::new().borders(Borders::BOTTOM));
        frame.render_widget(stats, stats_area);

        let FontSize {
            width: font_w,
            height: font_h,
        } = self.ratatui_image_picker.font_size();
        self.image_size = Size::new(image_area.width * font_w, image_area.height * font_h);
        self.image_cells = image_area.as_size();

        match &self.picture {
            Some(Picture::Protocol(protocol)) => {
                frame.render_widget(Image::new(protocol), image_area)
            }
            Some(Picture::Kitty(image)) => frame.render_widget(image, image_area),
            None => (),
        }
    }
}

pub fn run(terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    const INPUT_POLL: Duration = Duration::from_millis(30);

    let in_vscode = std::env::var("TERM_PROGRAM").is_ok_and(|p| p.contains("vscode"));
    let mut options = cap_parser::QueryStdioOptions::default();
    if in_vscode {
        options.blacklist_protocols.push(ProtocolType::Kitty);
    }
    let mut picker =
        Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks());
    // xterm.js decodes iTerm2 inline images asynchronously in the browser, so the
    // erase that precedes each frame is painted before the image arrives (black
    // flash). Its sixel decoder runs inline in the parser, so erase and image land
    // in the same synchronized update.
    if in_vscode && picker.protocol_type() == ProtocolType::Iterm2 {
        picker.set_protocol_type(ProtocolType::Sixel);
    }
    let mut app = App::new(picker);

    let mut dirty = true;
    loop {
        if dirty {
            // DEC 2026: the terminal must not paint until the whole frame has been
            // written, otherwise it may show half-rewritten image rows.
            queue!(terminal.backend_mut(), BeginSynchronizedUpdate)?;
            terminal.draw(|frame: &mut Frame| app.draw(frame))?;
            execute!(terminal.backend_mut(), EndSynchronizedUpdate)?;
            dirty = false;
        }
        app.request_render_if_needed();

        match event::listen_for_event(INPUT_POLL)? {
            Some(AppEvent::Quit) => return Ok(()),
            Some(event) => {
                app.apply(event);
                dirty = true;
            }
            None => (),
        }
        if app.poll_results() {
            dirty = true;
        }
    }
}
