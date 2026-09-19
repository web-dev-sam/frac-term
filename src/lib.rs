mod event;
mod mandelbrot;

use num_complex::Complex64;
use ratatui::{DefaultTerminal, Frame, layout::Size};
use ratatui_image::{
    FontSize, Image, Resize,
    picker::{Picker, ProtocolType, cap_parser},
};

use crate::event::AppEvent;

struct App {
    ratatui_image_picker: ratatui_image::picker::Picker,
    zoom: f64,
    position: Complex64,
    max_iterations: u16,
}

impl App {
    /// Initialize app state
    pub fn new() -> Self {
        let mut options = cap_parser::QueryStdioOptions::default();
        if std::env::var("TERM_PROGRAM").is_ok_and(|p| p.contains("vscode")) {
            options.blacklist_protocols.push(ProtocolType::Kitty);
        }
        let picker =
            Picker::from_query_stdio_with_options(options).unwrap_or_else(|_| Picker::halfblocks());

        Self {
            zoom: 1.0,
            position: Complex64::new(0.0, 0.0),
            max_iterations: 100,
            ratatui_image_picker: picker,
        }
    }

    /// Listens for the next user interaction to adjust app state
    pub fn wait_for_interaction(&mut self) -> Result<(), std::io::Error> {
        match event::listen_for_event()? {
            Some(AppEvent::ZoomIncrease) => self.zoom *= 1.2,
            Some(AppEvent::ZoomDecrease) => self.zoom /= 1.2,
            Some(AppEvent::MoveLeft) => self.position.re -= (2.0 / self.zoom) / 8.0, // 1/8 of half width
            Some(AppEvent::MoveRight) => self.position.re += (2.0 / self.zoom) / 8.0, // 1/8 of half width
            Some(AppEvent::MoveDown) => self.position.im -= (2.0 / self.zoom) / 8.0, // 1/8 of half width
            Some(AppEvent::MoveUp) => self.position.im += (2.0 / self.zoom) / 8.0, // 1/8 of half width
            Some(AppEvent::Quit) => std::process::exit(0),
            None => (),
        };
        Ok(())
    }
}

pub fn run(terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    let mut app = App::new();
    loop {
        terminal.draw(|frame: &mut Frame| {
            let area = frame.area();
            let FontSize { width, height } = app.ratatui_image_picker.font_size();
            let size = Size::new(area.width * width, area.height * height);

            let mandelbrot_img = mandelbrot::render(&app, &size);

            let image_protocol = app
                .ratatui_image_picker
                .new_protocol(mandelbrot_img, size, Resize::Fit(None))
                .expect("ratatui_image::picker::Picker::new_protocol() succeeds");
            frame.render_widget(Image::new(&image_protocol), area)
        })?;

        app.wait_for_interaction()?;
    }
}
