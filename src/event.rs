use std::time::Duration;

use crossterm::event::{Event, KeyCode};

pub enum AppEvent {
    ZoomIncrease,
    ZoomDecrease,
    IterationIncrease,
    IterationDecrease,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Resize,
    Quit,
}

pub fn listen_for_event(timeout: Duration) -> Result<Option<AppEvent>, std::io::Error> {
    if !crossterm::event::poll(timeout)? {
        return Ok(None);
    }
    Ok(match crossterm::event::read()? {
        Event::Key(ev) => match ev.code {
            KeyCode::Left | KeyCode::Char('a') => Some(AppEvent::MoveLeft),
            KeyCode::Right | KeyCode::Char('d') => Some(AppEvent::MoveRight),
            KeyCode::Up => Some(AppEvent::MoveUp),
            KeyCode::Down => Some(AppEvent::MoveDown),
            KeyCode::Char('w') => Some(AppEvent::ZoomIncrease),
            KeyCode::Char('s') => Some(AppEvent::ZoomDecrease),
            KeyCode::Char('+') => Some(AppEvent::IterationIncrease),
            KeyCode::Char('-') => Some(AppEvent::IterationDecrease),
            KeyCode::Char('q') | KeyCode::Esc => Some(AppEvent::Quit),
            _ => None,
        },
        Event::Resize(..) => Some(AppEvent::Resize),
        _ => None,
    })
}
