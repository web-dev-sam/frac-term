use crossterm::event::KeyCode;

pub enum AppEvent {
    ZoomIncrease,
    ZoomDecrease,
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
    Quit,
}

pub fn listen_for_event() -> Result<Option<AppEvent>, std::io::Error> {
    match crossterm::event::read()? {
        crossterm::event::Event::Key(ev) => match ev.code {
            KeyCode::Left => Ok(Some(AppEvent::MoveLeft)),
            KeyCode::Right => Ok(Some(AppEvent::MoveRight)),
            KeyCode::Up => Ok(Some(AppEvent::MoveUp)),
            KeyCode::Down => Ok(Some(AppEvent::MoveDown)),
            KeyCode::Char('w') => Ok(Some(AppEvent::ZoomIncrease)),
            KeyCode::Char('s') => Ok(Some(AppEvent::ZoomDecrease)),
            KeyCode::Char('q') => Ok(Some(AppEvent::Quit)),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}
