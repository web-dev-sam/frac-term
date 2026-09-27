use frac::Start;

const USAGE: &str = "\
Usage: frac [OPTIONS]

Open the viewer at a position. Values are the ones printed in the status line.

Options:
  --zoom <ZOOM>          Magnification, e.g. 2.065e35 (default 1)
  --re <RE>              Real part of the center as a decimal (default 0)
  --im <IM>              Imaginary part of the center as a decimal (default 0)
  --iterations <N>       Maximum iterations per pixel (default 100)
  -h, --help             Print this help

Keys:
  w / s                  Zoom in / out
  Arrow keys             Pan (also, a / d = left / right)
  + / -                  More / fewer iterations
  q, Esc                 Quit
";

fn parse_args() -> anyhow::Result<Start> {
    let (mut zoom, mut re, mut im, mut iterations) = (1.0, "0".to_owned(), "0".to_owned(), 100);
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "-h" || flag == "--help" {
            print!("{USAGE}");
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| anyhow::anyhow!("{flag} needs a value\n\n{USAGE}"))?;
        match flag.as_str() {
            "--zoom" => zoom = value.parse()?,
            "--re" => re = value,
            "--im" => im = value,
            "--iterations" => iterations = value.parse()?,
            _ => anyhow::bail!("unknown option {flag}\n\n{USAGE}"),
        }
    }
    Start::new(zoom, &re, &im, iterations)
}

fn main() -> anyhow::Result<()> {
    let start = parse_args()?;
    ratatui::run(|terminal| frac::run(terminal, &start))
}
