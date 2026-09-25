//! Kitty graphics protocol image widget with a PNG payload.
//!
//! ratatui-image transmits Kitty images as raw RGBA, which at full-screen
//! resolution is ~11 MB of base64 per frame through the PTY. PNG cuts that by
//! an order of magnitude for fractal renders. Placement uses Unicode
//! placeholders (virtual placement, `U=1`) exactly like ratatui-image, so the
//! terminal behaves identically; only the transmission differs.

use std::fmt::Write as _;
use std::io::Cursor;
use std::num::NonZeroU16;
use std::sync::atomic::{AtomicBool, Ordering};

use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{DynamicImage, ImageEncoder};
use ratatui::buffer::{Buffer, CellDiffOption};
use ratatui::layout::{Rect, Size};
use ratatui::widgets::Widget;

/// Kitty's placeholder codepoint; row/column are encoded as combining diacritics.
const PLACEHOLDER: char = '\u{10EEEE}';
/// Max base64 payload per chunk allowed by the protocol.
const CHUNK_CHARS: usize = 4096;
const UNIT_WIDTH: CellDiffOption = CellDiffOption::ForcedWidth(NonZeroU16::new(1).unwrap());

pub struct KittyImage {
    transmit: String,
    transmitted: AtomicBool,
    /// SGR sequence setting the fg color that carries the low 24 bits of the id.
    id_color: String,
    /// High byte of the id, carried by the third diacritic.
    id_extra: u16,
    /// Cell area the image is meant to cover.
    size: Size,
}

impl KittyImage {
    /// Encodes `img` as PNG and builds the chunked transmit sequence for image
    /// `id`. `previous` is the id shown by the last frame; it is freed in the
    /// same write so the terminal does not accumulate one image per frame.
    pub fn new(img: &DynamicImage, size: Size, id: u32, previous: Option<u32>) -> Self {
        let rgb = img.to_rgb8();
        let mut png = Vec::with_capacity(rgb.len() / 4);
        PngEncoder::new_with_quality(Cursor::new(&mut png), CompressionType::Fast, FilterType::Up)
            .write_image(
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                image::ExtendedColorType::Rgb8,
            )
            .expect("PNG encoding into a Vec cannot fail");

        let b64 = base64_simd::STANDARD.encode_to_string(&png);
        let mut transmit = String::with_capacity(b64.len() + b64.len() / CHUNK_CHARS * 24 + 64);
        let chunks: Vec<&str> = b64
            .as_bytes()
            .chunks(CHUNK_CHARS)
            // base64 is ASCII, so byte chunks are valid str boundaries
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect();
        for (i, chunk) in chunks.iter().enumerate() {
            transmit.push_str("\x1b_Gq=2,");
            if i == 0 {
                write!(transmit, "i={id},a=T,U=1,f=100,").unwrap();
            }
            let more = u8::from(i + 1 < chunks.len());
            write!(transmit, "m={more};{chunk}\x1b\\").unwrap();
        }
        if let Some(old) = previous {
            // d=I: delete the image data and all its placements by id.
            write!(transmit, "\x1b_Gq=2,a=d,d=I,i={old}\x1b\\").unwrap();
        }

        let [id_extra, r, g, b] = id.to_be_bytes();
        Self {
            transmit,
            transmitted: AtomicBool::new(false),
            id_color: format!("\x1b[38;2;{r};{g};{b}m"),
            id_extra: u16::from(id_extra),
            size,
        }
    }
}

impl Widget for &KittyImage {
    /// Each row is written into its first cell as one symbol: the (once-only)
    /// transmit sequence, then a run of placeholders carrying row/column
    /// diacritics, then a cursor restore so the terminal ends up where ratatui
    /// expects. Remaining cells in the row are marked skip so ratatui never
    /// overwrites the placeholders.
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = area.width.min(self.size.width);
        let height = area
            .height
            .min(self.size.height)
            .min(DIACRITICS.len() as u16);
        if width == 0 || height == 0 {
            return;
        }
        let mut transmit =
            (!self.transmitted.swap(true, Ordering::SeqCst)).then_some(self.transmit.as_str());

        let row_fill: String = std::iter::repeat_n(PLACEHOLDER, usize::from(width) - 1).collect();
        let restore_cursor = format!("\x1b[u\x1b[{}C\x1b[{}B", area.width - 1, area.height - 1);

        let mut symbol = String::new();
        for y in 0..height {
            symbol.clear();
            if let Some(seq) = transmit.take() {
                symbol.push_str(seq);
            }
            // Save cursor (incl. color), then first placeholder with explicit
            // row / column 0 / id-extra; the rest inherit them.
            write!(
                symbol,
                "\x1b[s{}{PLACEHOLDER}{}{}{}{row_fill}{restore_cursor}",
                self.id_color,
                diacritic(y),
                diacritic(0),
                diacritic(self.id_extra),
            )
            .unwrap();

            for x in 1..width {
                if let Some(cell) = buf.cell_mut((area.left() + x, area.top() + y)) {
                    cell.set_diff_option(CellDiffOption::Skip);
                }
            }
            if let Some(cell) = buf.cell_mut((area.left(), area.top() + y)) {
                cell.set_symbol(&symbol).set_diff_option(UNIT_WIDTH);
            }
        }
    }
}

#[inline]
fn diacritic(i: u16) -> char {
    DIACRITICS[usize::from(i).min(DIACRITICS.len() - 1)]
}

/// Row/column diacritics from the Kitty spec:
/// <https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders>
static DIACRITICS: [char; 297] = [
    '\u{305}',
    '\u{30D}',
    '\u{30E}',
    '\u{310}',
    '\u{312}',
    '\u{33D}',
    '\u{33E}',
    '\u{33F}',
    '\u{346}',
    '\u{34A}',
    '\u{34B}',
    '\u{34C}',
    '\u{350}',
    '\u{351}',
    '\u{352}',
    '\u{357}',
    '\u{35B}',
    '\u{363}',
    '\u{364}',
    '\u{365}',
    '\u{366}',
    '\u{367}',
    '\u{368}',
    '\u{369}',
    '\u{36A}',
    '\u{36B}',
    '\u{36C}',
    '\u{36D}',
    '\u{36E}',
    '\u{36F}',
    '\u{483}',
    '\u{484}',
    '\u{485}',
    '\u{486}',
    '\u{487}',
    '\u{592}',
    '\u{593}',
    '\u{594}',
    '\u{595}',
    '\u{597}',
    '\u{598}',
    '\u{599}',
    '\u{59C}',
    '\u{59D}',
    '\u{59E}',
    '\u{59F}',
    '\u{5A0}',
    '\u{5A1}',
    '\u{5A8}',
    '\u{5A9}',
    '\u{5AB}',
    '\u{5AC}',
    '\u{5AF}',
    '\u{5C4}',
    '\u{610}',
    '\u{611}',
    '\u{612}',
    '\u{613}',
    '\u{614}',
    '\u{615}',
    '\u{616}',
    '\u{617}',
    '\u{657}',
    '\u{658}',
    '\u{659}',
    '\u{65A}',
    '\u{65B}',
    '\u{65D}',
    '\u{65E}',
    '\u{6D6}',
    '\u{6D7}',
    '\u{6D8}',
    '\u{6D9}',
    '\u{6DA}',
    '\u{6DB}',
    '\u{6DC}',
    '\u{6DF}',
    '\u{6E0}',
    '\u{6E1}',
    '\u{6E2}',
    '\u{6E4}',
    '\u{6E7}',
    '\u{6E8}',
    '\u{6EB}',
    '\u{6EC}',
    '\u{730}',
    '\u{732}',
    '\u{733}',
    '\u{735}',
    '\u{736}',
    '\u{73A}',
    '\u{73D}',
    '\u{73F}',
    '\u{740}',
    '\u{741}',
    '\u{743}',
    '\u{745}',
    '\u{747}',
    '\u{749}',
    '\u{74A}',
    '\u{7EB}',
    '\u{7EC}',
    '\u{7ED}',
    '\u{7EE}',
    '\u{7EF}',
    '\u{7F0}',
    '\u{7F1}',
    '\u{7F3}',
    '\u{816}',
    '\u{817}',
    '\u{818}',
    '\u{819}',
    '\u{81B}',
    '\u{81C}',
    '\u{81D}',
    '\u{81E}',
    '\u{81F}',
    '\u{820}',
    '\u{821}',
    '\u{822}',
    '\u{823}',
    '\u{825}',
    '\u{826}',
    '\u{827}',
    '\u{829}',
    '\u{82A}',
    '\u{82B}',
    '\u{82C}',
    '\u{82D}',
    '\u{951}',
    '\u{953}',
    '\u{954}',
    '\u{F82}',
    '\u{F83}',
    '\u{F86}',
    '\u{F87}',
    '\u{135D}',
    '\u{135E}',
    '\u{135F}',
    '\u{17DD}',
    '\u{193A}',
    '\u{1A17}',
    '\u{1A75}',
    '\u{1A76}',
    '\u{1A77}',
    '\u{1A78}',
    '\u{1A79}',
    '\u{1A7A}',
    '\u{1A7B}',
    '\u{1A7C}',
    '\u{1B6B}',
    '\u{1B6D}',
    '\u{1B6E}',
    '\u{1B6F}',
    '\u{1B70}',
    '\u{1B71}',
    '\u{1B72}',
    '\u{1B73}',
    '\u{1CD0}',
    '\u{1CD1}',
    '\u{1CD2}',
    '\u{1CDA}',
    '\u{1CDB}',
    '\u{1CE0}',
    '\u{1DC0}',
    '\u{1DC1}',
    '\u{1DC3}',
    '\u{1DC4}',
    '\u{1DC5}',
    '\u{1DC6}',
    '\u{1DC7}',
    '\u{1DC8}',
    '\u{1DC9}',
    '\u{1DCB}',
    '\u{1DCC}',
    '\u{1DD1}',
    '\u{1DD2}',
    '\u{1DD3}',
    '\u{1DD4}',
    '\u{1DD5}',
    '\u{1DD6}',
    '\u{1DD7}',
    '\u{1DD8}',
    '\u{1DD9}',
    '\u{1DDA}',
    '\u{1DDB}',
    '\u{1DDC}',
    '\u{1DDD}',
    '\u{1DDE}',
    '\u{1DDF}',
    '\u{1DE0}',
    '\u{1DE1}',
    '\u{1DE2}',
    '\u{1DE3}',
    '\u{1DE4}',
    '\u{1DE5}',
    '\u{1DE6}',
    '\u{1DFE}',
    '\u{20D0}',
    '\u{20D1}',
    '\u{20D4}',
    '\u{20D5}',
    '\u{20D6}',
    '\u{20D7}',
    '\u{20DB}',
    '\u{20DC}',
    '\u{20E1}',
    '\u{20E7}',
    '\u{20E9}',
    '\u{20F0}',
    '\u{2CEF}',
    '\u{2CF0}',
    '\u{2CF1}',
    '\u{2DE0}',
    '\u{2DE1}',
    '\u{2DE2}',
    '\u{2DE3}',
    '\u{2DE4}',
    '\u{2DE5}',
    '\u{2DE6}',
    '\u{2DE7}',
    '\u{2DE8}',
    '\u{2DE9}',
    '\u{2DEA}',
    '\u{2DEB}',
    '\u{2DEC}',
    '\u{2DED}',
    '\u{2DEE}',
    '\u{2DEF}',
    '\u{2DF0}',
    '\u{2DF1}',
    '\u{2DF2}',
    '\u{2DF3}',
    '\u{2DF4}',
    '\u{2DF5}',
    '\u{2DF6}',
    '\u{2DF7}',
    '\u{2DF8}',
    '\u{2DF9}',
    '\u{2DFA}',
    '\u{2DFB}',
    '\u{2DFC}',
    '\u{2DFD}',
    '\u{2DFE}',
    '\u{2DFF}',
    '\u{A66F}',
    '\u{A67C}',
    '\u{A67D}',
    '\u{A6F0}',
    '\u{A6F1}',
    '\u{A8E0}',
    '\u{A8E1}',
    '\u{A8E2}',
    '\u{A8E3}',
    '\u{A8E4}',
    '\u{A8E5}',
    '\u{A8E6}',
    '\u{A8E7}',
    '\u{A8E8}',
    '\u{A8E9}',
    '\u{A8EA}',
    '\u{A8EB}',
    '\u{A8EC}',
    '\u{A8ED}',
    '\u{A8EE}',
    '\u{A8EF}',
    '\u{A8F0}',
    '\u{A8F1}',
    '\u{AAB0}',
    '\u{AAB2}',
    '\u{AAB3}',
    '\u{AAB7}',
    '\u{AAB8}',
    '\u{AABE}',
    '\u{AABF}',
    '\u{AAC1}',
    '\u{FE20}',
    '\u{FE21}',
    '\u{FE22}',
    '\u{FE23}',
    '\u{FE24}',
    '\u{FE25}',
    '\u{FE26}',
    '\u{10A0F}',
    '\u{10A38}',
    '\u{1D185}',
    '\u{1D186}',
    '\u{1D187}',
    '\u{1D188}',
    '\u{1D189}',
    '\u{1D1AA}',
    '\u{1D1AB}',
    '\u{1D1AC}',
    '\u{1D1AD}',
    '\u{1D242}',
    '\u{1D243}',
    '\u{1D244}',
];

#[cfg(test)]
mod tests {
    use super::*;

    fn image() -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_fn(64, 32, |x, y| {
            image::Rgb([x as u8, y as u8, 128])
        }))
    }

    /// Layout per the Kitty spec: transmit once, then per row a saved cursor,
    /// the id as fg color, a placeholder with row/column/id-extra diacritics,
    /// width-1 bare placeholders, and a cursor restore. Other cells are skipped.
    #[test]
    fn placeholder_layout_follows_spec() {
        let id = 0x0102_0304;
        let k = KittyImage::new(&image(), Size::new(8, 4), id, Some(5));
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 4));
        Widget::render(&k, buf.area, &mut buf);

        let first = buf.cell((0, 0)).unwrap().symbol();
        assert!(first.starts_with(&format!("\x1b_Gq=2,i={id},a=T,U=1,f=100,m=")));
        assert!(
            first.contains("\x1b_Gq=2,a=d,d=I,i=5\x1b\\"),
            "previous image freed"
        );
        assert!(
            first.contains("\x1b[s\x1b[38;2;2;3;4m\u{10EEEE}"),
            "id low bytes as color"
        );
        assert!(first.ends_with("\x1b[u\x1b[7C\x1b[3B"));
        assert_eq!(buf.cell((1, 0)).unwrap().diff_option, CellDiffOption::Skip);

        let row2 = buf.cell((0, 2)).unwrap().symbol();
        let expected = format!(
            "\x1b[s\x1b[38;2;2;3;4m{PLACEHOLDER}{}{}{}",
            DIACRITICS[2], DIACRITICS[0], DIACRITICS[1]
        );
        assert!(
            row2.starts_with(&expected),
            "row 2 diacritics: row, col 0, id extra"
        );
        assert_eq!(row2.chars().filter(|&c| c == PLACEHOLDER).count(), 8);
    }

    #[test]
    fn transmits_only_once() {
        let k = KittyImage::new(&image(), Size::new(8, 4), 9, None);
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 4));
        Widget::render(&k, buf.area, &mut buf);
        assert!(buf.cell((0, 0)).unwrap().symbol().starts_with("\x1b_G"));
        let mut buf = Buffer::empty(Rect::new(0, 0, 8, 4));
        Widget::render(&k, buf.area, &mut buf);
        assert!(buf.cell((0, 0)).unwrap().symbol().starts_with("\x1b[s"));
    }

    #[test]
    fn payload_is_png_chunked_to_spec() {
        let k = KittyImage::new(&image(), Size::new(8, 4), 9, None);
        let chunks: Vec<&str> = k
            .transmit
            .split("\x1b\\")
            .filter(|c| !c.is_empty())
            .collect();
        for chunk in &chunks {
            let payload = chunk.rsplit(';').next().unwrap();
            assert!(payload.len() <= CHUNK_CHARS);
        }
        let body: String = chunks
            .iter()
            .map(|c| c.rsplit(';').next().unwrap())
            .collect();
        let png = base64_simd::STANDARD.decode_to_vec(body).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert!(chunks.last().unwrap().contains("m=0;"));
    }
}
