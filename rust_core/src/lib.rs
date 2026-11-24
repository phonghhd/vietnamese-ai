use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

#[pyfunction]
fn fast_token_count(text: &str) -> PyResult<usize> {
    let count = text.par_split_whitespace().count();
    Ok(count)
}

#[pyfunction]
fn fast_keyword_search(text: &str, keyword: &str) -> PyResult<Vec<usize>> {
    if keyword.is_empty() || text.is_empty() {
        return Ok(Vec::new());
    }
    let kw_chars: Vec<char> = keyword.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    if kw_chars.is_empty() || text_chars.len() < kw_chars.len() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    for i in 0..=(text_chars.len() - kw_chars.len()) {
        if text_chars[i..i + kw_chars.len()] == kw_chars[..] {
            result.push(i);
        }
    }
    Ok(result)
}

#[pyfunction]
fn fast_batch_cosine_similarity(query: Vec<f32>, matrix: Vec<Vec<f32>>) -> PyResult<Vec<f32>> {
    let q_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
    let q_norm = q_norm_sq.sqrt().max(1e-9);

    let scores: Vec<f32> = matrix
        .par_iter()
        .map(|vec| {
            if vec.len() != query.len() {
                return 0.0;
            }
            let mut dot = 0.0;
            let mut v_norm_sq = 0.0;
            for i in 0..query.len() {
                dot += query[i] * vec[i];
                v_norm_sq += vec[i] * vec[i];
            }
            let v_norm = v_norm_sq.sqrt().max(1e-9);
            dot / (q_norm * v_norm)
        })
        .collect();

    Ok(scores)
}

#[inline]
fn read_f32_le(bytes: &[u8], idx: usize) -> f32 {
    let o = idx * 4;
    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

#[pyfunction]
fn fast_top_k_cosine_bytes(
    query_f32_le: &[u8],
    matrix_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if n_docs == 0 || dim == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    if query_f32_le.len() < dim * 4 || matrix_f32_le.len() < n_docs * dim * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm float32 không đủ cho truy vấn hoặc ma trận vector",
        ));
    }
    let query: Vec<f32> = (0..dim).map(|i| read_f32_le(query_f32_le, i)).collect();
    let q_norm = query.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-9);

    let mut scored: Vec<(usize, f32)> = (0..n_docs)
        .into_par_iter()
        .map(|doc_idx| {
            let base = doc_idx * dim;
            let mut dot = 0.0f32;
            let mut v_norm_sq = 0.0f32;
            for j in 0..dim {
                let v = read_f32_le(matrix_f32_le, base + j);
                dot += query[j] * v;
                v_norm_sq += v * v;
            }
            let v_norm = v_norm_sq.sqrt().max(1e-9);
            (doc_idx, dot / (q_norm * v_norm))
        })
        .collect();

    let k = top_k.min(scored.len());
    scored.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored)
}

#[pyfunction]
fn fast_clean_spaces(text: &str) -> PyResult<String> {
    let mut out = String::with_capacity(text.len());
    let mut in_space = false;
    for c in text.chars() {
        if c == '\u{200b}' || c == '\u{feff}' || c == '\u{200c}' || c == '\u{200d}' {
            continue;
        }
        if c.is_whitespace() {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
    Ok(out.trim().to_string())
}

fn tone_map() -> &'static HashMap<&'static str, &'static str> {
    static MAP: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    MAP.get_or_init(|| {
        let pairs: [(&str, &str); 65] = [
            ("hoà", "hòa"), ("hoá", "hóa"), ("hoả", "hỏa"), ("hoã", "hõa"), ("hoạ", "họa"),
            ("toà", "tòa"), ("toá", "tóa"), ("toả", "tỏa"), ("toã", "tõa"), ("toạ", "tọa"),
            ("xoà", "xòa"), ("xoá", "xóa"), ("xoả", "xỏa"), ("xoã", "xõa"), ("xoạ", "xọa"),
            ("loà", "lòa"), ("loá", "lóa"), ("loả", "lỏa"), ("loã", "lõa"), ("loạ", "lọa"),
            ("khoà", "khòa"), ("khoá", "khóa"), ("khoả", "khỏa"), ("khoã", "khõa"), ("khoạ", "khọa"),
            ("khoè", "khòe"), ("khoé", "khóe"), ("khoẻ", "khỏe"), ("khoẽ", "khõe"), ("khoẹ", "khọe"),
            ("hoè", "hòe"), ("hoé", "hóe"), ("hoẻ", "hỏe"), ("hoẽ", "hõe"), ("hoẹ", "họe"),
            ("xoè", "xòe"), ("xoé", "xóe"), ("xoẻ", "xỏe"), ("xoẽ", "xõe"), ("xoẹ", "xọe"),
            ("thuỳ", "thùy"), ("thuý", "thúy"), ("thuỷ", "thủy"), ("thuỹ", "thũy"), ("thuỵ", "thụy"),
            ("tuỳ", "tùy"), ("tuý", "túy"), ("tuỷ", "tủy"), ("tuỹ", "tũy"), ("tuỵ", "tụy"),
            ("huỳ", "hùy"), ("huý", "húy"), ("huỷ", "hủy"), ("huỹ", "hũy"), ("huỵ", "hụy"),
            ("nguỳ", "ngùy"), ("nguý", "ngúy"), ("nguỷ", "ngủy"), ("nguỹ", "ngũy"), ("nguỵ", "ngụy"),
            ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
        ];
        pairs.into_iter().collect()
    })
}

fn preserve_case(orig: &str, repl: &str) -> String {
    let mut chars = orig.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return repl.to_string(),
    };
    let all_upper = orig.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());
    if all_upper {
        return repl.to_uppercase();
    }
    if first.is_uppercase() {
        let mut rchars = repl.chars();
        if let Some(rc) = rchars.next() {
            let mut s: String = rc.to_uppercase().collect();
            s.push_str(rchars.as_str());
            return s;
        }
    }
    repl.to_string()
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<String, usize>,
    is_end: bool,
}

#[pyclass]
struct RustViTokenizer {
    nodes: Vec<TrieNode>,
}

impl RustViTokenizer {
    fn insert_phrase(&mut self, phrase: &str) {
        let mut curr = 0usize;
        let mut count = 0usize;
        for part in phrase.split_whitespace() {
            let key = part.to_lowercase();
            let next_idx = if let Some(&idx) = self.nodes[curr].children.get(&key) {
                idx
            } else {
                let idx = self.nodes.len();
                self.nodes.push(TrieNode::default());
                self.nodes[curr].children.insert(key, idx);
                idx
            };
            curr = next_idx;
            count += 1;
        }
        if count >= 2 {
            self.nodes[curr].is_end = true;
        }
    }
}

#[inline]
fn is_split_punct(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '!' | '?' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '–' | '—' | '…' | '/'
    )
}

#[inline]
fn is_zero_width(c: char) -> bool {
    matches!(c, '\u{200b}' | '\u{feff}' | '\u{200c}' | '\u{200d}')
}

#[pymethods]
impl RustViTokenizer {
    #[new]
    fn new(dict_words: Vec<String>) -> Self {
        let mut tok = Self {
            nodes: vec![TrieNode::default()],
        };
        for w in dict_words {
            tok.insert_phrase(&w);
        }
        tok
    }

    fn add_words(&mut self, dict_words: Vec<String>) {
        for w in dict_words {
            self.insert_phrase(&w);
        }
    }

    fn segment(&self, text: &str, join_compound: bool, sep: &str) -> PyResult<Vec<String>> {
        self.normalize_and_segment(text, join_compound, sep, false)
    }

    fn normalize_and_segment(
        &self,
        text: &str,
        join_compound: bool,
        sep: &str,
        normalize_tone: bool,
    ) -> PyResult<Vec<String>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut orig_tokens: Vec<String> = Vec::with_capacity(text.len() / 4 + 4);
        let mut lower_tokens: Vec<String> = Vec::with_capacity(text.len() / 4 + 4);
        let mut cur = String::new();
        let tmap = if normalize_tone { Some(tone_map()) } else { None };

        let push_word = |buf: &mut String, origs: &mut Vec<String>, lowers: &mut Vec<String>| {
            if buf.is_empty() {
                return;
            }
            let lower = buf.to_lowercase();
            if let Some(m) = tmap {
                if let Some(&repl) = m.get(lower.as_str()) {
                    origs.push(preserve_case(buf, repl));
                    lowers.push(repl.to_string());
                    buf.clear();
                    return;
                }
            }
            origs.push(std::mem::take(buf));
            lowers.push(lower);
        };

        for c in text.chars() {
            if is_zero_width(c) {
                continue;
            }
            if c.is_whitespace() {
                push_word(&mut cur, &mut orig_tokens, &mut lower_tokens);
            } else if is_split_punct(c) {
                push_word(&mut cur, &mut orig_tokens, &mut lower_tokens);
                let mut ps = String::with_capacity(4);
                ps.push(c);
                lower_tokens.push(ps.clone());
                orig_tokens.push(ps);
            } else {
                cur.push(c);
            }
        }
        push_word(&mut cur, &mut orig_tokens, &mut lower_tokens);

        let n = orig_tokens.len();
        let mut out: Vec<String> = Vec::with_capacity(n);
        let mut i = 0usize;
        let joiner = if join_compound { sep } else { " " };

        while i < n {
            let mut curr = 0usize;
            let mut best_end = i;
            let mut j = i;
            while j < n {
                if let Some(&next_idx) = self.nodes[curr].children.get(&lower_tokens[j]) {
                    curr = next_idx;
                    j += 1;
                    if self.nodes[curr].is_end {
                        best_end = j;
                    }
                } else {
                    break;
                }
            }
            if best_end > i + 1 {
                let mut merged = String::new();
                for k in i..best_end {
                    if k > i {
                        merged.push_str(joiner);
                    }
                    merged.push_str(&orig_tokens[k]);
                }
                out.push(merged);
                i = best_end;
            } else {
                out.push(orig_tokens[i].clone());
                i += 1;
            }
        }
        Ok(out)
    }
}

#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x03ff) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let mut m = mant;
        let mut e: i32 = -14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03ff;
        let f32_exp = ((e + 127) as u32) << 23;
        return f32::from_bits(sign | f32_exp | (m << 13));
    }
    if exp == 31 {
        let f32_exp = 0xffu32 << 23;
        return f32::from_bits(sign | f32_exp | (mant << 13));
    }
    let f32_exp = (exp + 112) << 23;
    f32::from_bits(sign | f32_exp | (mant << 13))
}

fn q4_0_matmul_slice(
    x: &[f32],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if k == 0 || k % 32 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Chiều k phải là bội số dương của 32 cho Q4_0",
        ));
    }
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 18;
    if x.len() < rows * k || raw_w.len() < out_cols * bytes_per_row {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q4_0 không đủ",
        ));
    }

    let col_results: Vec<Vec<f32>> = (0..out_cols)
        .into_par_iter()
        .map(|col| {
            let row_bytes = &raw_w[col * bytes_per_row..(col + 1) * bytes_per_row];
            let mut vals = vec![0.0f32; rows];
            for r in 0..rows {
                let x_row = &x[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk = &row_bytes[b * 18..(b + 1) * 18];
                    let scale = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let base = b * 32;
                    let mut blk_sum = 0.0f32;
                    for j in 0..16 {
                        let byte = blk[2 + j];
                        let q_lo = ((byte & 0x0f) as i32 - 8) as f32;
                        let q_hi = (((byte >> 4) & 0x0f) as i32 - 8) as f32;
                        blk_sum += x_row[base + 2 * j] * q_lo + x_row[base + 2 * j + 1] * q_hi;
                    }
                    acc += blk_sum * scale;
                }
                vals[r] = acc;
            }
            vals
        })
        .collect();

    let mut out = vec![0.0f32; rows * out_cols];
    for (col, vals) in col_results.into_iter().enumerate() {
        for r in 0..rows {
            out[r * out_cols + col] = vals[r];
        }
    }
    Ok(out)
}

fn q8_0_matmul_slice(
    x: &[f32],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if k == 0 || k % 32 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Chiều k phải là bội số dương của 32 cho Q8_0",
        ));
    }
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row * 34;
    if x.len() < rows * k || raw_w.len() < out_cols * bytes_per_row {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q8_0 không đủ",
        ));
    }

    let col_results: Vec<Vec<f32>> = (0..out_cols)
        .into_par_iter()
        .map(|col| {
            let row_bytes = &raw_w[col * bytes_per_row..(col + 1) * bytes_per_row];
            let mut vals = vec![0.0f32; rows];
            for r in 0..rows {
                let x_row = &x[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk = &row_bytes[b * 34..(b + 1) * 34];
                    let scale = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let base = b * 32;
                    let mut blk_sum = 0.0f32;
                    for j in 0..32 {
                        let q = (blk[2 + j] as i8) as f32;
                        blk_sum += x_row[base + j] * q;
                    }
                    acc += blk_sum * scale;
                }
                vals[r] = acc;
            }
            vals
        })
        .collect();

    let mut out = vec![0.0f32; rows * out_cols];
    for (col, vals) in col_results.into_iter().enumerate() {
        for r in 0..rows {
            out[r * out_cols + col] = vals[r];
        }
    }
    Ok(out)
}

#[pyfunction]
fn fast_q4_0_matmul(
    x: Vec<f32>,
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    q4_0_matmul_slice(&x, rows, k, raw_w, out_cols)
}

#[pyfunction]
fn fast_q8_0_matmul(
    x: Vec<f32>,
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    q8_0_matmul_slice(&x, rows, k, raw_w, out_cols)
}

#[pyfunction]
fn fast_q4_0_matmul_bytes(
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if x_f32_le.len() < rows * k * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 đầu vào Q4_0 không đủ kích thước",
        ));
    }
    let x: Vec<f32> = (0..rows * k).map(|i| read_f32_le(x_f32_le, i)).collect();
    q4_0_matmul_slice(&x, rows, k, raw_w, out_cols)
}

#[pyfunction]
fn fast_q8_0_matmul_bytes(
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if x_f32_le.len() < rows * k * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 đầu vào Q8_0 không đủ kích thước",
        ));
    }
    let x: Vec<f32> = (0..rows * k).map(|i| read_f32_le(x_f32_le, i)).collect();
    q8_0_matmul_slice(&x, rows, k, raw_w, out_cols)
}

#[pyfunction]
fn fast_q4_k_matmul_bytes(
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if k == 0 || k % 256 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Chiều k phải là bội số dương của 256 cho Q4_K",
        ));
    }
    let blocks_per_row = k / 256;
    let bytes_per_row = blocks_per_row * 144;
    if x_f32_le.len() < rows * k * 4 || raw_w.len() < out_cols * bytes_per_row {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q4_K không đủ",
        ));
    }
    let x: Vec<f32> = (0..rows * k).map(|i| read_f32_le(x_f32_le, i)).collect();

    let col_results: Vec<Vec<f32>> = (0..out_cols)
        .into_par_iter()
        .map(|col| {
            let row_bytes = &raw_w[col * bytes_per_row..(col + 1) * bytes_per_row];
            let mut vals = vec![0.0f32; rows];
            for r in 0..rows {
                let x_row = &x[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for b in 0..blocks_per_row {
                    let blk = &row_bytes[b * 144..(b + 1) * 144];
                    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                    let sc_raw = &blk[4..16];
                    let qs = &blk[16..144];

                    let mut sc = [0.0f32; 8];
                    let mut mn = [0.0f32; 8];
                    for idx in 0..4 {
                        sc[idx] = (sc_raw[idx] & 0x3f) as f32 * d;
                        mn[idx] = (sc_raw[4 + idx] & 0x3f) as f32 * dmin;
                        sc[4 + idx] = ((sc_raw[8 + idx] & 0x0f) | ((sc_raw[idx] >> 6) << 4)) as f32 * d;
                        mn[4 + idx] = ((sc_raw[8 + idx] >> 4) | ((sc_raw[4 + idx] >> 6) << 4)) as f32 * dmin;
                    }

                    let base = b * 256;
                    for g in 0..4 {
                        let q_grp = &qs[g * 32..(g + 1) * 32];
                        let sc_lo = sc[2 * g];
                        let mn_lo = mn[2 * g];
                        let sc_hi = sc[2 * g + 1];
                        let mn_hi = mn[2 * g + 1];
                        let pos_lo = base + g * 64;
                        let pos_hi = pos_lo + 32;
                        let mut sum_lo = 0.0f32;
                        let mut xsum_lo = 0.0f32;
                        let mut sum_hi = 0.0f32;
                        let mut xsum_hi = 0.0f32;
                        for j in 0..32 {
                            let byte = q_grp[j];
                            let xl = x_row[pos_lo + j];
                            let xh = x_row[pos_hi + j];
                            sum_lo += xl * ((byte & 0x0f) as f32);
                            xsum_lo += xl;
                            sum_hi += xh * (((byte >> 4) & 0x0f) as f32);
                            xsum_hi += xh;
                        }
                        acc += sum_lo * sc_lo - xsum_lo * mn_lo + sum_hi * sc_hi - xsum_hi * mn_hi;
                    }
                }
                vals[r] = acc;
            }
            vals
        })
        .collect();

    let mut out = vec![0.0f32; rows * out_cols];
    for (col, vals) in col_results.into_iter().enumerate() {
        for r in 0..rows {
            out[r * out_cols + col] = vals[r];
        }
    }
    Ok(out)
}

const LIMBS: usize = 32;

const RFC5114_P_HEX: &str = concat!(
    "87A8E61DB4B6663CFFBBD19C651959998CEEF608660DD0F2",
    "5D2CEED4435E3B00E00DF8F1D61957D4FAF7DF4561B2AA30",
    "16C3D91134096FAA3BF4296D830E9A7C209E0C6497517ABD",
    "5A8A9D306BCF67ED91F9E6725B4758C022E0B1EF4275BF7B",
    "6C5BFC11D45F9088B941F54EB1E59BB8BC39A0BF12307F5C",
    "4FDB70C581B23F76B63ACAE1CAA6B7902D52526735488A0E",
    "F13C6D9A51BFA4AB3AD8347796524D8EF6A167B5A41825D9",
    "67E144E5140564251CCACB83E6B486F6B3CA3F7971506026",
    "C0B857F689962856DED4010ABD0BE621C3A3960A54E710C3",
    "75F26375D7014103A4B54330C198AF126116D2276E11715F",
    "693877FAD7EF09CADB094AE91E1A1597"
);

struct MontCtx {
    p: [u64; LIMBS],
    n0_prime: u64,
    r_mod_p: [u64; LIMBS],
    r2_mod_p: [u64; LIMBS],
}

fn be_bytes_to_limbs(bytes: &[u8]) -> [u64; LIMBS] {
    let mut limbs = [0u64; LIMBS];
    let len = bytes.len();
    for i in 0..LIMBS {
        let end = len.saturating_sub(i * 8);
        let start = end.saturating_sub(8);
        let mut val = 0u64;
        for &b in &bytes[start..end] {
            val = (val << 8) | (b as u64);
        }
        limbs[i] = val;
    }
    limbs
}

fn limbs_to_be_bytes(limbs: &[u64; LIMBS]) -> Vec<u8> {
    let mut out = vec![0u8; LIMBS * 8];
    for i in 0..LIMBS {
        let be = limbs[LIMBS - 1 - i].to_be_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&be);
    }
    out
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let b = hex.as_bytes();
    let mut out = Vec::with_capacity(b.len() / 2);
    for chunk in b.chunks(2) {
        let hi = (chunk[0] as char).to_digit(16).unwrap_or(0) as u8;
        let lo = (chunk[1] as char).to_digit(16).unwrap_or(0) as u8;
        out.push((hi << 4) | lo);
    }
    out
}

#[inline]
fn ct_sub_if_ge(a: &mut [u64; LIMBS], extra: u64, p: &[u64; LIMBS]) {
    let mut diff = [0u64; LIMBS];
    let mut borrow = 0u128;
    for i in 0..LIMBS {
        let av = a[i] as u128;
        let pv = p[i] as u128 + borrow;
        if av < pv {
            diff[i] = ((1u128 << 64) + av - pv) as u64;
            borrow = 1;
        } else {
            diff[i] = (av - pv) as u64;
            borrow = 0;
        }
    }
    let underflow = (extra as u128) < borrow;
    let mask = if underflow { 0u64 } else { u64::MAX };
    let inv_mask = !mask;
    for i in 0..LIMBS {
        a[i] = (diff[i] & mask) | (a[i] & inv_mask);
    }
}

fn double_mod(a: &mut [u64; LIMBS], p: &[u64; LIMBS]) {
    let mut carry = 0u64;
    for i in 0..LIMBS {
        let next_carry = a[i] >> 63;
        a[i] = (a[i] << 1) | carry;
        carry = next_carry;
    }
    ct_sub_if_ge(a, carry, p);
}

fn rfc5114_ctx() -> &'static MontCtx {
    static CTX: OnceLock<MontCtx> = OnceLock::new();
    CTX.get_or_init(|| {
        let p_bytes = hex_to_bytes(RFC5114_P_HEX);
        let p = be_bytes_to_limbs(&p_bytes);
        let mut inv = 1u64;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(p[0].wrapping_mul(inv)));
        }
        let n0_prime = inv.wrapping_neg();

        let mut val = [0u64; LIMBS];
        val[0] = 1;
        for _ in 0..2048 {
            double_mod(&mut val, &p);
        }
        let r_mod_p = val;
        for _ in 0..2048 {
            double_mod(&mut val, &p);
        }
        let r2_mod_p = val;
        MontCtx {
            p,
            n0_prime,
            r_mod_p,
            r2_mod_p,
        }
    })
}

fn mont_mul(a: &[u64; LIMBS], b: &[u64; LIMBS], ctx: &MontCtx) -> [u64; LIMBS] {
    let mut t = [0u64; LIMBS + 2];
    let p = &ctx.p;
    let n0 = ctx.n0_prime;

    for i in 0..LIMBS {
        let ai = a[i] as u128;
        let mut carry = 0u128;
        for j in 0..LIMBS {
            let s = (t[j] as u128) + ai * (b[j] as u128) + carry;
            t[j] = s as u64;
            carry = s >> 64;
        }
        let s = (t[LIMBS] as u128) + carry;
        t[LIMBS] = s as u64;
        t[LIMBS + 1] = (s >> 64) as u64;

        let m = t[0].wrapping_mul(n0) as u128;
        let s0 = (t[0] as u128) + m * (p[0] as u128);
        let mut carry2 = s0 >> 64;
        for j in 1..LIMBS {
            let s = (t[j] as u128) + m * (p[j] as u128) + carry2;
            t[j - 1] = s as u64;
            carry2 = s >> 64;
        }
        let s_last = (t[LIMBS] as u128) + carry2;
        t[LIMBS - 1] = s_last as u64;
        t[LIMBS] = ((t[LIMBS + 1] as u128) + (s_last >> 64)) as u64;
        t[LIMBS + 1] = 0;
    }

    let mut res = [0u64; LIMBS];
    res.copy_from_slice(&t[..LIMBS]);
    ct_sub_if_ge(&mut res, t[LIMBS], p);
    res
}

fn ct_select_window(table: &[[u64; LIMBS]; 16], idx: u8) -> [u64; LIMBS] {
    let mut out = [0u64; LIMBS];
    for k in 0..16u8 {
        let eq = ((k ^ idx) == 0) as u64;
        let mask = eq.wrapping_neg();
        for limb in 0..LIMBS {
            out[limb] |= table[k as usize][limb] & mask;
        }
    }
    out
}

#[pyfunction]
fn fast_rfc5114_modpow(base_be: &[u8], exp_be: &[u8]) -> PyResult<Vec<u8>> {
    if base_be.len() > 256 || exp_be.len() > 32 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Đầu vào RFC 5114 phải có độ dài tối đa 256 bytes (base) và 32 bytes (exp)",
        ));
    }
    let ctx = rfc5114_ctx();
    let mut base = be_bytes_to_limbs(base_be);
    ct_sub_if_ge(&mut base, 0, &ctx.p);

    let a_mont = mont_mul(&base, &ctx.r2_mod_p, ctx);
    let mut table = [[0u64; LIMBS]; 16];
    table[0] = ctx.r_mod_p;
    table[1] = a_mont;
    for k in 2..16 {
        table[k] = mont_mul(&table[k - 1], &a_mont, ctx);
    }

    let mut acc = ctx.r_mod_p;
    for &byte in exp_be {
        let hi = (byte >> 4) & 0x0f;
        let lo = byte & 0x0f;
        for _ in 0..4 {
            acc = mont_mul(&acc, &acc, ctx);
        }
        let w_hi = ct_select_window(&table, hi);
        acc = mont_mul(&acc, &w_hi, ctx);

        for _ in 0..4 {
            acc = mont_mul(&acc, &acc, ctx);
        }
        let w_lo = ct_select_window(&table, lo);
        acc = mont_mul(&acc, &w_lo, ctx);
    }

    let mut one = [0u64; LIMBS];
    one[0] = 1;
    let result = mont_mul(&acc, &one, ctx);
    Ok(limbs_to_be_bytes(&result))
}

#[pymodule]
fn vietnamese_ai_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_token_count, m)?)?;
    m.add_function(wrap_pyfunction!(fast_keyword_search, m)?)?;
    m.add_function(wrap_pyfunction!(fast_batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(fast_top_k_cosine_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_clean_spaces, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_0_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q8_0_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_0_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q8_0_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_k_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_rfc5114_modpow, m)?)?;
    m.add_class::<RustViTokenizer>()?;
    Ok(())
}
