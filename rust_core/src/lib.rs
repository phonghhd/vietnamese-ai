use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

#[pyfunction]
fn fast_token_count(py: Python<'_>, text: &str) -> PyResult<usize> {
    let count = py.allow_threads(|| text.par_split_whitespace().count());
    Ok(count)
}

#[pyfunction]
fn fast_keyword_search(py: Python<'_>, text: &str, keyword: &str) -> PyResult<Vec<usize>> {
    if keyword.is_empty() || text.is_empty() {
        return Ok(Vec::new());
    }
    let result = py.allow_threads(|| {
        let kw_chars: Vec<char> = keyword.chars().collect();
        let text_chars: Vec<char> = text.chars().collect();
        if kw_chars.is_empty() || text_chars.len() < kw_chars.len() {
            return Vec::new();
        }
        let mut matches = Vec::new();
        for i in 0..=(text_chars.len() - kw_chars.len()) {
            if text_chars[i..i + kw_chars.len()] == kw_chars[..] {
                matches.push(i);
            }
        }
        matches
    });
    Ok(result)
}

#[pyfunction]
fn fast_batch_cosine_similarity(
    py: Python<'_>,
    query: Vec<f32>,
    matrix: Vec<Vec<f32>>,
) -> PyResult<Vec<f32>> {
    let scores = py.allow_threads(|| {
        let q_norm_sq: f32 = query.iter().map(|&x| x * x).sum();
        let q_norm = q_norm_sq.sqrt().max(1e-9);

        matrix
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
            .collect()
    });
    Ok(scores)
}

#[inline]
fn read_f32_le(bytes: &[u8], idx: usize) -> f32 {
    let o = idx * 4;
    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

#[pyfunction]
fn fast_top_k_cosine_bytes(
    py: Python<'_>,
    query_f32_le: &[u8],
    matrix_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if n_docs == 0 || dim == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let query_bytes_req = dim
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước vector truy vấn"))?;
    let matrix_bytes_req = n_docs
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận vector"))?;
    if query_f32_le.len() < query_bytes_req || matrix_f32_le.len() < matrix_bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm float32 không đủ cho truy vấn hoặc ma trận vector",
        ));
    }
    let scored = py.allow_threads(|| {
        let query: Vec<f32> = (0..dim).map(|i| read_f32_le(query_f32_le, i)).collect();
        let q_norm = query.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-9);

        let mut res: Vec<(usize, f32)> = (0..n_docs)
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

        let k = top_k.min(res.len());
        res.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        res.truncate(k);
        res.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        res
    });
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

fn split_vietnamese_vowel(c: char) -> Option<(char, u8)> {
    match c {
        'a' => Some(('a', 0)), 'à' => Some(('a', 1)), 'á' => Some(('a', 2)), 'ả' => Some(('a', 3)), 'ã' => Some(('a', 4)), 'ạ' => Some(('a', 5)),
        'ă' => Some(('ă', 0)), 'ằ' => Some(('ă', 1)), 'ắ' => Some(('ă', 2)), 'ẳ' => Some(('ă', 3)), 'ẵ' => Some(('ă', 4)), 'ặ' => Some(('ă', 5)),
        'â' => Some(('â', 0)), 'ầ' => Some(('â', 1)), 'ấ' => Some(('â', 2)), 'ẩ' => Some(('â', 3)), 'ẫ' => Some(('â', 4)), 'ậ' => Some(('â', 5)),
        'e' => Some(('e', 0)), 'è' => Some(('e', 1)), 'é' => Some(('e', 2)), 'ẻ' => Some(('e', 3)), 'ẽ' => Some(('e', 4)), 'ẹ' => Some(('e', 5)),
        'ê' => Some(('ê', 0)), 'ề' => Some(('ê', 1)), 'ế' => Some(('ê', 2)), 'ể' => Some(('ê', 3)), 'ễ' => Some(('ê', 4)), 'ệ' => Some(('ê', 5)),
        'i' => Some(('i', 0)), 'ì' => Some(('i', 1)), 'í' => Some(('i', 2)), 'ỉ' => Some(('i', 3)), 'ĩ' => Some(('i', 4)), 'ị' => Some(('i', 5)),
        'o' => Some(('o', 0)), 'ò' => Some(('o', 1)), 'ó' => Some(('o', 2)), 'ỏ' => Some(('o', 3)), 'õ' => Some(('o', 4)), 'ọ' => Some(('o', 5)),
        'ô' => Some(('ô', 0)), 'ồ' => Some(('ô', 1)), 'ố' => Some(('ô', 2)), 'ổ' => Some(('ô', 3)), 'ỗ' => Some(('ô', 4)), 'ộ' => Some(('ô', 5)),
        'ơ' => Some(('ơ', 0)), 'ờ' => Some(('ơ', 1)), 'ớ' => Some(('ơ', 2)), 'ở' => Some(('ơ', 3)), 'ỡ' => Some(('ơ', 4)), 'ợ' => Some(('ơ', 5)),
        'u' => Some(('u', 0)), 'ù' => Some(('u', 1)), 'ú' => Some(('u', 2)), 'ủ' => Some(('u', 3)), 'ũ' => Some(('u', 4)), 'ụ' => Some(('u', 5)),
        'ư' => Some(('ư', 0)), 'ừ' => Some(('ư', 1)), 'ứ' => Some(('ư', 2)), 'ử' => Some(('ư', 3)), 'ữ' => Some(('ư', 4)), 'ự' => Some(('ư', 5)),
        'y' => Some(('y', 0)), 'ỳ' => Some(('y', 1)), 'ý' => Some(('y', 2)), 'ỷ' => Some(('y', 3)), 'ỹ' => Some(('y', 4)), 'ỵ' => Some(('y', 5)),
        _ => None,
    }
}

fn apply_vietnamese_tone(base: char, tone: u8) -> char {
    match (base, tone) {
        ('a', 1) => 'à', ('a', 2) => 'á', ('a', 3) => 'ả', ('a', 4) => 'ã', ('a', 5) => 'ạ',
        ('ă', 1) => 'ằ', ('ă', 2) => 'ắ', ('ă', 3) => 'ẳ', ('ă', 4) => 'ẵ', ('ă', 5) => 'ặ',
        ('â', 1) => 'ầ', ('â', 2) => 'ấ', ('â', 3) => 'ẩ', ('â', 4) => 'ẫ', ('â', 5) => 'ậ',
        ('e', 1) => 'è', ('e', 2) => 'é', ('e', 3) => 'ẻ', ('e', 4) => 'ẽ', ('e', 5) => 'ẹ',
        ('ê', 1) => 'ề', ('ê', 2) => 'ế', ('ê', 3) => 'ể', ('ê', 4) => 'ễ', ('ê', 5) => 'ệ',
        ('i', 1) => 'ì', ('i', 2) => 'í', ('i', 3) => 'ỉ', ('i', 4) => 'ĩ', ('i', 5) => 'ị',
        ('o', 1) => 'ò', ('o', 2) => 'ó', ('o', 3) => 'ỏ', ('o', 4) => 'õ', ('o', 5) => 'ọ',
        ('ô', 1) => 'ồ', ('ô', 2) => 'ố', ('ô', 3) => 'ổ', ('ô', 4) => 'ỗ', ('ô', 5) => 'ộ',
        ('ơ', 1) => 'ờ', ('ơ', 2) => 'ớ', ('ơ', 3) => 'ở', ('ơ', 4) => 'ỡ', ('ơ', 5) => 'ợ',
        ('u', 1) => 'ù', ('u', 2) => 'ú', ('u', 3) => 'ủ', ('u', 4) => 'ũ', ('u', 5) => 'ụ',
        ('ư', 1) => 'ừ', ('ư', 2) => 'ứ', ('ư', 3) => 'ử', ('ư', 4) => 'ữ', ('ư', 5) => 'ự',
        ('y', 1) => 'ỳ', ('y', 2) => 'ý', ('y', 3) => 'ỷ', ('y', 4) => 'ỹ', ('y', 5) => 'ỵ',
        _ => base,
    }
}

#[inline]
fn is_vietnamese_consonant(c: char) -> bool {
    matches!(
        c,
        'b' | 'c' | 'd' | 'đ' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 'p' | 'q' | 'r' | 's' | 't' | 'v' | 'x'
    )
}

#[inline]
fn is_marked_vowel(c: char) -> bool {
    matches!(c, 'ă' | 'â' | 'ê' | 'ô' | 'ơ' | 'ư')
}

fn normalize_vietnamese_syllable_lower(lower: &str) -> Option<String> {
    if lower.is_empty() || lower.is_ascii() {
        return None;
    }
    let mut stripped: Vec<char> = Vec::with_capacity(8);
    let mut is_vowel_vec: Vec<bool> = Vec::with_capacity(8);
    let mut tone: u8 = 0;
    let mut tone_count: usize = 0;

    for c in lower.chars() {
        if let Some((base, t)) = split_vietnamese_vowel(c) {
            stripped.push(base);
            is_vowel_vec.push(true);
            if t > 0 {
                tone = t;
                tone_count += 1;
            }
        } else if is_vietnamese_consonant(c) {
            stripped.push(c);
            is_vowel_vec.push(false);
        } else {
            return None;
        }
    }

    if tone_count != 1 {
        return None;
    }

    let n = stripped.len();
    let mut v_start: Option<usize> = None;
    let mut v_end: usize = 0;
    let mut in_cluster = false;
    let mut clusters = 0usize;

    for i in 0..n {
        if is_vowel_vec[i] {
            if !in_cluster {
                clusters += 1;
                if clusters > 1 {
                    return None;
                }
                v_start = Some(i);
                in_cluster = true;
            }
            v_end = i + 1;
        } else {
            in_cluster = false;
        }
    }

    let mut start = v_start?;
    if start == 1 && stripped[0] == 'q' && stripped[1] == 'u' && v_end - start >= 2 {
        start = 2;
    } else if start == 1 && stripped[0] == 'g' && stripped[1] == 'i' && v_end - start >= 2 {
        start = 2;
    }

    let v_len = v_end - start;
    if v_len == 0 || v_len > 3 {
        return None;
    }

    let nucleus = &stripped[start..v_end];
    let has_coda = v_end < n;

    let mut last_marked: Option<usize> = None;
    for (rel, &ch) in nucleus.iter().enumerate() {
        if is_marked_vowel(ch) {
            last_marked = Some(rel);
        }
    }

    let target_rel = if let Some(m_idx) = last_marked {
        m_idx
    } else if v_len == 1 {
        0
    } else if v_len == 3 {
        1
    } else if has_coda {
        1
    } else {
        0
    };

    stripped[start + target_rel] = apply_vietnamese_tone(stripped[start + target_rel], tone);
    let candidate: String = stripped.into_iter().collect();
    if candidate == lower {
        None
    } else {
        Some(candidate)
    }
}

fn normalize_vietnamese_word(word: &str) -> Option<(String, String)> {
    let lower = word.to_lowercase();
    normalize_vietnamese_syllable_lower(&lower).map(|norm_lower| {
        let cased = preserve_case(word, &norm_lower);
        (cased, norm_lower)
    })
}

fn normalize_vietnamese_text_inner(text: &str) -> String {
    if text.is_empty() || text.is_ascii() {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut cur = String::new();
    for c in text.chars() {
        if c.is_alphabetic() {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                if let Some((cased, _)) = normalize_vietnamese_word(&cur) {
                    out.push_str(&cased);
                } else {
                    out.push_str(&cur);
                }
                cur.clear();
            }
            out.push(c);
        }
    }
    if !cur.is_empty() {
        if let Some((cased, _)) = normalize_vietnamese_word(&cur) {
            out.push_str(&cased);
        } else {
            out.push_str(&cur);
        }
    }
    out
}

#[pyfunction]
fn fast_normalize_vietnamese_tone(py: Python<'_>, text: &str) -> PyResult<String> {
    if text.is_empty() || text.is_ascii() {
        return Ok(text.to_string());
    }
    let res = py.allow_threads(|| normalize_vietnamese_text_inner(text));
    Ok(res)
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
    weight: f64,
}

#[pyclass]
struct RustViTokenizer {
    nodes: Vec<TrieNode>,
    unigram_weights: HashMap<String, f64>,
}

impl RustViTokenizer {
    fn insert_phrase(&mut self, phrase: &str, custom_weight: Option<f64>) {
        let mut curr = 0usize;
        let mut count = 0usize;
        let mut single_key = String::new();
        for part in phrase.split_whitespace() {
            let key = part.to_lowercase();
            if count == 0 {
                single_key = key.clone();
            }
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
            let default_w = (count as f64) * 2.2 + 0.5;
            self.nodes[curr].weight = custom_weight.unwrap_or_else(|| {
                if self.nodes[curr].weight > 0.0 {
                    self.nodes[curr].weight
                } else {
                    default_w
                }
            });
        } else if count == 1 {
            if let Some(w) = custom_weight {
                self.unigram_weights.insert(single_key, w);
            }
        }
    }

    fn tokenize_raw(&self, text: &str, normalize_tone: bool) -> (Vec<String>, Vec<String>) {
        let mut orig_tokens: Vec<String> = Vec::with_capacity(text.len() / 4 + 4);
        let mut lower_tokens: Vec<String> = Vec::with_capacity(text.len() / 4 + 4);
        let mut cur = String::new();

        let push_word = |buf: &mut String, origs: &mut Vec<String>, lowers: &mut Vec<String>| {
            if buf.is_empty() {
                return;
            }
            if normalize_tone && !buf.is_ascii() {
                let norm = normalize_vietnamese_text_inner(buf);
                lowers.push(norm.to_lowercase());
                origs.push(norm);
                buf.clear();
                return;
            }
            let lower = buf.to_lowercase();
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
        (orig_tokens, lower_tokens)
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
            unigram_weights: HashMap::new(),
        };
        for w in dict_words {
            tok.insert_phrase(&w, None);
        }
        tok
    }

    fn add_words(&mut self, dict_words: Vec<String>) {
        for w in dict_words {
            self.insert_phrase(&w, None);
        }
    }

    fn set_word_weights(&mut self, weights: HashMap<String, f64>) {
        for (phrase, w) in weights {
            self.insert_phrase(&phrase, Some(w));
        }
    }

    fn segment(&self, py: Python<'_>, text: &str, join_compound: bool, sep: &str) -> PyResult<Vec<String>> {
        self.normalize_and_segment(py, text, join_compound, sep, false)
    }

    fn normalize_and_segment(
        &self,
        py: Python<'_>,
        text: &str,
        join_compound: bool,
        sep: &str,
        normalize_tone: bool,
    ) -> PyResult<Vec<String>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let out = py.allow_threads(|| {
            let (orig_tokens, lower_tokens) = self.tokenize_raw(text, normalize_tone);
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
            out
        });
        Ok(out)
    }

    fn segment_viterbi(
        &self,
        py: Python<'_>,
        text: &str,
        join_compound: bool,
        sep: &str,
        normalize_tone: bool,
    ) -> PyResult<Vec<String>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let out = py.allow_threads(|| {
            let (orig_tokens, lower_tokens) = self.tokenize_raw(text, normalize_tone);
            let n = orig_tokens.len();
            if n == 0 {
                return Vec::new();
            }
            let mut dp = vec![f64::NEG_INFINITY; n + 1];
            let mut back = vec![0usize; n + 1];
            dp[0] = 0.0;

            for i in 0..n {
                if !dp[i].is_finite() {
                    continue;
                }
                let uni_w = self.unigram_weights.get(&lower_tokens[i]).copied().unwrap_or(0.0);
                let cand_uni = dp[i] + uni_w;
                if cand_uni > dp[i + 1] {
                    dp[i + 1] = cand_uni;
                    back[i + 1] = i;
                }

                let mut curr = 0usize;
                let mut j = i;
                while j < n {
                    if let Some(&next_idx) = self.nodes[curr].children.get(&lower_tokens[j]) {
                        curr = next_idx;
                        j += 1;
                        if j >= i + 2 && self.nodes[curr].is_end {
                            let cand = dp[i] + self.nodes[curr].weight;
                            if cand > dp[j] {
                                dp[j] = cand;
                                back[j] = i;
                            }
                        }
                    } else {
                        break;
                    }
                }
            }

            let mut spans: Vec<(usize, usize)> = Vec::new();
            let mut pos = n;
            while pos > 0 {
                let prev = back[pos];
                spans.push((prev, pos));
                pos = prev;
            }
            spans.reverse();

            let joiner = if join_compound { sep } else { " " };
            let mut out: Vec<String> = Vec::with_capacity(spans.len());
            for (s, e) in spans {
                if e > s + 1 {
                    let mut merged = String::new();
                    for k in s..e {
                        if k > s {
                            merged.push_str(joiner);
                        }
                        merged.push_str(&orig_tokens[k]);
                    }
                    out.push(merged);
                } else {
                    out.push(orig_tokens[s].clone());
                }
            }
            out
        });
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
) -> Result<Vec<f32>, &'static str> {
    if k == 0 || k % 32 != 0 {
        return Err("Chiều k phải là bội số dương của 32 cho Q4_0");
    }
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row
        .checked_mul(18)
        .ok_or("Tràn số kích thước hàng Q4_0")?;
    let x_req = rows
        .checked_mul(k)
        .ok_or("Tràn số kích thước ma trận đầu vào Q4_0")?;
    let w_req = out_cols
        .checked_mul(bytes_per_row)
        .ok_or("Tràn số kích thước trọng số Q4_0")?;
    let out_len = rows
        .checked_mul(out_cols)
        .ok_or("Tràn số kích thước đầu ra Q4_0")?;
    if x.len() < x_req || raw_w.len() < w_req {
        return Err("Kích thước bộ đệm đầu vào hoặc trọng số Q4_0 không đủ");
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
                        blk_sum += x_row[base + j] * q_lo + x_row[base + j + 16] * q_hi;
                    }
                    acc += blk_sum * scale;
                }
                vals[r] = acc;
            }
            vals
        })
        .collect();

    let mut out = vec![0.0f32; out_len];
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
) -> Result<Vec<f32>, &'static str> {
    if k == 0 || k % 32 != 0 {
        return Err("Chiều k phải là bội số dương của 32 cho Q8_0");
    }
    let blocks_per_row = k / 32;
    let bytes_per_row = blocks_per_row
        .checked_mul(34)
        .ok_or("Tràn số kích thước hàng Q8_0")?;
    let x_req = rows
        .checked_mul(k)
        .ok_or("Tràn số kích thước ma trận đầu vào Q8_0")?;
    let w_req = out_cols
        .checked_mul(bytes_per_row)
        .ok_or("Tràn số kích thước trọng số Q8_0")?;
    let out_len = rows
        .checked_mul(out_cols)
        .ok_or("Tràn số kích thước đầu ra Q8_0")?;
    if x.len() < x_req || raw_w.len() < w_req {
        return Err("Kích thước bộ đệm đầu vào hoặc trọng số Q8_0 không đủ");
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

    let mut out = vec![0.0f32; out_len];
    for (col, vals) in col_results.into_iter().enumerate() {
        for r in 0..rows {
            out[r * out_cols + col] = vals[r];
        }
    }
    Ok(out)
}

#[pyfunction]
fn fast_q4_0_matmul(
    py: Python<'_>,
    x: Vec<f32>,
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| q4_0_matmul_slice(&x, rows, k, raw_w, out_cols))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn fast_q8_0_matmul(
    py: Python<'_>,
    x: Vec<f32>,
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    py.allow_threads(|| q8_0_matmul_slice(&x, rows, k, raw_w, out_cols))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn fast_q4_0_matmul_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    let n_x = rows
        .checked_mul(k)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Q4_0"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào Q4_0"))?;
    if x_f32_le.len() < x_bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 đầu vào Q4_0 không đủ kích thước",
        ));
    }
    py.allow_threads(|| {
        let x: Vec<f32> = (0..n_x).map(|i| read_f32_le(x_f32_le, i)).collect();
        q4_0_matmul_slice(&x, rows, k, raw_w, out_cols)
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn fast_q8_0_matmul_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    let n_x = rows
        .checked_mul(k)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Q8_0"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào Q8_0"))?;
    if x_f32_le.len() < x_bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 đầu vào Q8_0 không đủ kích thước",
        ));
    }
    py.allow_threads(|| {
        let x: Vec<f32> = (0..n_x).map(|i| read_f32_le(x_f32_le, i)).collect();
        q8_0_matmul_slice(&x, rows, k, raw_w, out_cols)
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
fn fast_q4_k_matmul_bytes(
    py: Python<'_>,
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
    let bytes_per_row = blocks_per_row
        .checked_mul(144)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước hàng Q4_K"))?;
    let n_x = rows
        .checked_mul(k)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Q4_K"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào Q4_K"))?;
    let w_req = out_cols
        .checked_mul(bytes_per_row)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước trọng số Q4_K"))?;
    let out_len = rows
        .checked_mul(out_cols)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước đầu ra Q4_K"))?;
    if x_f32_le.len() < x_bytes_req || raw_w.len() < w_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q4_K không đủ",
        ));
    }
    let out = py.allow_threads(|| {
        let x: Vec<f32> = (0..n_x).map(|i| read_f32_le(x_f32_le, i)).collect();

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

        let mut res = vec![0.0f32; out_len];
        for (col, vals) in col_results.into_iter().enumerate() {
            for r in 0..rows {
                res[r * out_cols + col] = vals[r];
            }
        }
        res
    });
    Ok(out)
}

#[pyfunction]
fn fast_q5_k_matmul_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if k == 0 || k % 256 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Chiều k phải là bội số dương của 256 cho Q5_K",
        ));
    }
    let blocks_per_row = k / 256;
    let bytes_per_row = blocks_per_row
        .checked_mul(176)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước hàng Q5_K"))?;
    let n_x = rows
        .checked_mul(k)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Q5_K"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào Q5_K"))?;
    let w_req = out_cols
        .checked_mul(bytes_per_row)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước trọng số Q5_K"))?;
    let out_len = rows
        .checked_mul(out_cols)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước đầu ra Q5_K"))?;
    if x_f32_le.len() < x_bytes_req || raw_w.len() < w_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q5_K không đủ",
        ));
    }
    let out = py.allow_threads(|| {
        let x: Vec<f32> = (0..n_x).map(|i| read_f32_le(x_f32_le, i)).collect();

        let col_results: Vec<Vec<f32>> = (0..out_cols)
            .into_par_iter()
            .map(|col| {
                let row_bytes = &raw_w[col * bytes_per_row..(col + 1) * bytes_per_row];
                let mut vals = vec![0.0f32; rows];
                for r in 0..rows {
                    let x_row = &x[r * k..(r + 1) * k];
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &row_bytes[b * 176..(b + 1) * 176];
                        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                        let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
                        let sc_raw = &blk[4..16];
                        let qh = &blk[16..48];
                        let qs = &blk[48..176];

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
                            let u1 = 1u8 << (2 * g);
                            let u2 = 1u8 << (2 * g + 1);
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
                                let hbyte = qh[j];
                                let xl = x_row[pos_lo + j];
                                let xh = x_row[pos_hi + j];
                                let q_lo = (byte & 0x0f) as f32 + if (hbyte & u1) != 0 { 16.0 } else { 0.0 };
                                let q_hi = ((byte >> 4) & 0x0f) as f32 + if (hbyte & u2) != 0 { 16.0 } else { 0.0 };
                                sum_lo += xl * q_lo;
                                xsum_lo += xl;
                                sum_hi += xh * q_hi;
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

        let mut res = vec![0.0f32; out_len];
        for (col, vals) in col_results.into_iter().enumerate() {
            for r in 0..rows {
                res[r * out_cols + col] = vals[r];
            }
        }
        res
    });
    Ok(out)
}

#[pyfunction]
fn fast_q6_k_matmul_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Vec<f32>> {
    if k == 0 || k % 256 != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Chiều k phải là bội số dương của 256 cho Q6_K",
        ));
    }
    let blocks_per_row = k / 256;
    let bytes_per_row = blocks_per_row
        .checked_mul(210)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước hàng Q6_K"))?;
    let n_x = rows
        .checked_mul(k)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Q6_K"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào Q6_K"))?;
    let w_req = out_cols
        .checked_mul(bytes_per_row)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước trọng số Q6_K"))?;
    let out_len = rows
        .checked_mul(out_cols)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước đầu ra Q6_K"))?;
    if x_f32_le.len() < x_bytes_req || raw_w.len() < w_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm đầu vào hoặc trọng số Q6_K không đủ",
        ));
    }
    let out = py.allow_threads(|| {
        let x: Vec<f32> = (0..n_x).map(|i| read_f32_le(x_f32_le, i)).collect();

        let col_results: Vec<Vec<f32>> = (0..out_cols)
            .into_par_iter()
            .map(|col| {
                let row_bytes = &raw_w[col * bytes_per_row..(col + 1) * bytes_per_row];
                let mut vals = vec![0.0f32; rows];
                for r in 0..rows {
                    let x_row = &x[r * k..(r + 1) * k];
                    let mut acc = 0.0f32;
                    for b in 0..blocks_per_row {
                        let blk = &row_bytes[b * 210..(b + 1) * 210];
                        let ql = &blk[0..128];
                        let qh = &blk[128..192];
                        let sc = &blk[192..208];
                        let d = f16_bits_to_f32(u16::from_le_bytes([blk[208], blk[209]]));

                        for h in 0..2 {
                            let ql_h = &ql[h * 64..(h + 1) * 64];
                            let qh_h = &qh[h * 32..(h + 1) * 32];
                            let sc_h = &sc[h * 8..(h + 1) * 8];
                            let base = b * 256 + h * 128;
                            for sub in 0..2 {
                                let s0 = d * (sc_h[sub] as i8) as f32;
                                let s2 = d * (sc_h[sub + 2] as i8) as f32;
                                let s4 = d * (sc_h[sub + 4] as i8) as f32;
                                let s6 = d * (sc_h[sub + 6] as i8) as f32;
                                let mut acc0 = 0.0f32;
                                let mut acc2 = 0.0f32;
                                let mut acc4 = 0.0f32;
                                let mut acc6 = 0.0f32;
                                for j in 0..16 {
                                    let idx = sub * 16 + j;
                                    let ql0 = ql_h[idx];
                                    let ql1 = ql_h[32 + idx];
                                    let qhb = qh_h[idx];
                                    let q1 = (((ql0 & 0x0f) | ((qhb & 0x03) << 4)) as i32 - 32) as f32;
                                    let q2 = (((ql1 & 0x0f) | (((qhb >> 2) & 0x03) << 4)) as i32 - 32) as f32;
                                    let q3 = ((((ql0 >> 4) & 0x0f) | (((qhb >> 4) & 0x03) << 4)) as i32 - 32) as f32;
                                    let q4 = ((((ql1 >> 4) & 0x0f) | (((qhb >> 6) & 0x03) << 4)) as i32 - 32) as f32;
                                    acc0 += x_row[base + idx] * q1;
                                    acc2 += x_row[base + 32 + idx] * q2;
                                    acc4 += x_row[base + 64 + idx] * q3;
                                    acc6 += x_row[base + 96 + idx] * q4;
                                }
                                acc += acc0 * s0 + acc2 * s2 + acc4 * s4 + acc6 * s6;
                            }
                        }
                    }
                    vals[r] = acc;
                }
                vals
            })
            .collect();

        let mut res = vec![0.0f32; out_len];
        for (col, vals) in col_results.into_iter().enumerate() {
            for r in 0..rows {
                res[r * out_cols + col] = vals[r];
            }
        }
        res
    });
    Ok(out)
}

#[pyfunction]
fn fast_rmsnorm_f32(
    py: Python<'_>,
    x_f32_le: &[u8],
    gamma_f32_le: &[u8],
    rows: usize,
    dim: usize,
    eps: f32,
) -> PyResult<Vec<f32>> {
    if rows == 0 || dim == 0 {
        return Ok(Vec::new());
    }
    let n_x = rows
        .checked_mul(dim)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước RMSNorm"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào RMSNorm"))?;
    let g_bytes_req = dim
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte trọng số gamma RMSNorm"))?;
    if x_f32_le.len() < x_bytes_req || gamma_f32_le.len() < g_bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 cho RMSNorm không đủ kích thước",
        ));
    }
    let out = py.allow_threads(|| {
        let gamma: Vec<f32> = (0..dim).map(|i| read_f32_le(gamma_f32_le, i)).collect();
        let row_vecs: Vec<Vec<f32>> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let base = r * dim;
                let mut sum_sq = 0.0f32;
                let mut row_vals = Vec::with_capacity(dim);
                for j in 0..dim {
                    let v = read_f32_le(x_f32_le, base + j);
                    sum_sq += v * v;
                    row_vals.push(v);
                }
                let inv_rms = 1.0f32 / (sum_sq / (dim as f32) + eps).sqrt();
                for j in 0..dim {
                    row_vals[j] = row_vals[j] * inv_rms * gamma[j];
                }
                row_vals
            })
            .collect();
        let mut flat = Vec::with_capacity(n_x);
        for rv in row_vecs {
            flat.extend_from_slice(&rv);
        }
        flat
    });
    Ok(out)
}

#[pyfunction]
fn fast_swiglu_f32(
    py: Python<'_>,
    gate_f32_le: &[u8],
    up_f32_le: &[u8],
    n: usize,
) -> PyResult<Vec<f32>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let bytes_req = n
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước SwiGLU"))?;
    if gate_f32_le.len() < bytes_req || up_f32_le.len() < bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 cho SwiGLU không đủ kích thước",
        ));
    }
    let out = py.allow_threads(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let g = read_f32_le(gate_f32_le, i);
                let u = read_f32_le(up_f32_le, i);
                let g_clamped = g.clamp(-20.0, 20.0);
                let silu = g / (1.0f32 + (-g_clamped).exp());
                silu * u
            })
            .collect()
    });
    Ok(out)
}

fn f32_slice_to_pybytes(py: Python<'_>, vals: &[f32]) -> Py<PyBytes> {
    let mut raw = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    PyBytes::new_bound(py, &raw).unbind()
}

#[pyfunction]
fn fast_q5_k_matmul_f32_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Py<PyBytes>> {
    let out = fast_q5_k_matmul_bytes(py, x_f32_le, rows, k, raw_w, out_cols)?;
    Ok(f32_slice_to_pybytes(py, &out))
}

#[pyfunction]
fn fast_q6_k_matmul_f32_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    rows: usize,
    k: usize,
    raw_w: &[u8],
    out_cols: usize,
) -> PyResult<Py<PyBytes>> {
    let out = fast_q6_k_matmul_bytes(py, x_f32_le, rows, k, raw_w, out_cols)?;
    Ok(f32_slice_to_pybytes(py, &out))
}

#[pyfunction]
fn fast_rmsnorm_f32_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    gamma_f32_le: &[u8],
    rows: usize,
    dim: usize,
    eps: f32,
) -> PyResult<Py<PyBytes>> {
    if rows == 0 || dim == 0 {
        return Ok(PyBytes::new_bound(py, &[]).unbind());
    }
    let n_x = rows
        .checked_mul(dim)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước RMSNorm"))?;
    let x_bytes_req = n_x
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte đầu vào RMSNorm"))?;
    let g_bytes_req = dim
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số byte trọng số gamma RMSNorm"))?;
    if x_f32_le.len() < x_bytes_req || gamma_f32_le.len() < g_bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 cho RMSNorm không đủ kích thước",
        ));
    }
    let raw = py.allow_threads(|| {
        let gamma: Vec<f32> = (0..dim).map(|i| read_f32_le(gamma_f32_le, i)).collect();
        let mut out_bytes = vec![0u8; x_bytes_req];
        let process_row = |r: usize, dst: &mut [u8]| {
            let base = r * dim;
            let mut sum_sq = 0.0f32;
            for j in 0..dim {
                let v = read_f32_le(x_f32_le, base + j);
                sum_sq += v * v;
            }
            let inv_rms = 1.0f32 / (sum_sq / (dim as f32) + eps).sqrt();
            for j in 0..dim {
                let v = read_f32_le(x_f32_le, base + j) * inv_rms * gamma[j];
                dst[j * 4..(j + 1) * 4].copy_from_slice(&v.to_le_bytes());
            }
        };
        if n_x >= 16384 && rows >= 8 {
            out_bytes
                .par_chunks_exact_mut(dim * 4)
                .enumerate()
                .for_each(|(r, dst)| process_row(r, dst));
        } else {
            for (r, dst) in out_bytes.chunks_exact_mut(dim * 4).enumerate() {
                process_row(r, dst);
            }
        }
        out_bytes
    });
    Ok(PyBytes::new_bound(py, &raw).unbind())
}

#[pyfunction]
fn fast_swiglu_f32_bytes(
    py: Python<'_>,
    gate_f32_le: &[u8],
    up_f32_le: &[u8],
    n: usize,
) -> PyResult<Py<PyBytes>> {
    if n == 0 {
        return Ok(PyBytes::new_bound(py, &[]).unbind());
    }
    let bytes_req = n
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước SwiGLU"))?;
    if gate_f32_le.len() < bytes_req || up_f32_le.len() < bytes_req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 cho SwiGLU không đủ kích thước",
        ));
    }
    let raw = py.allow_threads(|| {
        let mut out_bytes = vec![0u8; bytes_req];
        if n >= 16384 {
            out_bytes
                .par_chunks_exact_mut(4)
                .enumerate()
                .for_each(|(i, dst)| {
                    let g = read_f32_le(gate_f32_le, i);
                    let u = read_f32_le(up_f32_le, i);
                    let g_clamped = g.clamp(-20.0, 20.0);
                    let val = (g / (1.0f32 + (-g_clamped).exp())) * u;
                    dst.copy_from_slice(&val.to_le_bytes());
                });
        } else {
            for (i, dst) in out_bytes.chunks_exact_mut(4).enumerate() {
                let g = read_f32_le(gate_f32_le, i);
                let u = read_f32_le(up_f32_le, i);
                let g_clamped = g.clamp(-20.0, 20.0);
                let val = (g / (1.0f32 + (-g_clamped).exp())) * u;
                dst.copy_from_slice(&val.to_le_bytes());
            }
        }
        out_bytes
    });
    Ok(PyBytes::new_bound(py, &raw).unbind())
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
    let mut borrow = 0u64;
    for i in 0..LIMBS {
        let (d1, b1) = a[i].overflowing_sub(p[i]);
        let (d2, b2) = d1.overflowing_sub(borrow);
        diff[i] = d2;
        borrow = (b1 as u64) | (b2 as u64);
    }
    let (_, underflow) = extra.overflowing_sub(borrow);
    let mask = ((underflow as u64) ^ 1u64).wrapping_neg();
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
fn fast_rfc5114_modpow(py: Python<'_>, base_be: &[u8], exp_be: &[u8]) -> PyResult<Vec<u8>> {
    if base_be.len() > 256 || exp_be.len() > 32 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Đầu vào RFC 5114 phải có độ dài tối đa 256 bytes (base) và 32 bytes (exp)",
        ));
    }
    let out = py.allow_threads(|| {
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
        limbs_to_be_bytes(&result)
    });
    Ok(out)
}

#[pymodule]
fn vietnamese_ai_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_token_count, m)?)?;
    m.add_function(wrap_pyfunction!(fast_keyword_search, m)?)?;
    m.add_function(wrap_pyfunction!(fast_batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(fast_top_k_cosine_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_clean_spaces, m)?)?;
    m.add_function(wrap_pyfunction!(fast_normalize_vietnamese_tone, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_0_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q8_0_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_0_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q8_0_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q4_k_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q5_k_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q6_k_matmul_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q5_k_matmul_f32_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_q6_k_matmul_f32_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_rmsnorm_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fast_swiglu_f32, m)?)?;
    m.add_function(wrap_pyfunction!(fast_rmsnorm_f32_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_swiglu_f32_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(fast_rfc5114_modpow, m)?)?;
    m.add_class::<RustViTokenizer>()?;
    Ok(())
}
