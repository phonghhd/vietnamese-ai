use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[inline]
fn read_f32_le(bytes: &[u8], idx: usize) -> f32 {
    let o = idx * 4;
    f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: &mut usize) -> PyResult<u32> {
    if *offset + 4 > bytes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dữ liệu nhị phân HNSW bị cắt cụt (u32)",
        ));
    }
    let v = u32::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
    ]);
    *offset += 4;
    Ok(v)
}

#[inline]
fn read_i64_le(bytes: &[u8], offset: &mut usize) -> PyResult<i64> {
    if *offset + 8 > bytes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Dữ liệu nhị phân HNSW bị cắt cụt (i64)",
        ));
    }
    let v = i64::from_le_bytes([
        bytes[*offset],
        bytes[*offset + 1],
        bytes[*offset + 2],
        bytes[*offset + 3],
        bytes[*offset + 4],
        bytes[*offset + 5],
        bytes[*offset + 6],
        bytes[*offset + 7],
    ]);
    *offset += 8;
    Ok(v)
}

#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut ca = a[..n].chunks_exact(4);
    let mut cb = b[..n].chunks_exact(4);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        s0 += xa[0] * xb[0];
        s1 += xa[1] * xb[1];
        s2 += xa[2] * xb[2];
        s3 += xa[3] * xb[3];
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for (&xa, &xb) in ca.remainder().iter().zip(cb.remainder().iter()) {
        s += xa * xb;
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    while i + 8 <= n {
        let va = _mm256_loadu_ps(ap.add(i));
        let vb = _mm256_loadu_ps(bp.add(i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + (lanes[4] + lanes[5])
        + (lanes[6] + lanes[7]);
    while i < n {
        s += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    s
}

#[inline]
fn dot_f32_unrolled(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    dot_f32_scalar(a, b)
}

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let mut s0 = 0i32;
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut s3 = 0i32;
    let mut ca = a[..n].chunks_exact(8);
    let mut cb = b[..n].chunks_exact(8);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        s0 += (xa[0] as i16 * xb[0] as i16) as i32 + (xa[1] as i16 * xb[1] as i16) as i32;
        s1 += (xa[2] as i16 * xb[2] as i16) as i32 + (xa[3] as i16 * xb[3] as i16) as i32;
        s2 += (xa[4] as i16 * xb[4] as i16) as i32 + (xa[5] as i16 * xb[5] as i16) as i32;
        s3 += (xa[6] as i16 * xb[6] as i16) as i32 + (xa[7] as i16 * xb[7] as i16) as i32;
    }
    let mut s = (s0 + s1) + (s2 + s3);
    for (&xa, &xb) in ca.remainder().iter().zip(cb.remainder().iter()) {
        s += (xa as i32) * (xb as i32);
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_i8_sse41(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let mut acc = _mm_setzero_si128();
    let mut i = 0usize;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    while i + 16 <= n {
        let va = _mm_loadu_si128(ap.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(bp.add(i) as *const __m128i);
        let va_lo = _mm_cvtepi8_epi16(va);
        let vb_lo = _mm_cvtepi8_epi16(vb);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(va_lo, vb_lo));
        let va_hi = _mm_cvtepi8_epi16(_mm_srli_si128::<8>(va));
        let vb_hi = _mm_cvtepi8_epi16(_mm_srli_si128::<8>(vb));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(va_hi, vb_hi));
        i += 16;
    }
    let mut lanes = [0i32; 4];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, acc);
    let mut s = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        s += (*ap.add(i) as i32) * (*bp.add(i) as i32);
        i += 1;
    }
    s
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let mut acc = _mm256_setzero_si256();
    let mut i = 0usize;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    while i + 16 <= n {
        let va = _mm256_cvtepi8_epi16(_mm_loadu_si128(ap.add(i) as *const __m128i));
        let vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(bp.add(i) as *const __m128i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
        i += 16;
    }
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut s = (lanes[0] + lanes[1])
        + (lanes[2] + lanes[3])
        + (lanes[4] + lanes[5])
        + (lanes[6] + lanes[7]);
    while i < n {
        s += (*ap.add(i) as i32) * (*bp.add(i) as i32);
        i += 1;
    }
    s
}

#[inline]
fn dot_i8_unrolled(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            return unsafe { dot_i8_sse41(a, b) };
        }
    }
    dot_i8_scalar(a, b)
}

#[inline]
fn dot_i8_u8_slice(a: &[i8], b_raw: &[u8]) -> i32 {
    let b_i8 = unsafe { std::slice::from_raw_parts(b_raw.as_ptr() as *const i8, b_raw.len()) };
    dot_i8_unrolled(a, b_i8)
}

#[inline]
fn normalize_f32_from_bytes(bytes: &[u8], base_elem: usize, dim: usize, out: &mut [f32]) {
    let mut norm_sq = 0.0f32;
    for j in 0..dim {
        let v = read_f32_le(bytes, base_elem + j);
        out[j] = v;
        norm_sq += v * v;
    }
    let norm = norm_sq.sqrt();
    if norm > 1e-10 {
        let inv = 1.0 / norm;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
}

#[inline]
fn quantize_unit_vec_sq8(unit: &[f32], out_codes: &mut [i8]) -> f32 {
    let mut absmax = 0.0f32;
    for &v in unit {
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }
    if absmax < 1e-10 {
        for c in out_codes.iter_mut() {
            *c = 0;
        }
        return 1.0 / 127.0;
    }
    let scale = absmax / 127.0;
    let inv_scale = 127.0 / absmax;
    for (dst, &v) in out_codes.iter_mut().zip(unit.iter()) {
        *dst = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

#[inline]
fn pack_binary_words(vec: &[f32], out_words: &mut [u64]) {
    for w in out_words.iter_mut() {
        *w = 0;
    }
    for (j, &v) in vec.iter().enumerate() {
        if v >= 0.0 {
            out_words[j >> 6] |= 1u64 << (j & 63);
        }
    }
}

#[pyfunction]
pub fn fast_sq8_quantize_bytes(
    py: Python<'_>,
    matrix_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
) -> PyResult<(Py<PyBytes>, Py<PyBytes>)> {
    if n_docs == 0 || dim == 0 {
        return Ok((PyBytes::new_bound(py, &[]).unbind(), PyBytes::new_bound(py, &[]).unbind()));
    }
    let req = n_docs
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận SQ8"))?;
    if matrix_f32_le.len() < req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 không đủ kích thước cho fast_sq8_quantize_bytes",
        ));
    }

    let (codes_raw, scales_raw) = py.allow_threads(|| {
        let mut codes_all = vec![0u8; n_docs * dim];
        let mut scales_all = vec![0.0f32; n_docs];

        codes_all
            .par_chunks_mut(dim)
            .zip(scales_all.par_iter_mut())
            .enumerate()
            .for_each(|(doc_idx, (dst_u8, dst_scale))| {
                let mut unit = vec![0.0f32; dim];
                normalize_f32_from_bytes(matrix_f32_le, doc_idx * dim, dim, &mut unit);
                let mut q_i8 = vec![0i8; dim];
                let scale = quantize_unit_vec_sq8(&unit, &mut q_i8);
                for j in 0..dim {
                    dst_u8[j] = q_i8[j] as u8;
                }
                *dst_scale = scale;
            });

        let mut scales_bytes = Vec::with_capacity(n_docs * 4);
        for s in scales_all {
            scales_bytes.extend_from_slice(&s.to_le_bytes());
        }
        (codes_all, scales_bytes)
    });

    Ok((
        PyBytes::new_bound(py, &codes_raw).unbind(),
        PyBytes::new_bound(py, &scales_raw).unbind(),
    ))
}

#[pyfunction]
pub fn fast_sq8_top_k_bytes(
    py: Python<'_>,
    query_f32_le: &[u8],
    codes_i8: &[u8],
    scales_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if n_docs == 0 || dim == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    if query_f32_le.len() < dim * 4 || codes_i8.len() < n_docs * dim || scales_f32_le.len() < n_docs * 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm không hợp lệ cho fast_sq8_top_k_bytes",
        ));
    }

    let scored = py.allow_threads(|| {
        let mut q_unit = vec![0.0f32; dim];
        normalize_f32_from_bytes(query_f32_le, 0, dim, &mut q_unit);
        let mut q_i8 = vec![0i8; dim];
        let q_scale = quantize_unit_vec_sq8(&q_unit, &mut q_i8);

        let mut res: Vec<(usize, f32)> = (0..n_docs)
            .into_par_iter()
            .map(|doc_idx| {
                let d_slice = &codes_i8[doc_idx * dim..(doc_idx + 1) * dim];
                let d_scale = read_f32_le(scales_f32_le, doc_idx);
                let dot = dot_i8_u8_slice(&q_i8, d_slice);
                (doc_idx, (dot as f32) * (q_scale * d_scale))
            })
            .collect();

        let k = top_k.min(res.len());
        res.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        res.truncate(k);
        res.sort_by(|a, b| b.1.total_cmp(&a.1));
        res
    });

    Ok(scored)
}

#[pyfunction]
pub fn fast_binary_quantize_bytes(
    py: Python<'_>,
    matrix_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
) -> PyResult<Py<PyBytes>> {
    if n_docs == 0 || dim == 0 {
        return Ok(PyBytes::new_bound(py, &[]).unbind());
    }
    let req = n_docs
        .checked_mul(dim)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận Binary"))?;
    if matrix_f32_le.len() < req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm float32 không đủ kích thước cho fast_binary_quantize_bytes",
        ));
    }

    let words_per_vec = (dim + 63) / 64;
    let bytes_per_vec = words_per_vec * 8;

    let out_bytes = py.allow_threads(|| {
        let mut raw = vec![0u8; n_docs * bytes_per_vec];
        raw.par_chunks_mut(bytes_per_vec)
            .enumerate()
            .for_each(|(doc_idx, dst_bytes)| {
                let base = doc_idx * dim;
                let mut words = vec![0u64; words_per_vec];
                for j in 0..dim {
                    if read_f32_le(matrix_f32_le, base + j) >= 0.0 {
                        words[j >> 6] |= 1u64 << (j & 63);
                    }
                }
                for (w_idx, w) in words.iter().enumerate() {
                    let le = w.to_le_bytes();
                    dst_bytes[w_idx * 8..(w_idx + 1) * 8].copy_from_slice(&le);
                }
            });
        raw
    });

    Ok(PyBytes::new_bound(py, &out_bytes).unbind())
}

#[inline]
fn hamming_doc_u64(q_words: &[u64], doc_bytes: &[u8]) -> u32 {
    let mut dist = 0u32;
    for (w_idx, &qw) in q_words.iter().enumerate() {
        let o = w_idx * 8;
        let dw = u64::from_le_bytes([
            doc_bytes[o],
            doc_bytes[o + 1],
            doc_bytes[o + 2],
            doc_bytes[o + 3],
            doc_bytes[o + 4],
            doc_bytes[o + 5],
            doc_bytes[o + 6],
            doc_bytes[o + 7],
        ]);
        dist += (qw ^ dw).count_ones();
    }
    dist
}

#[pyfunction]
pub fn fast_binary_hamming_top_k_bytes(
    py: Python<'_>,
    query_f32_le: &[u8],
    binary_u64_le: &[u8],
    n_docs: usize,
    dim: usize,
    top_k: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if n_docs == 0 || dim == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let words_per_vec = (dim + 63) / 64;
    let bytes_per_vec = words_per_vec * 8;
    if query_f32_le.len() < dim * 4 || binary_u64_le.len() < n_docs * bytes_per_vec {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm không hợp lệ cho fast_binary_hamming_top_k_bytes",
        ));
    }

    let scored = py.allow_threads(|| {
        let q_vec: Vec<f32> = (0..dim).map(|j| read_f32_le(query_f32_le, j)).collect();
        let mut q_words = vec![0u64; words_per_vec];
        pack_binary_words(&q_vec, &mut q_words);
        let inv_dim = 2.0f32 / (dim as f32);

        let mut res: Vec<(usize, u32)> = (0..n_docs)
            .into_par_iter()
            .map(|doc_idx| {
                let d_bytes = &binary_u64_le[doc_idx * bytes_per_vec..(doc_idx + 1) * bytes_per_vec];
                (doc_idx, hamming_doc_u64(&q_words, d_bytes))
            })
            .collect();

        let k = top_k.min(res.len());
        res.select_nth_unstable_by(k - 1, |a, b| a.1.cmp(&b.1));
        res.truncate(k);
        res.sort_by(|a, b| a.1.cmp(&b.1));
        res.into_iter()
            .map(|(idx, h)| (idx, 1.0 - (h as f32) * inv_dim))
            .collect()
    });

    Ok(scored)
}

#[pyfunction]
pub fn fast_binary_rescore_sq8_top_k_bytes(
    py: Python<'_>,
    query_f32_le: &[u8],
    binary_u64_le: &[u8],
    codes_i8: &[u8],
    scales_f32_le: &[u8],
    n_docs: usize,
    dim: usize,
    top_k: usize,
    oversample: usize,
) -> PyResult<Vec<(usize, f32)>> {
    if n_docs == 0 || dim == 0 || top_k == 0 {
        return Ok(Vec::new());
    }
    let words_per_vec = (dim + 63) / 64;
    let bytes_per_vec = words_per_vec * 8;
    if query_f32_le.len() < dim * 4
        || binary_u64_le.len() < n_docs * bytes_per_vec
        || codes_i8.len() < n_docs * dim
        || scales_f32_le.len() < n_docs * 4
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Kích thước bộ đệm không hợp lệ cho fast_binary_rescore_sq8_top_k_bytes",
        ));
    }

    let scored = py.allow_threads(|| {
        let mut q_unit = vec![0.0f32; dim];
        normalize_f32_from_bytes(query_f32_le, 0, dim, &mut q_unit);
        let mut q_words = vec![0u64; words_per_vec];
        pack_binary_words(&q_unit, &mut q_words);
        let mut q_i8 = vec![0i8; dim];
        let q_scale = quantize_unit_vec_sq8(&q_unit, &mut q_i8);

        let mut stage1: Vec<(usize, u32)> = (0..n_docs)
            .into_par_iter()
            .map(|doc_idx| {
                let d_bytes = &binary_u64_le[doc_idx * bytes_per_vec..(doc_idx + 1) * bytes_per_vec];
                (doc_idx, hamming_doc_u64(&q_words, d_bytes))
            })
            .collect();

        let cand_k = (top_k.saturating_mul(oversample.max(2))).max(64).min(n_docs);
        stage1.select_nth_unstable_by(cand_k - 1, |a, b| a.1.cmp(&b.1));
        stage1.truncate(cand_k);

        let mut stage2: Vec<(usize, f32)> = stage1
            .into_iter()
            .map(|(doc_idx, _)| {
                let d_slice = &codes_i8[doc_idx * dim..(doc_idx + 1) * dim];
                let d_scale = read_f32_le(scales_f32_le, doc_idx);
                let dot = dot_i8_u8_slice(&q_i8, d_slice);
                (doc_idx, (dot as f32) * (q_scale * d_scale))
            })
            .collect();

        let k = top_k.min(stage2.len());
        stage2.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        stage2.truncate(k);
        stage2.sort_by(|a, b| b.1.total_cmp(&a.1));
        stage2
    });

    Ok(scored)
}

#[derive(Copy, Clone)]
struct MaxDistNode {
    dist: f32,
    idx: u32,
}

impl PartialEq for MaxDistNode {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.dist.to_bits() == other.dist.to_bits()
    }
}
impl Eq for MaxDistNode {}
impl PartialOrd for MaxDistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxDistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

#[derive(Copy, Clone)]
struct MinDistNode {
    dist: f32,
    idx: u32,
}

impl PartialEq for MinDistNode {
    fn eq(&self, other: &Self) -> bool {
        self.idx == other.idx && self.dist.to_bits() == other.dist.to_bits()
    }
}
impl Eq for MinDistNode {}
impl PartialOrd for MinDistNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinDistNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.total_cmp(&self.dist)
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        let val = (x >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        val.max(1e-12)
    }
}

enum HnswStorage {
    Fp32(Vec<f32>),
    Sq8 { codes: Vec<i8>, scales: Vec<f32> },
}

impl HnswStorage {
    #[inline]
    fn distance_nodes(&self, dim: usize, a: usize, b: usize) -> f32 {
        match self {
            HnswStorage::Fp32(vecs) => {
                let va = &vecs[a * dim..(a + 1) * dim];
                let vb = &vecs[b * dim..(b + 1) * dim];
                1.0 - dot_f32_unrolled(va, vb)
            }
            HnswStorage::Sq8 { codes, scales } => {
                let ca = &codes[a * dim..(a + 1) * dim];
                let cb = &codes[b * dim..(b + 1) * dim];
                let sim = (dot_i8_unrolled(ca, cb) as f32) * (scales[a] * scales[b]);
                1.0 - sim
            }
        }
    }
}

fn select_neighbors_heuristic(
    storage: &HnswStorage,
    dim: usize,
    cands: &[(f32, u32)],
    skip_id: usize,
    limit: usize,
) -> Vec<u32> {
    let mut sel: Vec<u32> = Vec::with_capacity(limit);
    for &(d_cand, cid) in cands {
        if cid as usize == skip_id {
            continue;
        }
        let mut occluded = false;
        for &chosen in &sel {
            if storage.distance_nodes(dim, cid as usize, chosen as usize) < d_cand {
                occluded = true;
                break;
            }
        }
        if !occluded {
            sel.push(cid);
            if sel.len() >= limit {
                break;
            }
        }
    }
    if sel.len() < limit {
        for &(_, cid) in cands {
            if cid as usize != skip_id && !sel.contains(&cid) {
                sel.push(cid);
                if sel.len() >= limit {
                    break;
                }
            }
        }
    }
    sel
}

#[pyclass]
pub struct RustHNSWIndex {
    dim: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
    n_docs: usize,
    entry_point: Option<u32>,
    layers: Vec<Vec<Vec<u32>>>,
    storage: HnswStorage,
}

impl RustHNSWIndex {
    #[inline]
    fn distance_query_f32(&self, q_unit: &[f32], q_i8: &[i8], q_scale: f32, node: usize) -> f32 {
        match &self.storage {
            HnswStorage::Fp32(vecs) => {
                let d = &vecs[node * self.dim..(node + 1) * self.dim];
                1.0 - dot_f32_unrolled(q_unit, d)
            }
            HnswStorage::Sq8 { codes, scales } => {
                let d = &codes[node * self.dim..(node + 1) * self.dim];
                let sim = (dot_i8_unrolled(q_i8, d) as f32) * (q_scale * scales[node]);
                1.0 - sim
            }
        }
    }

    fn search_layer_with_query(
        &self,
        q_unit: &[f32],
        q_i8: &[i8],
        q_scale: f32,
        ep: u32,
        ef: usize,
        layer: usize,
        visited_bits: &mut [u8],
    ) -> Vec<(f32, u32)> {
        let ep_usize = ep as usize;
        visited_bits[ep_usize >> 3] |= 1u8 << (ep_usize & 7);

        let d0 = self.distance_query_f32(q_unit, q_i8, q_scale, ep_usize);
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        candidates.push(MinDistNode { dist: d0, idx: ep });
        results.push(MaxDistNode { dist: d0, idx: ep });

        let adj = &self.layers[layer];
        while let Some(curr) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if curr.dist > worst.dist && results.len() >= ef {
                    break;
                }
            }
            let curr_u = curr.idx as usize;
            if curr_u >= adj.len() {
                continue;
            }
            for &nb in &adj[curr_u] {
                let nb_u = nb as usize;
                let byte_i = nb_u >> 3;
                let mask = 1u8 << (nb_u & 7);
                if (visited_bits[byte_i] & mask) != 0 {
                    continue;
                }
                visited_bits[byte_i] |= mask;
                let d_nb = self.distance_query_f32(q_unit, q_i8, q_scale, nb_u);
                let should_add = results.len() < ef
                    || results.peek().map_or(true, |w| d_nb < w.dist);
                if should_add {
                    candidates.push(MinDistNode { dist: d_nb, idx: nb });
                    results.push(MaxDistNode { dist: d_nb, idx: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<(f32, u32)> = results.into_iter().map(|n| (n.dist, n.idx)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }

    fn search_single_unit(&self, q_unit: &[f32], top_k: usize, ef_override: Option<usize>) -> Vec<(usize, f32)> {
        if self.n_docs == 0 || top_k == 0 || self.layers.is_empty() {
            return Vec::new();
        }
        let ep_start = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };

        let mut q_i8 = Vec::new();
        let mut q_scale = 1.0f32;
        if matches!(self.storage, HnswStorage::Sq8 { .. }) {
            q_i8.resize(self.dim, 0i8);
            q_scale = quantize_unit_vec_sq8(q_unit, &mut q_i8);
        }

        let mut ep = ep_start;
        let mut curr_d = self.distance_query_f32(q_unit, &q_i8, q_scale, ep as usize);
        for layer in (1..self.layers.len()).rev() {
            let adj = &self.layers[layer];
            loop {
                let mut changed = false;
                let ep_u = ep as usize;
                if ep_u < adj.len() {
                    for &nb in &adj[ep_u] {
                        let d_nb = self.distance_query_f32(q_unit, &q_i8, q_scale, nb as usize);
                        if d_nb < curr_d {
                            curr_d = d_nb;
                            ep = nb;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let ef = ef_override.unwrap_or(self.ef_search).max(top_k).min(self.n_docs);
        let mut visited_bits = vec![0u8; (self.n_docs + 7) / 8];
        let cands = self.search_layer_with_query(q_unit, &q_i8, q_scale, ep, ef, 0, &mut visited_bits);
        cands
            .into_iter()
            .take(top_k)
            .map(|(dist, idx)| (idx as usize, 1.0 - dist))
            .collect()
    }
}

#[pymethods]
impl RustHNSWIndex {
    #[new]
    #[pyo3(signature = (dim, m=16, ef_construction=100, ef_search=64, quantization="sq8", seed=42))]
    fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        quantization: &str,
        seed: u64,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Số chiều vector phải lớn hơn 0"));
        }
        let q_mode = quantization.trim().to_lowercase();
        let storage = match q_mode.as_str() {
            "fp32" => HnswStorage::Fp32(Vec::new()),
            "sq8" => HnswStorage::Sq8 {
                codes: Vec::new(),
                scales: Vec::new(),
            },
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "quantization chỉ hỗ trợ 'fp32' hoặc 'sq8'",
                ));
            }
        };
        let m_val = m.max(4);
        Ok(Self {
            dim,
            m: m_val,
            ef_construction: ef_construction.max(m_val * 2),
            ef_search: ef_search.max(m_val),
            seed,
            n_docs: 0,
            entry_point: None,
            layers: Vec::new(),
            storage,
        })
    }

    fn build_from_bytes(&mut self, py: Python<'_>, matrix_f32_le: &[u8], n_docs: usize) -> PyResult<usize> {
        if n_docs == 0 {
            self.n_docs = 0;
            self.entry_point = None;
            self.layers.clear();
            return Ok(0);
        }
        let req = n_docs
            .checked_mul(self.dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận HNSW"))?;
        if matrix_f32_le.len() < req {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Bộ đệm float32 không đủ kích thước để dựng RustHNSWIndex",
            ));
        }

        let dim = self.dim;
        let is_sq8 = matches!(self.storage, HnswStorage::Sq8 { .. });
        let m = self.m;
        let max_m0 = m * 2;
        let ef_c = self.ef_construction;
        let seed = self.seed;

        let (storage_new, layers_new, entry_pt) = py.allow_threads(|| {
            let storage_built = if is_sq8 {
                let mut codes = vec![0i8; n_docs * dim];
                let mut scales = vec![0.0f32; n_docs];
                codes
                    .par_chunks_mut(dim)
                    .zip(scales.par_iter_mut())
                    .enumerate()
                    .for_each(|(doc_idx, (dst_c, dst_s))| {
                        let mut unit = vec![0.0f32; dim];
                        normalize_f32_from_bytes(matrix_f32_le, doc_idx * dim, dim, &mut unit);
                        *dst_s = quantize_unit_vec_sq8(&unit, dst_c);
                    });
                HnswStorage::Sq8 { codes, scales }
            } else {
                let mut vecs = vec![0.0f32; n_docs * dim];
                vecs.par_chunks_mut(dim)
                    .enumerate()
                    .for_each(|(doc_idx, dst_v)| {
                        normalize_f32_from_bytes(matrix_f32_le, doc_idx * dim, dim, dst_v);
                    });
                HnswStorage::Fp32(vecs)
            };

            let mut tmp = RustHNSWIndex {
                dim,
                m,
                ef_construction: ef_c,
                ef_search: self.ef_search,
                seed,
                n_docs,
                entry_point: Some(0),
                layers: vec![vec![Vec::new(); n_docs]],
                storage: storage_built,
            };

            let m_l = 1.0f64 / (m as f64).ln();
            let mut rng = XorShift64::new(seed);
            let mut max_level: usize = 0;
            let mut ep: u32 = 0;

            let mut visited_epoch = vec![0u32; n_docs];
            let mut cur_epoch: u32 = 1;
            let mut q_unit_buf = vec![0.0f32; dim];
            let mut q_i8_buf = vec![0i8; dim];

            for idx in 1..n_docs {
                let u = rng.next_f64();
                let level = ((-u.ln()) * m_l) as usize;
                while tmp.layers.len() <= level {
                    tmp.layers.push(vec![Vec::new(); n_docs]);
                }

                let mut q_scale = 1.0f32;
                match &tmp.storage {
                    HnswStorage::Fp32(v) => {
                        q_unit_buf.copy_from_slice(&v[idx * dim..(idx + 1) * dim]);
                    }
                    HnswStorage::Sq8 { codes, scales } => {
                        q_i8_buf.copy_from_slice(&codes[idx * dim..(idx + 1) * dim]);
                        q_scale = scales[idx];
                    }
                }

                let mut curr_ep = ep;
                let mut curr_d = tmp.distance_query_f32(&q_unit_buf, &q_i8_buf, q_scale, curr_ep as usize);

                if max_level > level {
                    for tang in (level + 1..=max_level).rev() {
                        loop {
                            let mut changed = false;
                            for &nb in &tmp.layers[tang][curr_ep as usize] {
                                let d_nb = tmp.distance_query_f32(&q_unit_buf, &q_i8_buf, q_scale, nb as usize);
                                if d_nb < curr_d {
                                    curr_d = d_nb;
                                    curr_ep = nb;
                                    changed = true;
                                }
                            }
                            if !changed {
                                break;
                            }
                        }
                    }
                }

                let top_tang = level.min(max_level);
                for tang in (0..=top_tang).rev() {
                    cur_epoch = cur_epoch.wrapping_add(1);
                    if cur_epoch == 0 {
                        visited_epoch.fill(0);
                        cur_epoch = 1;
                    }
                    visited_epoch[curr_ep as usize] = cur_epoch;

                    let d0 = tmp.distance_query_f32(&q_unit_buf, &q_i8_buf, q_scale, curr_ep as usize);
                    let mut candidates = BinaryHeap::new();
                    let mut results = BinaryHeap::new();
                    candidates.push(MinDistNode { dist: d0, idx: curr_ep });
                    results.push(MaxDistNode { dist: d0, idx: curr_ep });

                    while let Some(c) = candidates.pop() {
                        if let Some(w) = results.peek() {
                            if c.dist > w.dist && results.len() >= ef_c {
                                break;
                            }
                        }
                        for &nb in &tmp.layers[tang][c.idx as usize] {
                            let nb_u = nb as usize;
                            if visited_epoch[nb_u] == cur_epoch {
                                continue;
                            }
                            visited_epoch[nb_u] = cur_epoch;
                            let d_nb = tmp.distance_query_f32(&q_unit_buf, &q_i8_buf, q_scale, nb_u);
                            if results.len() < ef_c || results.peek().map_or(true, |w| d_nb < w.dist) {
                                candidates.push(MinDistNode { dist: d_nb, idx: nb });
                                results.push(MaxDistNode { dist: d_nb, idx: nb });
                                if results.len() > ef_c {
                                    results.pop();
                                }
                            }
                        }
                    }

                    let mut sorted_cands: Vec<(f32, u32)> =
                        results.into_iter().map(|n| (n.dist, n.idx)).collect();
                    sorted_cands.sort_by(|a, b| a.0.total_cmp(&b.0));

                    if let Some(&(_, best_id)) = sorted_cands.first() {
                        curr_ep = best_id;
                    }

                    let cap = if tang == 0 { max_m0 } else { m };
                    let selected = select_neighbors_heuristic(&tmp.storage, dim, &sorted_cands, idx, cap);
                    tmp.layers[tang][idx] = selected.clone();
                    let idx_u32 = idx as u32;
                    for &nb in &selected {
                        let nb_u = nb as usize;
                        if !tmp.layers[tang][nb_u].contains(&idx_u32) {
                            tmp.layers[tang][nb_u].push(idx_u32);
                        }
                        if tmp.layers[tang][nb_u].len() > cap {
                            let mut scored_nb: Vec<(f32, u32)> = tmp.layers[tang][nb_u]
                                .iter()
                                .map(|&x| (tmp.storage.distance_nodes(dim, nb_u, x as usize), x))
                                .collect();
                            scored_nb.sort_by(|a, b| a.0.total_cmp(&b.0));
                            tmp.layers[tang][nb_u] =
                                select_neighbors_heuristic(&tmp.storage, dim, &scored_nb, nb_u, cap);
                        }
                    }
                }

                if level > max_level {
                    max_level = level;
                    ep = idx as u32;
                }
            }

            (tmp.storage, tmp.layers, Some(ep))
        });

        self.storage = storage_new;
        self.layers = layers_new;
        self.entry_point = entry_pt;
        self.n_docs = n_docs;
        Ok(n_docs)
    }

    #[pyo3(signature = (query_f32_le, top_k=10, ef_search=None))]
    fn search_bytes(
        &self,
        py: Python<'_>,
        query_f32_le: &[u8],
        top_k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<(usize, f32)>> {
        if query_f32_le.len() < self.dim * 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Bộ đệm query float32 không đủ kích thước cho RustHNSWIndex",
            ));
        }
        let dim = self.dim;
        let out = py.allow_threads(|| {
            let mut q_unit = vec![0.0f32; dim];
            normalize_f32_from_bytes(query_f32_le, 0, dim, &mut q_unit);
            self.search_single_unit(&q_unit, top_k, ef_search)
        });
        Ok(out)
    }

    #[pyo3(signature = (queries_f32_le, n_queries, top_k=10, ef_search=None))]
    fn search_batch_bytes(
        &self,
        py: Python<'_>,
        queries_f32_le: &[u8],
        n_queries: usize,
        top_k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        if n_queries == 0 {
            return Ok(Vec::new());
        }
        let req = n_queries
            .checked_mul(self.dim)
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước batch query"))?;
        if queries_f32_le.len() < req {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Bộ đệm batch query float32 không đủ kích thước",
            ));
        }
        let dim = self.dim;
        let out = py.allow_threads(|| {
            (0..n_queries)
                .into_par_iter()
                .map(|qi| {
                    let mut q_unit = vec![0.0f32; dim];
                    normalize_f32_from_bytes(queries_f32_le, qi * dim, dim, &mut q_unit);
                    self.search_single_unit(&q_unit, top_k, ef_search)
                })
                .collect()
        });
        Ok(out)
    }

    fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let raw = py.allow_threads(|| {
            let mut buf = Vec::with_capacity(self.memory_bytes() + 128);
            buf.extend_from_slice(b"VHNS");
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
            buf.extend_from_slice(&(self.m as u32).to_le_bytes());
            buf.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
            buf.extend_from_slice(&(self.ef_search as u32).to_le_bytes());
            buf.extend_from_slice(&self.seed.to_le_bytes());
            let q_flag: u8 = match self.storage {
                HnswStorage::Fp32(_) => 0,
                HnswStorage::Sq8 { .. } => 1,
            };
            buf.push(q_flag);
            buf.extend_from_slice(&(self.n_docs as u32).to_le_bytes());
            let ep_i64: i64 = self.entry_point.map_or(-1, |e| e as i64);
            buf.extend_from_slice(&ep_i64.to_le_bytes());
            buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());

            for layer in &self.layers {
                let non_empty: Vec<(u32, &Vec<u32>)> = layer
                    .iter()
                    .enumerate()
                    .filter(|(_, nbs)| !nbs.is_empty())
                    .map(|(nid, nbs)| (nid as u32, nbs))
                    .collect();
                buf.extend_from_slice(&(non_empty.len() as u32).to_le_bytes());
                for (nid, nbs) in non_empty {
                    buf.extend_from_slice(&nid.to_le_bytes());
                    buf.extend_from_slice(&(nbs.len() as u32).to_le_bytes());
                    for &nb in nbs {
                        buf.extend_from_slice(&nb.to_le_bytes());
                    }
                }
            }

            match &self.storage {
                HnswStorage::Fp32(vecs) => {
                    for &v in vecs {
                        buf.extend_from_slice(&v.to_le_bytes());
                    }
                }
                HnswStorage::Sq8 { codes, scales } => {
                    for &c in codes {
                        buf.push(c as u8);
                    }
                    for &s in scales {
                        buf.extend_from_slice(&s.to_le_bytes());
                    }
                }
            }
            buf
        });
        Ok(PyBytes::new_bound(py, &raw).unbind())
    }

    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        if data.len() < 49 || &data[0..4] != b"VHNS" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Định dạng nhị phân RustHNSWIndex không hợp lệ (sai magic header VHNS)",
            ));
        }
        let mut off = 4usize;
        let version = read_u32_le(data, &mut off)?;
        if version != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Phiên bản nhị phân RustHNSWIndex không được hỗ trợ",
            ));
        }
        let dim = read_u32_le(data, &mut off)? as usize;
        let m = read_u32_le(data, &mut off)? as usize;
        let ef_construction = read_u32_le(data, &mut off)? as usize;
        let ef_search = read_u32_le(data, &mut off)? as usize;
        let seed = read_i64_le(data, &mut off)? as u64;
        if off >= data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Dữ liệu HNSW bị cắt cụt"));
        }
        let q_flag = data[off];
        off += 1;
        let n_docs = read_u32_le(data, &mut off)? as usize;
        let ep_i64 = read_i64_le(data, &mut off)?;
        let entry_point = if ep_i64 >= 0 { Some(ep_i64 as u32) } else { None };
        let num_layers = read_u32_le(data, &mut off)? as usize;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut adj = vec![Vec::new(); n_docs];
            let count = read_u32_le(data, &mut off)? as usize;
            for _ in 0..count {
                let nid = read_u32_le(data, &mut off)? as usize;
                let deg = read_u32_le(data, &mut off)? as usize;
                let mut nbs = Vec::with_capacity(deg);
                for _ in 0..deg {
                    nbs.push(read_u32_le(data, &mut off)?);
                }
                if nid < n_docs {
                    adj[nid] = nbs;
                }
            }
            layers.push(adj);
        }

        let storage = py.allow_threads(|| -> PyResult<HnswStorage> {
            if q_flag == 0 {
                let n_elems = n_docs * dim;
                if off + n_elems * 4 > data.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Thiếu dữ liệu vector fp32 trong RustHNSWIndex",
                    ));
                }
                let mut vecs = Vec::with_capacity(n_elems);
                for i in 0..n_elems {
                    let o = off + i * 4;
                    vecs.push(f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]));
                }
                Ok(HnswStorage::Fp32(vecs))
            } else if q_flag == 1 {
                let n_elems = n_docs * dim;
                if off + n_elems + n_docs * 4 > data.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Thiếu dữ liệu vector sq8 trong RustHNSWIndex",
                    ));
                }
                let codes: Vec<i8> = data[off..off + n_elems].iter().map(|&b| b as i8).collect();
                off += n_elems;
                let mut scales = Vec::with_capacity(n_docs);
                for i in 0..n_docs {
                    let o = off + i * 4;
                    scales.push(f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]));
                }
                Ok(HnswStorage::Sq8 { codes, scales })
            } else {
                Err(pyo3::exceptions::PyValueError::new_err(
                    "Mã lượng tử hóa không hợp lệ trong RustHNSWIndex",
                ))
            }
        })?;

        Ok(Self {
            dim,
            m,
            ef_construction,
            ef_search,
            seed,
            n_docs,
            entry_point,
            layers,
            storage,
        })
    }

    fn memory_bytes(&self) -> usize {
        let vec_bytes = match &self.storage {
            HnswStorage::Fp32(v) => v.len() * 4,
            HnswStorage::Sq8 { codes, scales } => codes.len() + scales.len() * 4,
        };
        let graph_bytes: usize = self
            .layers
            .iter()
            .map(|layer| layer.iter().map(|nbs| nbs.len() * 4).sum::<usize>())
            .sum();
        vec_bytes + graph_bytes
    }

    fn vector_bytes(&self) -> usize {
        match &self.storage {
            HnswStorage::Fp32(v) => v.len() * 4,
            HnswStorage::Sq8 { codes, scales } => codes.len() + scales.len() * 4,
        }
    }

    fn len(&self) -> usize {
        self.n_docs
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn quantization(&self) -> String {
        match self.storage {
            HnswStorage::Fp32(_) => "fp32".to_string(),
            HnswStorage::Sq8 { .. } => "sq8".to_string(),
        }
    }
}

#[pyfunction]
pub fn fast_bitnet_158_pack_bytes(
    py: Python<'_>,
    weights_f32_le: &[u8],
    out_features: usize,
    in_features: usize,
) -> PyResult<(Py<PyBytes>, f32)> {
    if out_features == 0 || in_features == 0 {
        return Ok((PyBytes::new_bound(py, &[]).unbind(), 1.0));
    }
    let total = out_features
        .checked_mul(in_features)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước ma trận BitNet"))?;
    let req = total
        .checked_mul(4)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tràn số kích thước byte BitNet"))?;
    if weights_f32_le.len() < req {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm trọng số float32 không đủ cho fast_bitnet_158_pack_bytes",
        ));
    }

    let packed_cols = (in_features + 3) / 4;
    let (packed, scale) = py.allow_threads(|| {
        let sum_abs: f64 = (0..total)
            .into_par_iter()
            .map(|i| (read_f32_le(weights_f32_le, i) as f64).abs())
            .sum();
        let scale = ((sum_abs / (total as f64)) as f32).max(1e-5);
        let inv_scale = 1.0f32 / scale;

        let mut out = vec![0u8; out_features * packed_cols];
        out.par_chunks_mut(packed_cols)
            .enumerate()
            .for_each(|(r, row_bytes)| {
                let row_base = r * in_features;
                for b in 0..packed_cols {
                    let k0 = b * 4;
                    let mut byte_val = 0u8;
                    for slot in 0..4 {
                        let k = k0 + slot;
                        let code: u8 = if k < in_features {
                            let q = (read_f32_le(weights_f32_le, row_base + k) * inv_scale)
                                .round()
                                .clamp(-1.0, 1.0) as i8;
                            (q + 1) as u8
                        } else {
                            1u8
                        };
                        byte_val |= code << (6 - slot * 2);
                    }
                    row_bytes[b] = byte_val;
                }
            });
        (out, scale)
    });

    Ok((PyBytes::new_bound(py, &packed).unbind(), scale))
}

#[pyfunction]
pub fn fast_bitnet_158_matmul_bytes(
    py: Python<'_>,
    x_f32_le: &[u8],
    packed_w: &[u8],
    n_rows: usize,
    in_features: usize,
    out_features: usize,
    scale: f32,
) -> PyResult<Py<PyBytes>> {
    if n_rows == 0 || in_features == 0 || out_features == 0 {
        return Ok(PyBytes::new_bound(py, &[]).unbind());
    }
    let packed_cols = (in_features + 3) / 4;
    if x_f32_le.len() < n_rows * in_features * 4 || packed_w.len() < out_features * packed_cols {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Bộ đệm đầu vào hoặc trọng số 2-bit BitNet không đủ kích thước",
        ));
    }

    let out_bytes = py.allow_threads(|| {
        let x_f32: Vec<f32> = (0..n_rows * in_features)
            .map(|i| read_f32_le(x_f32_le, i))
            .collect();
        let total_out = n_rows * out_features;
        let mut y_f32 = vec![0.0f32; total_out];

        #[inline]
        fn build_nibble_lut(a: f32, b: f32) -> [f32; 16] {
            [
                -a - b,
                -a,
                -a + b,
                -a,
                -b,
                0.0,
                b,
                0.0,
                a - b,
                a,
                a + b,
                a,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        }

        let mut lut: Vec<[[f32; 16]; 2]> = vec![[[0.0f32; 16]; 2]; packed_cols];
        for row_i in 0..n_rows {
            let x_row = &x_f32[row_i * in_features..(row_i + 1) * in_features];
            for b in 0..packed_cols {
                let k = b * 4;
                let x0 = if k < in_features { x_row[k] } else { 0.0 };
                let x1 = if k + 1 < in_features { x_row[k + 1] } else { 0.0 };
                let x2 = if k + 2 < in_features { x_row[k + 2] } else { 0.0 };
                let x3 = if k + 3 < in_features { x_row[k + 3] } else { 0.0 };
                lut[b][0] = build_nibble_lut(x0, x1);
                lut[b][1] = build_nibble_lut(x2, x3);
            }

            let y_row = &mut y_f32[row_i * out_features..(row_i + 1) * out_features];
            let lut_ref = &lut;
            y_row.par_chunks_mut(32).enumerate().for_each(|(chunk_i, dst_chunk)| {
                let o_base = chunk_i * 32;
                for (local_i, dst) in dst_chunk.iter_mut().enumerate() {
                    let o_col = o_base + local_i;
                    let w_row = &packed_w[o_col * packed_cols..(o_col + 1) * packed_cols];
                    let mut a0 = 0.0f32;
                    let mut a1 = 0.0f32;
                    let mut a2 = 0.0f32;
                    let mut a3 = 0.0f32;
                    let mut b = 0usize;
                    while b + 4 <= packed_cols {
                        let b0 = w_row[b];
                        let b1 = w_row[b + 1];
                        let b2 = w_row[b + 2];
                        let b3 = w_row[b + 3];
                        let l0 = &lut_ref[b];
                        let l1 = &lut_ref[b + 1];
                        let l2 = &lut_ref[b + 2];
                        let l3 = &lut_ref[b + 3];
                        a0 += l0[0][(b0 >> 4) as usize] + l0[1][(b0 & 0x0F) as usize];
                        a1 += l1[0][(b1 >> 4) as usize] + l1[1][(b1 & 0x0F) as usize];
                        a2 += l2[0][(b2 >> 4) as usize] + l2[1][(b2 & 0x0F) as usize];
                        a3 += l3[0][(b3 >> 4) as usize] + l3[1][(b3 & 0x0F) as usize];
                        b += 4;
                    }
                    let mut acc = (a0 + a1) + (a2 + a3);
                    while b < packed_cols {
                        let bv = w_row[b];
                        let lb = &lut_ref[b];
                        acc += lb[0][(bv >> 4) as usize] + lb[1][(bv & 0x0F) as usize];
                        b += 1;
                    }
                    *dst = acc * scale;
                }
            });
        }

        let mut bytes = Vec::with_capacity(total_out * 4);
        for v in y_f32 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    });

    Ok(PyBytes::new_bound(py, &out_bytes).unbind())
}
