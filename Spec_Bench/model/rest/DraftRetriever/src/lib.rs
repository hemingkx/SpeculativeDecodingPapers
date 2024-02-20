// The code for retrival is adapted from https://github.com/Intsights/PySubstringSearch; 
// The code for drafft buffer is adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py#L31-L124
use ahash::AHashSet;
use bstr::io::BufReadExt;
use byteorder::{ReadBytesExt, WriteBytesExt, ByteOrder, LittleEndian};
use memchr::memmem;
use parking_lot::Mutex;
use pyo3::exceptions;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::str;
use std::sync::Arc;
use std::collections::HashMap;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::cmp;
use std::cmp::Ordering;
use pyo3::types::PyList;
use std::collections::BinaryHeap;
use std::fs;
use std::io::Cursor;
use std::fs::OpenOptions;

extern "C" {
    pub fn libsais_int(
        data: *const i32,
        suffix_array: *mut i32,
        data_len: i32,
        suffix_array_extra_space: i32,
        symbol_frequency_table: i32,
    ) -> i32;
}

fn construct_suffix_array(
    buffer: &[i32],
    vocab_size: i32,
) -> Vec<i32> {
    let mut suffix_array = vec![0; buffer.len()];

    unsafe {
        libsais_int(
            buffer.as_ptr(),
            suffix_array.as_mut_ptr(),
            buffer.len() as i32,
            vocab_size,
            0,
        );
    }

    suffix_array
}

#[pyclass]
struct Writer {
    index_file: BufWriter<File>,
    buffer: Vec<i32>,
    vocab_size: i32,
}

#[pymethods]
impl Writer {
    #[new]
    fn new(
        index_file_path: &str,
        max_chunk_len: Option<usize>,
        vocab_size: Option<i32>,
    ) -> PyResult<Self> {
        let index_file = File::create(index_file_path)?;
        let index_file = BufWriter::new(index_file);

        let max_chunk_len = max_chunk_len.unwrap_or(512 * 1024 * 1024);
        let vocab_size = vocab_size.unwrap_or(35000);

        Ok(
            Writer {
                index_file,
                buffer: Vec::with_capacity(max_chunk_len),
                vocab_size,
            }
        )
    }

    fn add_entry(
        &mut self,
        py_text: &PyList,
    ) -> PyResult<()> {

        let mut text = Vec::new();
        for item in py_text.iter() {
            let num: i32 = item.extract()?;
            text.push(num);
        }

        if text.len() > self.buffer.capacity() {
            return Err(exceptions::PyValueError::new_err("entry is too big"));
        }

        if self.buffer.len() + text.len() > self.buffer.capacity() {
            self.dump_data()?;
        }
        self.buffer.extend_from_slice(&text);

        // self.buffer.push(34999);

        Ok(())
    }

    fn dump_data(
        &mut self,
    ) -> PyResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        self.index_file.write_u32::<LittleEndian>((self.buffer.len() * 2) as u32)?;

        for &item in &self.buffer {
            self.index_file.write_u16::<LittleEndian>(item as u16)?;
        }

        let suffix_array = construct_suffix_array(&self.buffer, self.vocab_size);
        self.index_file.write_u32::<LittleEndian>((suffix_array.len() * 4) as u32)?;
        for suffix in suffix_array {
            self.index_file.write_i32::<LittleEndian>(suffix)?;
        }
        self.buffer.clear();

        Ok(())
    }

    fn finalize(
        &mut self,
    ) -> PyResult<()> {
        if !self.buffer.is_empty() {
            self.dump_data()?;
        }
        self.index_file.flush()?;

        Ok(())
    }
}

impl Drop for Writer {
    fn drop(
        &mut self,
    ) {
        self.finalize().unwrap();
    }
}

struct SubIndex {
    data: Vec<i32>,
    index_file: Cursor<Vec<u8>>, // BufReader<File>, // Cursor<Vec<u8>>,
    suffixes_file_start: usize,
    suffixes_file_end: usize,
}

#[pyclass]
struct Reader {
    sub_indexes: Vec<SubIndex>,
}

#[pymethods]
impl Reader {
    #[new]
    fn new(
        index_file_path: &str,
    ) -> PyResult<Self> {
        let index_file = File::open(index_file_path)?;
        let mut index_file = BufReader::new(index_file);
        let index_file_metadata = std::fs::metadata(index_file_path)?;
        let index_file_len = index_file_metadata.len();
        let mut bytes_read = 0;

        let mut sub_indexes = Vec::new();

        while bytes_read < index_file_len {
            let data_file_len = index_file.read_u32::<LittleEndian>()?;
            let mut data_u8 = vec![0; data_file_len as usize];
            index_file.read_exact(&mut data_u8)?;

            let suffixes_file_len = index_file.read_u32::<LittleEndian>()? as usize;
            let suffixes_file_start = index_file.seek(SeekFrom::Current(0))? as usize;
            let suffixes_file_end = suffixes_file_start + suffixes_file_len;
            index_file.seek(SeekFrom::Current(suffixes_file_len as i64))?;

            bytes_read += 4 + 4 + data_file_len as u64 + suffixes_file_len as u64;


            let mut data: Vec<i32> = Vec::new();

            for i in (0..data_u8.len()).step_by(2) {
                let int = LittleEndian::read_u16(&data_u8[i..i+2]) as i32;
                data.push(int);
            }

            sub_indexes.push(
                SubIndex {
                    data,
                    index_file: Cursor::new(fs::read(index_file_path).unwrap()), // BufReader::new(File::open(index_file_path)?), // Cursor::new(fs::read(index_file_path).unwrap()),
                    suffixes_file_start,
                    suffixes_file_end,
                }
            );
        }

        Ok(Reader { sub_indexes })
    }

    fn search(
        &mut self,
        py_substring: &PyList,
        k: Option<i32>,
        choices: Option<i32>,
        long: Option<i32>,
    ) -> PyResult<(Vec<Vec<i32>>, Vec<Vec<i32>>, Vec<i32>, Vec<i32>, Vec<Vec<i32>>)> {

        let mut substring_i32 = Vec::new();
        for item in py_substring.iter() {
            let num: i32 = item.extract()?;
            substring_i32.push(num);
        }

        let results = Arc::new(Mutex::new(Vec::new()));

        self.sub_indexes.par_iter_mut().for_each(
            |sub_index| {
                let mut start_of_indices = None;
                let mut end_of_indices = None;

                let mut left_anchor = sub_index.suffixes_file_start;
                let mut right_anchor = sub_index.suffixes_file_end - 4;
                while left_anchor <= right_anchor {
                    let middle_anchor = left_anchor + ((right_anchor - left_anchor) / 4 / 2 * 4);
                    sub_index.index_file.seek(SeekFrom::Start(middle_anchor as u64)).unwrap();
                    let data_index = sub_index.index_file.read_i32::<LittleEndian>().unwrap();
                    let line = &sub_index.data[(data_index) as usize..];

                    if line.starts_with(&substring_i32) {
                        start_of_indices = Some(middle_anchor);
                        right_anchor = middle_anchor - 4;
                    } else {
                        match line.cmp(&substring_i32) {
                            std::cmp::Ordering::Less => left_anchor = middle_anchor + 4,
                            std::cmp::Ordering::Greater => right_anchor = middle_anchor - 4,
                            std::cmp::Ordering::Equal => {},
                        };
                    }
                }
                if start_of_indices.is_none() {
                    return;
                }
                
                let mut right_anchor = sub_index.suffixes_file_end - 4;
                while left_anchor <= right_anchor {
                    let middle_anchor = left_anchor + ((right_anchor - left_anchor) / 4 / 2 * 4);
                    sub_index.index_file.seek(SeekFrom::Start(middle_anchor as u64)).unwrap();
                    let data_index = sub_index.index_file.read_i32::<LittleEndian>().unwrap();
                    let line = &sub_index.data[(data_index) as usize..];
                    if line.starts_with(&substring_i32) {
                        end_of_indices = Some(middle_anchor);
                        left_anchor = middle_anchor + 4;
                    } else {
                        match line.cmp(&substring_i32) {
                            std::cmp::Ordering::Less => left_anchor = middle_anchor + 4,
                            std::cmp::Ordering::Greater => right_anchor = middle_anchor - 4,
                            std::cmp::Ordering::Equal => {},
                        };
                    }
                }

                let start_of_indices = start_of_indices.unwrap();
                let end_of_indices = end_of_indices.unwrap();

                let mut suffixes = vec![0; end_of_indices - start_of_indices + 4];

                sub_index.index_file.seek(SeekFrom::Start(start_of_indices as u64)).unwrap();
                sub_index.index_file.read_exact(&mut suffixes).unwrap();

                let mut matches_ranges = AHashSet::new();

                let mut cnt = 0;
                let k = k.unwrap_or(5000);
                let long = long.unwrap_or(10);
                let indices_size = (end_of_indices - start_of_indices + 4) / 4;
                let initial_capacity = std::cmp::min(indices_size, k as usize);
                let mut local_results = Vec::with_capacity(initial_capacity);

                for suffix in suffixes.chunks_mut(4) {
                    let data_index = LittleEndian::read_i32(suffix);
                    if matches_ranges.insert(data_index) {
                        let sub_string_plus = &sub_index.data[data_index as usize + substring_i32.len() ..std::cmp::min(data_index as usize + substring_i32.len() + long as usize,  sub_index.data.len())];
                    
                        local_results.push(sub_string_plus.to_vec());
                        cnt += 1;
                        if cnt >= k as usize {
                            break;
                        }

                    }
                }

                results.lock().extend(local_results);
            }
        );

        let results = results.lock();

        if results.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        let mut cnt = HashMap::new();
        for retrieved_token in &*results {
            for j in 0..retrieved_token.len() {
                let tmp_token = &retrieved_token[0..=j];
                let counter = cnt.entry(tmp_token).or_insert(0);
                *counter += 1;
            }
        }
        
        let choices = choices.unwrap_or(64);
        // The items in the heap must be a Trie.
        let mut heap = BinaryHeap::new();
        for (k, v) in &cnt {
            if heap.len() < (choices as usize) {
                heap.push((Reverse(*v), k));
            } else if let Some(&(Reverse(top_v), _)) = heap.peek() {
                if *v > top_v {
                    heap.pop();
                    heap.push((Reverse(*v), k));
                }
            }
        }
        let verified: Vec<_> = heap.into_iter().map(|(_, k)| k.to_vec()).collect();
        // Convert into a HashSet to remove duplicates
        let verified: std::collections::HashSet<_> = verified.into_iter().collect();
        let verified: Vec<_> = verified.into_iter().collect();

        // Because multiple nodes in the Trie may have same weights around the threshold, the number of draft tokens may exceed choices
        // We roughly cut nodes to be less than choices in most cases. 
        let paths = cut_to_choices(verified, choices);

        let (draft_choices, max_branch) = get_draft_choices(paths.clone());

        let (draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices) = generate_draft_buffers(draft_choices.clone(), max_branch);

        let max_length = paths.iter().map(|path| path.len()).max().unwrap_or(0);

        Ok((paths.into_iter().map(|path| pad_path(path, max_length, -2)).collect::<Vec<Vec<i32>>>(), draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices))
    }
}



fn cut_to_choices(paths: Vec<Vec<i32>>, choices: i32) -> Vec<Vec<i32>> {
    let mut count: Vec<(usize, usize)> = paths.iter()
                                               .map(|p| (p.iter().collect::<std::collections::HashSet<&i32>>().len(), paths.iter().position(|x| x == p).unwrap()))
                                               .collect();
    count.sort_by(|a, b| b.0.cmp(&a.0));

    let mut total_unique = count.iter().map(|(x, _)| x).sum::<usize>();
    let mut to_remove = Vec::new();

    for (c, i) in count {
        if total_unique > choices as usize {
            total_unique -= c;
            to_remove.push(i);
        } else {
            break;
        }
    }

    paths.into_iter().enumerate().filter(|(i, _)| !to_remove.contains(i)).map(|(_, p)| p).collect()
}


fn get_draft_choices(paths: Vec<Vec<i32>>) -> (Vec<Vec<i32>>, i32) {
    let mut path_dict: HashMap<i32, HashMap<i32, i32>> = HashMap::new();
    let mut cnt_dict: HashMap<i32, i32> = HashMap::new();
    let max_depth = paths.iter().map(|path| path.len() as i32).max().unwrap();

    for depth in 0..max_depth {
        cnt_dict.insert(depth, 0);
    }

    for path in &paths {
        for (depth, item) in path.iter().enumerate() {
            let depth = depth as i32;
            if !path_dict.contains_key(&depth) {
                path_dict.insert(depth, HashMap::new());
            }

            let current_path_dict = path_dict.get_mut(&depth).unwrap();
            if !current_path_dict.contains_key(item) {
                let current_cnt = cnt_dict.get(&depth).unwrap().clone();
                current_path_dict.insert(*item, current_cnt);
                *cnt_dict.get_mut(&depth).unwrap() += 1;
            }
        }
    }

    let max_branch = path_dict.values().map(|v| v.len() as i32).max().unwrap();

    let mut draft_choices: HashSet<Vec<i32>> = HashSet::new();
    for path in paths {
        for (depth, _) in path.iter().enumerate() {
            let depth = depth as i32;
            let draft_choice: Vec<i32> = (0..=depth)
                .map(|prev_depth| {
                    let prev_item = *path.get(prev_depth as usize).unwrap();
                    *path_dict.get(&prev_depth).unwrap().get(&prev_item).unwrap()
                })
                .collect();
            draft_choices.insert(draft_choice);
        }
    }

    let draft_choices: Vec<Vec<i32>> = draft_choices.into_iter().collect();
    (draft_choices, max_branch)
}



fn pad_path(path: Vec<i32>, length: usize, pad_value: i32) -> Vec<i32> {
    let mut path = path;
    while path.len() < length {
        path.push(pad_value);
    }
    path
}


fn generate_draft_buffers(draft_choices: Vec<Vec<i32>>, topk: i32) -> (Vec<Vec<i32>>, Vec<i32>, Vec<i32>, Vec<Vec<i32>>) {

    // Sort the draft_choices based on their lengths and then their values
    let mut sorted_draft_choices = draft_choices;
    sorted_draft_choices.sort_by(|a, b| match a.len().cmp(&b.len()) {
        Ordering::Equal => a.cmp(b),
        other => other,
    });

    let draft_len = sorted_draft_choices.len() + 1;
    assert! (draft_len <= 65, "draft_len should not exceed 65");
    // Initialize depth_counts to keep track of how many choices have a particular depth
    let mut depth_counts:Vec<i32> = vec![0; draft_len];
    let mut prev_depth = 0;
    for path in &sorted_draft_choices {
        let depth = path.len();
        if depth != prev_depth {
            depth_counts[depth - 1] = 0;
        }
        depth_counts[depth - 1] += 1;
        prev_depth = depth;
    }
    // Create the attention mask for draft
    let mut draft_attn_mask:Vec<Vec<i32>> = vec![vec![0; draft_len]; draft_len];
    for i in 0..draft_len {
        draft_attn_mask[i][0] = 1;
        draft_attn_mask[i][i] = 1;
    }

    let mut start = 0;
    for i in 0..depth_counts.len() {
        for j in 0..depth_counts[i] {
            let cur_draft_choice: Vec<i32> = sorted_draft_choices[(start + j) as usize].clone();
            if cur_draft_choice.len() == 1 {
                continue;
            }

            let mut ancestor_idx = vec![];
            for c in 0..(cur_draft_choice.len() - 1) {
                let index = sorted_draft_choices.iter().position(|x| x[..=cmp::min(c, x.len() - 1)] == cur_draft_choice[..=cmp::min(c, cur_draft_choice.len() - 1)]).unwrap() + 1;
                ancestor_idx.push(index);
            }

            for idx in ancestor_idx {
                draft_attn_mask[(j + start + 1) as usize][idx] = 1;
            }
        }
        start += depth_counts[i];
    }

    // Generate tree indices for the draft structure
    let mut draft_tree_indices: Vec<i32> = vec![0; draft_len];
    let mut start = 0;
    for i in 0..depth_counts.len() {
        for j in 0..depth_counts[i] {
            let cur_draft_choice = &sorted_draft_choices[(start + j) as usize];
            draft_tree_indices[(start + j + 1) as usize] = cur_draft_choice.last().unwrap() + topk * (i as i32) + 1;
        }
        start += depth_counts[i];
    }

    // Generate position IDs for the draft structure
    let mut draft_position_ids: Vec<i32> = vec![0; draft_len];
    start = 0;
    for i in 0..depth_counts.len() {
        for j in start + 1..start + depth_counts[i] + 1 {
            draft_position_ids[j as usize] = (i as i32) + 1;
        }
        start += depth_counts[i];
    }

    // Generate retrieval indices for draft structure verification
    let mut retrieve_indices_nest = Vec::new();
    let mut retrieve_paths = Vec::new();
    for i in 0..sorted_draft_choices.len() {
        let cur_draft_choice = sorted_draft_choices[sorted_draft_choices.len() - 1 - i].clone();
        let mut retrieve_indice = Vec::new();
        if retrieve_paths.contains(&cur_draft_choice) {
            continue;
        } else {
            for c in 0..cur_draft_choice.len() {
                let index = sorted_draft_choices.iter().position(|x| *x == cur_draft_choice[0..=c]).unwrap();
                retrieve_indice.push(index as i32);
                retrieve_paths.push(cur_draft_choice[0..=c].to_vec());
            }
        }
        retrieve_indices_nest.push(retrieve_indice);
    }
    let max_length = retrieve_indices_nest.iter().map(|x| x.len()).max().unwrap();
    let mut retrieve_indices: Vec<Vec<i32>> = retrieve_indices_nest.iter().map(|x| pad_path(x.clone(), max_length, -2)).collect();

    for i in 0..retrieve_indices.len() {
        for j in 0..retrieve_indices[i].len() {
            retrieve_indices[i][j] += 1;
        }
    }

    for i in 0..retrieve_indices.len() {
        retrieve_indices[i].insert(0, 0);
    }


    (draft_attn_mask, draft_tree_indices, draft_position_ids, retrieve_indices)
}


#[pymodule]
fn draftretriever(
    _py: Python,
    m: &PyModule,
) -> PyResult<()> {
    m.add_class::<Writer>()?;
    m.add_class::<Reader>()?;

    Ok(())
}

