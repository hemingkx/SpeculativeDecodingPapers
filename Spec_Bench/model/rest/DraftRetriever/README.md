Retriever for searching draft tokens for speculative decoding


## Table of Contents

- [Table of Contents](#table-of-contents)
- [About The Project](#about-the-project)
  - [Built With](#built-with)
  - [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## About The Project

DraftRerriever is a library designed to searching draft tokens for speculative decoding. In order to achieve speed and efficiency, the library is written in Rust. For string indexing, the library uses [libsais](https://github.com/IlyaGrebnov/libsais) suffix array construction library. The datastore created consists of the original 16bit tokens and a 32bit suffix array struct. 

The module implements a method for searching.
- `search` - Find multiple candidates give preceding tokens, and return the most probable draft tokens by constructing a Trie. It also returns draft buffer.


### Built With

* [libsais](https://github.com/IlyaGrebnov/libsais)


### Installation

**Use pre-compiled wheels**
```sh
pip3 install wheels/draftretriever-0.1.0-cp39-cp39-manylinux_2_34_x86_64.whl
```

**Build from source**
```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
maturin build --release --strip -i python3.9 # will produce a .whl file
pip3 install [.whl]
```


## Usage

Create a datastore
```python
import draftretriever

# creating a new datastore
# if a file with this name is already exists, it will be overwritten
writer = draftretriever.Writer(
    index_file_path='output.idx',
    # vocab_size=tokenizer.vocab_size
)

# adding entries to the new datastore
writer.add_entry([1, 2, 3, 4]) # a list of token
writer.add_entry([1, 2, 3, 4])
writer.add_entry([2, 3, 5, 6])

# making sure the data is dumped to the file
writer.finalize()
```

Search draft tokens
```python
import draftretriever

# opening a datastore for searching
reader = draftretriever.Reader(
    index_file_path='output.idx',
)

# search for draft tokens
preceding = [2, 3]
# "choices" is the number of maximum draft tokens. The implementation is not very strict and has some randomness.
retrieved_token_list, _draft_attn_mask, _tree_indices, _draft_position_ids, _retrieve_indices = reader.search(preceding, choices=2)
print(retrieved_token_list)
>>> [[4]] or [[4], [5]]
# retrieved_token_list is a list of selected paths(sequences) in the Trie. Each sequence are padded (-2) to the maximum length of these sequences.
```

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgement
The main framework is from [PySubstringSearch](https://github.com/Intsights/PySubstringSearch)


