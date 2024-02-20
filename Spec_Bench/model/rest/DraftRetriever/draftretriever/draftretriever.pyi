import typing


class Writer:
    def __init__(
        self,
        index_file_path: str,
        max_chunk_len: typing.Optional[int] = None,
        vocab_size: typing.Optional[int] = None,
    ) -> None: ...


    def add_entry(
        self,
        py_text: typing.List,
    ) -> None: ...

    def dump_data(
        self,
    ) -> None: ...

    def finalize(
        self,
    ) -> None: ...


class Reader:
    def __init__(
        self,
        index_file_path: str,
    ) -> None: ...

    def search(
        self,
        py_substring: typing.List,
        k: typing.Optional[int] = None,
        choices: typing.Optional[int] = None,
        long: typing.Optional[int] = None,
    ): ...
