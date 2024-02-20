import typing

from . import draftretriever


class Writer:
    def __init__(
        self,
        index_file_path: str,
        max_chunk_len: typing.Optional[int] = None,
        vocab_size: typing.Optional[int] = None,
    ) -> None:
        self.writer = draftretriever.Writer(
            index_file_path=index_file_path,
            max_chunk_len=max_chunk_len,
            vocab_size=vocab_size,
        )

    def add_entry(
        self,
        py_text: typing.List,
    ) -> None:
        self.writer.add_entry(
            py_text=py_text,
        )

    def dump_data(
        self,
    ) -> None:
        self.writer.dump_data()

    def finalize(
        self,
    ) -> None:
        self.writer.finalize()


class Reader:
    def __init__(
        self,
        index_file_path: str,
    ) -> None:
        self.reader = draftretriever.Reader(
            index_file_path=index_file_path,
        )

    def search(
        self,
        py_substring: typing.List,
        k: typing.Optional[int] = None,
        choices: typing.Optional[int] = None,
        long: typing.Optional[int] = None,
    ):
        return self.reader.search(
            py_substring=py_substring,
            k=k,
            choices=choices,
            long=long,
        )

