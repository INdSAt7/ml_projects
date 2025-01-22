import bz2
import pickle
import lzma
import gzip
from typing import Any

def read_from_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_to_pickle(data: Any, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def read_from_pickle_lzma(file_path: str) -> Any:
    with lzma.open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_to_pickle_lzma(data: Any, file_path: str) -> None:
    with lzma.open(file_path, 'wb') as file:
        pickle.dump(data, file)

def read_from_pickle_gzip(file_path: str) -> Any:
    with gzip.open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_to_pickle_gzip(data: Any, file_path: str) -> None:
    with gzip.open(file_path, 'wb') as file:
        pickle.dump(data, file)

def read_from_pickle_bz(file_path: str) -> Any:
    with bz2.BZ2File(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_to_pickle_bz(data: Any, file_path: str) -> None:
    with bz2.BZ2File(file_path, 'wb') as file:
        pickle.dump(data, file)

def write_to_pickle_bytes(data: Any) -> bytes:
    return pickle.dumps(data)

def read_from_pickle_bytes(pickle_bytes: bytes) -> Any:
    return pickle.loads(pickle_bytes)

def write_to_lzma_bytes(data: Any) -> bytes:
    return lzma.compress(pickle.dumps(data))

def read_from_lzma_bytes(xz_bytes: bytes) -> Any:
    return pickle.loads(lzma.decompress(xz_bytes))

def write_to_gzip_bytes(data: Any) -> bytes:
    return gzip.compress(pickle.dumps(data))

def read_from_gzip_bytes(xz_bytes: bytes) -> Any:
    return pickle.loads(gzip.decompress(xz_bytes))

def write_to_bz2_bytes(data: Any) -> bytes:
    return bz2.compress(pickle.dumps(data))

def read_from_bz2_bytes(xz_bytes: bytes) -> Any:
    return pickle.loads(bz2.decompress(xz_bytes))

