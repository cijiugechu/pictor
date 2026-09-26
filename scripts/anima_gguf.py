"""Strict metadata reader and streamed dequantizer for pictor's pinned P3 GGUF."""
import ctypes
import math
import struct
import numpy as np


class GGUF:
    def __init__(self, path, library):
        self.path = path
        self.file = open(path, "rb")
        self.lib = ctypes.CDLL(str(library))
        self.dequant = self.lib.dequantize_row_q4_K
        self.dequant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        self.dequant.restype = None
        f = self.file
        def read(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        def string():
            return f.read(read("Q")).decode("utf-8")
        def value(t):
            formats = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
            if t == 8:
                return string()
            if t == 9:
                et, n = read("I"), read("Q")
                return [value(et) for _ in range(n)]
            return read(formats[t])
        if f.read(4) != b"GGUF" or read("I") != 3:
            raise ValueError("expected GGUF v3")
        nt, nk = read("Q"), read("Q")
        self.metadata = {}
        for _ in range(nk):
            key = string()
            self.metadata[key] = value(read("I"))
        self.tensors = {}
        for _ in range(nt):
            name = string()
            shape = tuple(reversed([read("Q") for _ in range(read("I"))]))
            typ, offset = read("I"), read("Q")
            if typ not in (12, 30):
                raise ValueError(f"unsupported P3 tensor type {typ}: {name}")
            if name in self.tensors:
                raise ValueError(f"duplicate tensor {name}")
            self.tensors[name] = (shape, typ, offset)
        alignment = self.metadata.get("general.alignment", 32)
        self.start = (f.tell() + alignment - 1) // alignment * alignment

    def tensor(self, name):
        shape, typ, offset = self.tensors[name]
        n = math.prod(shape)
        self.file.seek(self.start + offset)
        if typ == 30:
            raw = np.frombuffer(self.file.read(n * 2), dtype="<u2")
            return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        if n % 256:
            raise ValueError("Q4_K element count must be divisible by 256")
        raw = self.file.read(n // 256 * 144)
        if len(raw) != n // 256 * 144:
            raise ValueError("truncated GGUF tensor")
        out = np.empty(n, dtype=np.float32)
        self.dequant(raw, out.ctypes.data, n)
        return out.reshape(shape)
