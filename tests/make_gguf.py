#!/usr/bin/env python3
"""Write a minimal GGUF header with chosen metadata, for the ai-auto tests.

    make_gguf.py OUT "key|type|value" ...      type: str, u32, f32, bool, nested (value = depth)

No tensors are written: ai-auto only reads the header.
"""
import struct
import sys

TYPES = {"u32": (4, "<I"), "f32": (6, "<f"), "bool": (7, "<?")}


def gguf_str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def main():
    out, items = sys.argv[1], sys.argv[2:]
    body = b""
    for item in items:
        key, typ, value = item.split("|", 2)
        key = key.encode().decode("unicode_escape")  # allows \n in test keys and values
        value = value.encode().decode("unicode_escape")
        body += gguf_str(key)
        if typ == "str":
            body += struct.pack("<I", 8) + gguf_str(value)
        elif typ == "nested":  # array of array of ... of an empty u32 array, `value` levels deep
            depth = int(value)
            body += struct.pack("<I", 9) + struct.pack("<IQ", 9, 1) * (depth - 1) + struct.pack("<IQ", 4, 0)
        else:
            code, fmt = TYPES[typ]
            v = {"u32": int, "f32": float, "bool": lambda x: x == "true"}[typ](value)
            body += struct.pack("<I", code) + struct.pack(fmt, v)
    with open(out, "wb") as f:
        f.write(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, len(items)) + body)


if __name__ == "__main__":
    main()
