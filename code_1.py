

import argparse
import hashlib
import sys
import random
from typing import Tuple, List

from PIL import Image
import numpy as np


def derive_seed(key: str, nonce: int) -> int:
    """Derive a deterministic 64-bit seed from (key, nonce) using SHA-256."""
    h = hashlib.sha256()
    h.update(key.encode('utf-8'))
    h.update(str(nonce).encode('utf-8'))
    return int.from_bytes(h.digest()[:8], byteorder='big', signed=False)


def prng_bytes(length: int, key: str, nonce: int) -> np.ndarray:
    """Generate a deterministic keystream of bytes (0..255)."""
    seed = derive_seed(key, nonce)
    rng = random.Random(seed)
    out = np.frombuffer(bytes([rng.randrange(256) for _ in range(length)]), dtype=np.uint8)
    return out


def load_image_as_bytes(path: str) -> Tuple[np.ndarray, Tuple[int, int], str]:
    """Load image, convert to RGBA, return flat byte array + size + mode."""
    img = Image.open(path).convert('RGBA')
    arr = np.array(img, dtype=np.uint8)  # (H, W, 4)
    flat = arr.reshape(-1)               # flatten to 1D
    return flat, img.size, 'RGBA'


def save_bytes_as_image(flat: np.ndarray, size: Tuple[int, int], mode: str, path: str):
    """Save flat byte array back into an image file (PNG)."""
    w, h = size
    arr = flat.reshape((h, w, len(mode)))
    img = Image.fromarray(arr, mode=mode)
    img.save(path, format='PNG')


# --- Pixel operations ---

def op_xor(data: np.ndarray, key: str, nonce: int) -> np.ndarray:
    ks = prng_bytes(len(data), key, nonce)
    return np.bitwise_xor(data, ks, dtype=np.uint8)


def op_add(data: np.ndarray, key: str, nonce: int, invert: bool = False) -> np.ndarray:
    ks = prng_bytes(len(data), key, nonce)
    return (data - ks).astype(np.uint8) if invert else (data + ks).astype(np.uint8)


def op_swap(data: np.ndarray, key: str, nonce: int, invert: bool = False) -> np.ndarray:
    n = len(data)
    seed = derive_seed(key, nonce)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    perm = np.array(perm, dtype=np.int64)

    if not invert:
        return data[perm]
    else:
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(n, dtype=np.int64)
        return data[inv_perm]


# --- Encryption / Decryption ---

def encrypt_bytes(data: np.ndarray, key: str, nonce: int, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == 'xor':
        return op_xor(data, key, nonce)
    elif mode == 'add':
        return op_add(data, key, nonce, invert=False)
    elif mode == 'swap':
        return op_swap(data, key, nonce, invert=False)
    elif mode == 'mix':
        d1 = op_swap(data, key, nonce + 1000, invert=False)
        d2 = op_xor(d1, key, nonce + 2000)
        d3 = op_add(d2, key, nonce + 3000, invert=False)
        return d3
    else:
        raise ValueError(f"Unknown mode: {mode}")


def decrypt_bytes(data: np.ndarray, key: str, nonce: int, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == 'xor':
        return op_xor(data, key, nonce)  # symmetric
    elif mode == 'add':
        return op_add(data, key, nonce, invert=True)
    elif mode == 'swap':
        return op_swap(data, key, nonce, invert=True)
    elif mode == 'mix':
        d1 = op_add(data, key, nonce + 3000, invert=True)
        d2 = op_xor(d1, key, nonce + 2000)
        d3 = op_swap(d2, key, nonce + 1000, invert=True)
        return d3
    else:
        raise ValueError(f"Unknown mode: {mode}")


# --- CLI ---

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Image Encryptor/Decryptor")
    sub = parser.add_subparsers(dest='command', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('input', help='Input image path')
    common.add_argument('output', help='Output image path (PNG recommended)')
    common.add_argument('--mode', choices=['xor', 'add', 'swap', 'mix'],
                        default='mix', help='Operation mode')
    common.add_argument('--key', required=True, help='Secret key (string)')
    common.add_argument('--nonce', type=int, default=0, help='Nonce/IV (integer)')

    sub.add_parser('encrypt', parents=[common], help='Encrypt an image')
    sub.add_parser('decrypt', parents=[common], help='Decrypt an image')

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    data, size, mode = load_image_as_bytes(args.input)

    if args.command == 'encrypt':
        out = encrypt_bytes(data, args.key, args.nonce, args.mode)
    else:
        out = decrypt_bytes(data, args.key, args.nonce, args.mode)

    save_bytes_as_image(out, size, mode, args.output)
    print(f"{args.command.title()}ed '{args.input}' -> '{args.output}' using mode={args.mode}, nonce={args.nonce}.")
    return 0


if _name_ == '_main_':
    sys.exit(main(sys.argv[1:]))