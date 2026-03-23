#!/usr/bin/env python3
"""
Phase 1: Fast Bitcoin Block Header Extraction via Electrum Protocol

Uses the Electrum protocol (blockchain.block.headers) to fetch up to 2016
headers per request. At ~445 requests for 890K headers, this completes
in minutes rather than hours.

Connects to public Electrum servers via SSL.
"""

import hashlib
import struct
import socket
import ssl
import json
import time
import sys
import os
import signal
from pathlib import Path

import numpy as np

# ─── Globals ───
SHUTDOWN = False
def handle_signal(signum, frame):
    global SHUTDOWN
    SHUTDOWN = True
    print("\nShutdown signal received...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ─── Electrum JSON-RPC over SSL ───

class ElectrumClient:
    """Simple Electrum protocol client over SSL TCP."""

    SERVERS = [
        ("electrum.blockstream.info", 50002),
        ("electrum1.bluewallet.io", 443),
        ("bolt.schulzemic.net", 50002),
        ("electrum.emzy.de", 50002),
    ]

    def __init__(self):
        self.sock = None
        self.ssl_sock = None
        self.rfile = None
        self.msg_id = 0

    def connect(self, server=None, port=None):
        """Connect to an Electrum server."""
        servers_to_try = [(server, port)] if server else self.SERVERS

        for host, p in servers_to_try:
            try:
                print(f"  Connecting to {host}:{p}...")
                self.sock = socket.create_connection((host, p), timeout=15)
                ctx = ssl.create_default_context()
                self.ssl_sock = ctx.wrap_socket(self.sock, server_hostname=host)
                self.rfile = self.ssl_sock.makefile('rb')
                # Test connection with server.version
                resp = self.call("server.version", ["SHA256-ML-Redux", "1.4"])
                print(f"  Connected: {resp}")
                return True
            except Exception as e:
                print(f"  Failed {host}:{p}: {e}")
                self.close()
                continue
        return False

    def close(self):
        try:
            if self.rfile:
                self.rfile.close()
            if self.ssl_sock:
                self.ssl_sock.close()
            if self.sock:
                self.sock.close()
        except:
            pass
        self.sock = None
        self.ssl_sock = None
        self.rfile = None

    def call(self, method, params=None):
        """Send JSON-RPC request and read response."""
        self.msg_id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self.msg_id,
            "method": method,
            "params": params or [],
        }
        msg = json.dumps(req) + "\n"
        self.ssl_sock.sendall(msg.encode())

        # Read response line
        line = self.rfile.readline()
        if not line:
            raise ConnectionError("Server closed connection")
        resp = json.loads(line)

        if "error" in resp and resp["error"]:
            raise Exception(f"Electrum error: {resp['error']}")

        return resp.get("result")

    def get_headers(self, start_height, count=2016):
        """
        Fetch block headers using blockchain.block.headers.
        Returns hex string of concatenated 80-byte headers.
        """
        result = self.call("blockchain.block.headers", [start_height, count])
        return result


# ─── Hash/format utilities ───

def sha256d(data: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def count_leading_zero_bits(data: bytes) -> int:
    count = 0
    for byte in data:
        if byte == 0:
            count += 8
        else:
            count += (8 - byte.bit_length())
            break
    return count

def sha256_pad_80bytes(header: bytes) -> bytes:
    assert len(header) == 80
    padded = bytearray(header)
    padded.append(0x80)
    padded.extend(b'\x00' * 39)
    padded += struct.pack('>Q', 640)
    return bytes(padded)

def header_to_training_bits(header_80: bytes) -> np.ndarray:
    padded = sha256_pad_80bytes(header_80)
    all_bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8))
    NONCE_START, NONCE_END = 608, 640
    non_nonce = np.concatenate([all_bits[:NONCE_START], all_bits[NONCE_END:]])
    nonce_bits = all_bits[NONCE_START:NONCE_END]
    return np.concatenate([non_nonce, nonce_bits]).astype(np.uint8)


# ─── Status file ───
def write_status(sandbox, phase, message, progress=None):
    status = {
        "phase": phase,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "progress": progress,
    }
    with open(Path(sandbox) / "status.json", 'w') as f:
        json.dump(status, f, indent=2)


# ─── Main extraction ───

def extract_all_headers(sandbox):
    """Download all Bitcoin block headers via Electrum protocol."""
    global SHUTDOWN
    sandbox = Path(sandbox)
    data_dir = sandbox / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = data_dir / "download_checkpoint.json"

    # Resume from checkpoint
    start_height = 0
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            cp = json.load(f)
        start_height = cp.get('next_height', 0)
        print(f"Resuming from height {start_height}")

    # Load existing data
    dataset_file = data_dir / "dataset_real_bitcoin.npy"
    if dataset_file.exists() and start_height > 0:
        existing = np.load(dataset_file)
        all_bits = list(existing)
        print(f"Loaded {len(existing)} existing examples")
    else:
        all_bits = []

    # Connect to Electrum server
    client = ElectrumClient()
    if not client.connect():
        print("ERROR: Could not connect to any Electrum server")
        return 0

    # Get current tip height
    try:
        tip = client.call("blockchain.headers.subscribe")
        current_height = tip['height']
        print(f"Current blockchain height: {current_height}")
    except Exception as e:
        print(f"Could not get tip: {e}")
        current_height = 890000

    total_needed = current_height - start_height + 1
    print(f"Downloading headers {start_height} to {current_height} ({total_needed} blocks)")

    BATCH = 2016  # Max headers per Electrum request
    SAVE_EVERY = 50000
    t0 = time.time()
    downloaded = 0
    height = start_height
    errors = 0

    while height <= current_height and not SHUTDOWN:
        count = min(BATCH, current_height - height + 1)

        try:
            result = client.get_headers(height, count)
            hex_data = result['hex']
            num_returned = result['count']

            # Parse concatenated 80-byte headers
            raw = bytes.fromhex(hex_data)
            for i in range(num_returned):
                header = raw[i*80:(i+1)*80]
                assert len(header) == 80

                # Verify hash
                block_hash = sha256d(header)
                lz = count_leading_zero_bits(block_hash)

                # Convert to training format
                training = header_to_training_bits(header)
                all_bits.append(training)

            downloaded += num_returned
            height += num_returned
            errors = 0  # Reset error counter on success

        except Exception as e:
            errors += 1
            print(f"  Error at height {height}: {e}")
            if errors >= 5:
                print("  Too many errors, reconnecting...")
                client.close()
                time.sleep(2)
                if not client.connect():
                    print("  Reconnection failed, saving and exiting")
                    break
                errors = 0
            else:
                time.sleep(1)
            continue

        # Progress
        elapsed = time.time() - t0
        rate = downloaded / max(elapsed, 0.1)
        remaining = current_height - height + 1
        eta_s = remaining / max(rate, 0.01)
        eta_str = f"{eta_s/60:.1f}m" if eta_s < 3600 else f"{eta_s/3600:.1f}h"

        print(f"  Height {height}: {len(all_bits)} total, "
              f"{rate:.0f} hdr/s, ETA {eta_str}")

        write_status(sandbox, "phase1",
                     f"Height {height}/{current_height}",
                     progress=f"{len(all_bits)} headers, {rate:.0f} hdr/s, ETA {eta_str}")

        # Save periodically
        if downloaded % SAVE_EVERY < BATCH:
            save_data(data_dir, checkpoint_file, all_bits, height)

        # Check shutdown signal
        if (sandbox / "shutdown.signal").exists():
            print("Shutdown signal detected")
            SHUTDOWN = True

    # Final save
    save_data(data_dir, checkpoint_file, all_bits, height)
    client.close()

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_bits)} headers in {elapsed:.0f}s ({downloaded/max(elapsed,1):.0f} hdr/s)")
    write_status(sandbox, "phase1",
                 f"Complete: {len(all_bits)} headers",
                 progress=f"{len(all_bits)}/{len(all_bits)}")
    return len(all_bits)


def save_data(data_dir, checkpoint_file, all_bits, next_height):
    if all_bits:
        dataset = np.stack(all_bits, axis=0).astype(np.uint8)
        np.save(data_dir / "dataset_real_bitcoin.npy", dataset)
        print(f"  Saved {len(dataset)} examples to dataset_real_bitcoin.npy")
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'next_height': next_height,
            'total_examples': len(all_bits),
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)


# ─── Difficulty-stratified datasets ───

def generate_difficulty_datasets(sandbox, difficulties=None, samples_per=50000):
    """Re-mine real headers at specified difficulty levels."""
    global SHUTDOWN
    import random
    if difficulties is None:
        difficulties = [1, 2, 3, 4, 5, 6, 7, 8]

    sandbox = Path(sandbox)
    data_dir = sandbox / "data"
    dataset_file = data_dir / "dataset_real_bitcoin.npy"

    if not dataset_file.exists():
        print("No real Bitcoin dataset found. Run extraction first.")
        return

    real_data = np.load(dataset_file)
    print(f"Loaded {len(real_data)} real headers for re-mining")
    random.seed(42)

    for n_zeros in difficulties:
        if SHUTDOWN:
            break

        output_file = data_dir / f"dataset_n{n_zeros:02d}_real.npy"
        if output_file.exists():
            existing = np.load(output_file)
            print(f"  n={n_zeros}: {len(existing)} examples exist, skipping")
            continue

        target = min(samples_per, len(real_data))
        print(f"\n  Mining n={n_zeros} ({target} examples)...")
        write_status(sandbox, "phase1_stratified",
                     f"Mining n={n_zeros}", progress=f"0/{target}")

        examples = []
        t0 = time.time()
        indices = list(range(len(real_data)))
        random.shuffle(indices)

        for idx in indices:
            if len(examples) >= target or SHUTDOWN:
                break

            bits = real_data[idx]
            header_bits = np.concatenate([bits[:608], bits[992:1024]])
            header = np.packbits(header_bits).tobytes()
            stub = header[:76]

            start_nonce = random.randint(0, 2**32 - 1)
            max_tries = min(2**(n_zeros + 3), 2**20)
            for i in range(max_tries):
                nonce = (start_nonce + i) & 0xFFFFFFFF
                full_header = stub + struct.pack('<I', nonce)
                h = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                if count_leading_zero_bits(h) >= n_zeros:
                    training = header_to_training_bits(full_header)
                    examples.append(training)
                    break

            if len(examples) % 5000 == 0 and len(examples) > 0:
                elapsed = time.time() - t0
                rate = len(examples) / elapsed
                print(f"    {len(examples)}/{target} ({rate:.0f} ex/s)")

        if examples:
            dataset = np.stack(examples).astype(np.uint8)
            np.save(output_file, dataset)
            elapsed = time.time() - t0
            print(f"  n={n_zeros}: Saved {len(dataset)} examples ({elapsed:.1f}s)")

    write_status(sandbox, "phase1_stratified", "Complete")


# ─── Stats ───

def print_stats(sandbox):
    data_dir = Path(sandbox) / "data"
    dataset_file = data_dir / "dataset_real_bitcoin.npy"
    if not dataset_file.exists():
        print("No dataset found.")
        return

    data = np.load(dataset_file)
    print(f"\nReal Bitcoin headers: {len(data)} examples, shape {data.shape}")

    bit_means = data.mean(axis=0)
    fields = [
        (0, 32, "version"),
        (32, 288, "prev_hash"),
        (288, 544, "merkle_root"),
        (544, 576, "timestamp"),
        (576, 608, "bits/target"),
        (608, 992, "SHA-256 padding"),
        (992, 1024, "nonce"),
    ]
    print(f"\n  {'Field':<20} {'Bits':>6} {'Mean':>8}")
    print(f"  {'-'*38}")
    for start, end, name in fields:
        print(f"  {name:<20} {end-start:>6} {bit_means[start:end].mean():>8.4f}")

    nonce_means = bit_means[992:1024]
    print(f"\n  Nonce bits: mean={nonce_means.mean():.4f}, "
          f"range=[{nonce_means.min():.4f}, {nonce_means.max():.4f}]")

    print(f"\n  Difficulty-stratified datasets:")
    for n in range(1, 9):
        f = data_dir / f"dataset_n{n:02d}_real.npy"
        if f.exists():
            d = np.load(f)
            print(f"    n={n}: {len(d)} examples")
        else:
            print(f"    n={n}: not yet generated")


# ─── CLI ───

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Fast Bitcoin Header Extraction (Electrum)")
    parser.add_argument("--sandbox", default="/mnt/d/sha256-ml-redux")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.sandbox)
    elif args.stratified:
        generate_difficulty_datasets(args.sandbox)
    else:
        print("=" * 70)
        print("PHASE 1: BITCOIN HEADER EXTRACTION (Electrum Protocol)")
        print("Fetches up to 2016 headers per request — ~445 requests total")
        print("=" * 70)
        extract_all_headers(args.sandbox)
