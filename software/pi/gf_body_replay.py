#!/usr/bin/env python3
"""Replicate the Fuji GFX body's power-on sequence to bring a G-mount lens
into the idle state, driven by a Raspberry Pi 5 as SPI master.

Phase 1 (startup): replays the body-side transactions captured in
startup_replay.json (exported from power_on.txt by fuji_spi.py) with the
original timing, and compares the lens's live responses against the capture.
Phase 2 (idle): synthesizes the body's steady-state polling loop — a 4-transaction
status burst every 40 ms, expanded to 6 transactions every third burst — with
packet tails (tag2 + check5) computed, not replayed, and the rolling transport
counter maintained.

Wiring (lens pad numbering per fuji-G-mount/electrical/README.md; all logic 3.3V):

    Lens Pin 5/6  -> Pi GND            (also common with bench supply grounds)
    Lens Pin 9    -> Pi GPIO10 (MOSI, phys pin 19)   body data out
    Lens Pin 10   -> Pi GPIO11 (SCLK, phys pin 23)   1.5 MHz clock, idle high
    Lens Pin 11   -> Pi GPIO9  (MISO, phys pin 21)   body data in
    CE0 (GPIO8)   -> leave unconnected (the mount has no chip select)

    Lens power (external bench supplies, NOT the Pi):
    Pin 2 = 5.3V, Pin 3 = 6.7V, Pin 4 = 8.0V (values measured on a GF45;
    a GF250 with OIS may draw substantially more current).
    The camera leaves the bus quiet for ~1.4s after power before the first
    packet; --settle reproduces that delay after this script starts.

Enable SPI on the Pi (dtparam=spi=on in /boot/firmware/config.txt) and install
python3-spidev.

Usage:
    python3 gf_body_replay.py                     # real hardware, 10s idle
    python3 gf_body_replay.py --idle-seconds 30
    python3 gf_body_replay.py --dry-run           # no hardware, echo expected rx
    python3 gf_body_replay.py --transcript run1.tsv   # Saleae-style TSV log,
                                                  # analyzable with fuji_spi.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SPI_SPEED_HZ = 1_500_000
SPI_MODE = 3  # CPOL=1, CPHA=1
INTRA_BURST_GAP_S = 0.0003   # gap between transactions within a burst
IDLE_PERIOD_S = 0.040        # status burst period
BYTE_TIME_S = 8 / SPI_SPEED_HZ


# ---------------------------------------------------------------------------
# Packet building (see protocol README: tail = tag2<<6 | check5<<1)
# ---------------------------------------------------------------------------

def check5(b0: int, b1: int, cmd: int, tag2: int) -> int:
    g0 = b0 >> 3
    g1 = ((b0 & 0x07) << 2) | (b1 >> 6)
    g2 = (b1 >> 1) & 0x1F
    g3 = ((b1 & 0x01) << 4) | (cmd >> 4)
    g4 = ((cmd & 0x0F) << 1) | ((tag2 >> 1) & 1)
    g5 = tag2 & 1
    return (g0 + g1 + g2 + g3 + g4 + g5) & 0x1F


def pkt(b0: int, b1: int, cmd: int, tag2: int = 0) -> bytes:
    return bytes([b0, b1, cmd, (tag2 << 6) | (check5(b0, b1, cmd, tag2) << 1)])


IDLE_PKT = bytes(4)
STATUS_POLL = pkt(0x00, 0x00, 0x08)          # 00 00 08 20
POLL_09 = pkt(0x00, 0x00, 0x09, tag2=2)      # 00 00 09 a6
ACK_08 = pkt(0x08, 0x00, 0x88)               # 08 00 88 32
# ACKs carry the tag2 of the packet being acknowledged; the 0x09 idle
# response is tag2=2, so its ACK is too.
ACK_89 = pkt(0x08, 0x00, 0x89, tag2=2)       # 08 00 89 b8


def transport(n: int) -> bytes:
    return pkt(n, 0x10, 0x80)                # n 10 80 xx


def ack_88(n: int) -> bytes:
    return pkt(n, 0x00, 0x88)                # n 00 88 xx


# ---------------------------------------------------------------------------
# SPI link (real spidev or dry-run)
# ---------------------------------------------------------------------------

class SpiLink:
    def __init__(self, bus: int, device: int, dry_run: bool):
        self.dry_run = dry_run
        self._dry_queue: list[bytes] = []
        if dry_run:
            self.spi = None
            return
        import spidev  # noqa: PLC0415 -- only needed on the Pi
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = SPI_SPEED_HZ
        self.spi.mode = SPI_MODE

    def queue_dry_response(self, rx: bytes) -> None:
        self._dry_queue.append(rx)

    def xfer(self, tx: bytes) -> bytes:
        if self.dry_run:
            time.sleep(len(tx) * BYTE_TIME_S)
            return self._dry_queue.pop(0) if self._dry_queue else bytes(len(tx))
        return bytes(self.spi.xfer2(list(tx)))

    def close(self) -> None:
        if self.spi:
            self.spi.close()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class BodySession:
    def __init__(self, link: SpiLink, transcript_path: Path | None):
        self.link = link
        self.t0 = time.perf_counter()
        self.counter = 8  # rolling transport counter, cycles 0x8..0xf
        self.transcript = None
        if transcript_path:
            self.transcript = transcript_path.open("w")
            self.transcript.write("name\ttype\tstart_time\tduration\tmosi\tmiso\n")

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def wait_until(self, t: float) -> None:
        # sleep for the bulk, busy-wait the last 500us for accuracy
        while True:
            dt = t - self.now()
            if dt <= 0:
                return
            if dt > 0.0015:
                time.sleep(dt - 0.001)
            elif dt > 0.0005:
                time.sleep(0)
            # else spin

    def next_counter(self) -> int:
        n = self.counter
        self.counter = 8 + ((self.counter + 1) & 0x07)
        return n

    def xfer(self, tx: bytes) -> bytes:
        t = self.now()
        rx = self.link.xfer(tx)
        if self.transcript:
            # one row per byte, same shape as a Saleae export so the
            # fuji_spi.py toolkit can analyze the transcript directly
            for i, (m, s) in enumerate(zip(tx, rx)):
                self.transcript.write(
                    f"SPI [1]\tresult\t{t + i * 5.28e-6:.8f}\t0.00000512"
                    f"\t0x{m:02X}\t0x{s:02X}\n"
                )
        return rx

    def close(self) -> None:
        if self.transcript:
            self.transcript.close()
        self.link.close()


# ---------------------------------------------------------------------------
# Phase 1: startup replay
# ---------------------------------------------------------------------------

def decode_ident(rx: bytes) -> str | None:
    """Pull the ASCII identity out of a 131-byte identification block."""
    if len(rx) < 131 or not any(rx[2:]):
        return None
    text = "".join(chr(c) if 32 <= c < 127 else " " for c in rx[2:0x52])
    return " ".join(text.split()) or None


def run_startup(sess: BodySession, replay: list[dict], settle: float) -> bool:
    print(f"settling {settle:.1f}s (bus quiet time after lens power-on)...")
    time.sleep(settle)
    sess.t0 = time.perf_counter()

    matches = mismatches = 0
    lens_identity = None
    for i, entry in enumerate(replay):
        tx = bytes.fromhex(entry["tx"])
        expected = bytes.fromhex(entry["rx_expected"])
        if sess.link.dry_run:
            sess.link.queue_dry_response(expected)
        sess.wait_until(entry["t"])
        rx = sess.xfer(tx)

        if len(tx) > 4:
            ident = decode_ident(rx)
            if ident and lens_identity is None:
                lens_identity = ident
                print(f"  [{i}] lens identification: {ident}")
            continue

        if rx == expected:
            matches += 1
        else:
            mismatches += 1
            # differing payloads are expected where lens state differs from
            # the capture (ring positions, focus position); log for review
            print(f"  [{i}] t={entry['t']:.4f} tx {tx.hex(' ')} -> "
                  f"rx {rx.hex(' ')} (capture had {expected.hex(' ')})")

    print(f"startup replay done: {matches} responses matched capture, "
          f"{mismatches} differed")
    if lens_identity is None and not sess.link.dry_run:
        print("WARNING: no lens identification received — check wiring/power")
    return lens_identity is not None or sess.link.dry_run


# ---------------------------------------------------------------------------
# Phase 2: synthesized idle loop
# ---------------------------------------------------------------------------

def describe_status(rx: bytes) -> str | None:
    """Decode a lens tag-0 0x08 status response (non-ack)."""
    if len(rx) != 4 or (rx[2] & 0x7F) != 0x08 or (rx[2] & 0x80):
        return None
    pend = rx[1] & 0x7F
    names = []
    if pend & 0x08:
        names.append("focus-ring")
    if pend & 0x10:
        names.append("aperture-ring")
    if pend & 0x02:
        names.append("detail/actuation")
    return f"b0={rx[0]:02x} b1={rx[1]:02x} pending={'+'.join(names) or 'none'}"


def run_idle(sess: BodySession, duration: float) -> None:
    print(f"entering idle loop for {duration:.0f}s "
          "(4-txn burst @40ms, 6-txn every third)...")
    start = sess.now()
    burst_i = 0
    last_status = None
    next_burst = start + IDLE_PERIOD_S

    while sess.now() - start < duration:
        sess.wait_until(next_burst)
        long_burst = burst_i % 3 == 0

        if long_burst:
            n, n2 = sess.next_counter(), sess.next_counter()
            burst = [STATUS_POLL, transport(n), POLL_09, ack_88(n2),
                     IDLE_PKT, ACK_89]
        else:
            n = sess.next_counter()
            burst = [STATUS_POLL, transport(n), IDLE_PKT, ACK_08]

        responses = []
        for j, tx in enumerate(burst):
            if j:
                time.sleep(INTRA_BURST_GAP_S)
            responses.append(sess.xfer(tx))

        for rx in responses:
            status = describe_status(rx)
            if status and status != last_status:
                print(f"  t={sess.now():8.3f} lens status changed: {status}")
                last_status = status

        burst_i += 1
        next_burst += IDLE_PERIOD_S

    print("idle loop complete")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", type=Path,
                    default=Path(__file__).parent / "startup_replay.json")
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--settle", type=float, default=1.4,
                    help="seconds of bus silence before first packet (default 1.4)")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--transcript", type=Path,
                    help="write Saleae-style TSV of the session (byte-accurate "
                         "tx/rx, analyzable with fuji_spi.py)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no SPI hardware; lens responses simulated from capture")
    args = ap.parse_args()

    replay = json.loads(args.replay.read_text())["transactions"]
    print(f"loaded {len(replay)} startup transactions from {args.replay}")

    link = SpiLink(args.bus, args.device, args.dry_run)
    sess = BodySession(link, args.transcript)
    try:
        ok = run_startup(sess, replay, args.settle)
        if not ok:
            print("aborting before idle loop (no lens response)")
            sys.exit(1)
        run_idle(sess, args.idle_seconds)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        sess.close()
        if args.transcript:
            print(f"transcript written to {args.transcript}")


if __name__ == "__main__":
    main()
