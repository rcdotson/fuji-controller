#!/usr/bin/env python3
"""Toolkit for decoding Fuji G-mount body<->lens SPI captures.

Input format: Saleae Logic 2 SPI analyzer export (tab- or comma-separated)
with columns: name, type, start_time, duration, mosi, miso — one byte per row.

Protocol reference: fuji-G-mount/protocol/README.md
  - Most packets are 4 bytes on the wire: payload_hi payload_lo cmd tail
  - Payloads are typically big-endian signed 16-bit values or small indexes.
  - cmd bit7 is an ACK flag (cmd | 0x80).
  - tail byte: bits 6..7 = 'tag2' field, bits 1..5 = check5 (truncating 5-bit
    chunk sum over the packet), bit 0 always low.
  - Focus control ring deltas: lens->body (miso) cmd 0x0c with tag2 == 2,
    payload = signed BE16 relative ring movement (+CW / -CCW).
  - Aperture ring index: miso cmd 0x0c with tag2 == 0, payload = ordinal.

Usage:
  python fuji_spi.py summary <capture.txt>
  python fuji_spi.py dump    <capture.txt> [--cmd 0c] [--dir rx|tx]
  python fuji_spi.py focus   <capture.txt> [--csv out.csv] [--plot out.png]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Gap between consecutive byte start times that begins a new transaction.
# In-transaction byte spacing is ~5.3 us; inter-packet gaps are >200 us.
DEFAULT_GAP_S = 50e-6


# ---------------------------------------------------------------------------
# Parsing / framing
# ---------------------------------------------------------------------------

@dataclass
class SpiByte:
    t: float
    mosi: int
    miso: int


@dataclass
class Transaction:
    start: float
    bytes_: list[SpiByte] = field(default_factory=list)

    @property
    def mosi(self) -> bytes:
        return bytes(b.mosi for b in self.bytes_)

    @property
    def miso(self) -> bytes:
        return bytes(b.miso for b in self.bytes_)

    def __len__(self) -> int:
        return len(self.bytes_)


def parse_saleae(path: Path) -> list[SpiByte]:
    """Parse a Saleae SPI export. Autodetects tab/comma delimiter."""
    out: list[SpiByte] = []
    with path.open(newline="") as f:
        sample = f.readline()
        delim = "\t" if "\t" in sample else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        try:
            t_col = cols["start_time"]
            mosi_col = cols["mosi"]
            miso_col = cols["miso"]
        except KeyError as e:
            raise SystemExit(f"missing expected column {e} in {path}")
        for row in reader:
            try:
                out.append(
                    SpiByte(
                        t=float(row[t_col]),
                        mosi=int(row[mosi_col], 16),
                        miso=int(row[miso_col], 16),
                    )
                )
            except (ValueError, TypeError):
                continue  # skip malformed / non-result rows
    return out


def frame_transactions(raw: list[SpiByte], gap_s: float = DEFAULT_GAP_S) -> list[Transaction]:
    """Group the byte stream into transactions using inter-byte time gaps."""
    txns: list[Transaction] = []
    for b in raw:
        if not txns or (b.t - txns[-1].bytes_[-1].t) > gap_s:
            txns.append(Transaction(start=b.t))
        txns[-1].bytes_.append(b)
    return txns


# ---------------------------------------------------------------------------
# Packet decode
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    t: float
    direction: str  # 'tx' = body->lens (mosi), 'rx' = lens->body (miso)
    raw: bytes

    @property
    def b0(self) -> int:
        return self.raw[0]

    @property
    def b1(self) -> int:
        return self.raw[1]

    @property
    def cmd(self) -> int:
        return self.raw[2]

    @property
    def tail(self) -> int:
        return self.raw[3]

    @property
    def cmd_base(self) -> int:
        return self.cmd & 0x7F

    @property
    def is_ack(self) -> bool:
        return bool(self.cmd & 0x80)

    @property
    def tag2(self) -> int:
        return self.tail >> 6

    @property
    def check5(self) -> int:
        return (self.tail >> 1) & 0x1F

    @property
    def payload_be16(self) -> int:
        return (self.b0 << 8) | self.b1

    @property
    def payload_s16(self) -> int:
        v = self.payload_be16
        return v - 0x10000 if v >= 0x8000 else v

    @property
    def is_idle(self) -> bool:
        return self.raw == b"\x00\x00\x00\x00"

    def expected_check5(self) -> int:
        b0, b1, cmd, tag2 = self.b0, self.b1, self.cmd, self.tag2
        g0 = b0 >> 3
        g1 = ((b0 & 0x07) << 2) | (b1 >> 6)
        g2 = (b1 >> 1) & 0x1F
        g3 = ((b1 & 0x01) << 4) | (cmd >> 4)
        g4 = ((cmd & 0x0F) << 1) | ((tag2 >> 1) & 1)
        g5 = tag2 & 1
        return (g0 + g1 + g2 + g3 + g4 + g5) & 0x1F

    @property
    def check_ok(self) -> bool:
        return self.check5 == self.expected_check5()

    def hex(self) -> str:
        return " ".join(f"{b:02x}" for b in self.raw)

    def __str__(self) -> str:
        flags = []
        if self.is_ack:
            flags.append("ack")
        if not self.check_ok:
            flags.append("BADCK")
        return (
            f"{self.t:12.6f} {self.direction} {self.hex()}  "
            f"cmd={self.cmd_base:02x} tag={self.tag2} "
            f"val={self.payload_s16:6d} {' '.join(flags)}"
        )


def extract_packets(txns: list[Transaction]) -> list[Packet]:
    """Split each transaction into tx (mosi) and rx (miso) 4-byte packets.

    Idle fills (00 00 00 00) are kept — callers filter with .is_idle.
    Non-4-byte transactions (e.g. 131-byte identification blocks) are
    emitted whole so nothing is silently dropped.
    """
    pkts: list[Packet] = []
    for txn in txns:
        pkts.append(Packet(t=txn.start, direction="tx", raw=txn.mosi))
        pkts.append(Packet(t=txn.start, direction="rx", raw=txn.miso))
    return pkts


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(p: Packet) -> str:
    """Best-effort human label for a 4-byte packet, per protocol README."""
    if p.is_idle:
        return "idle/readout clock"
    c, tag = p.cmd_base, p.tag2
    if p.cmd == 0x80:
        return f"transport/status (n={p.b0 & 0x0f})"
    if p.is_ack:
        return f"ACK 0x{c:02x} (n={p.b0 & 0x0f})"
    if c == 0x05:
        return f"startup handshake 0x05 tag{tag}"
    if c == 0x06:
        return "startup handshake 0x06"
    if c == 0x08:
        if tag == 0:
            if p.raw[:3] == b"\x00\x00\x08":
                return "status poll"
            pend = p.b1 & 0x7F
            names = []
            if pend & 0x08:
                names.append("focus-ring")
            if pend & 0x10:
                names.append("aperture-ring")
            return f"lens status b0={p.b0:02x} pending={'+'.join(names) or 'none'}"
        if tag == 1:
            if p.direction == "tx":
                return "focus position feedback request"
            return f"focus position feedback = {p.payload_s16}"
        if tag == 2:
            if p.direction == "tx":
                return "aperture feedback request"
            return f"aperture feedback idx={p.b0 & 0x1f} flags={p.b0 >> 5:03b} aux={p.b1:02x}"
    if c == 0x09:
        return f"0x09 tag{tag} val={p.payload_s16}"
    if c == 0x0C:
        if tag == 0:
            return "aperture ring poll" if p.payload_be16 == 0 else f"aperture ring index = {p.b1}"
        if tag == 1:
            return "ring-mode poll?" if p.payload_be16 == 0 else f"ring-mode? = {p.b1:02x}"
        if tag == 2:
            return "focus ring poll" if p.payload_be16 == 0 else f"focus ring delta = {p.payload_s16}"
    if c == 0x0F:
        return "0x0f poll (undecoded)"
    if c == 0x15:
        kind = {0: "envelope/budget", 1: "speed/mode", 2: "target position"}[tag] if tag < 3 else "?"
        return f"focus motor {kind} = {p.payload_s16}"
    if c == 0x18:
        return f"aperture setpoint idx={(p.payload_be16 - 0x1000) // 0x40 + 1}"
    if c == 0x20:
        return "OIS command"
    if c == 0x28:
        return f"0x28 transition ({p.hex()[:5]})"
    if c == 0x2A:
        return f"focus-mode-context 0x2a payload={p.payload_be16:04x}"
    if c == 0x3F:
        return "EXECUTE/latch"
    return f"0x{c:02x} tag{tag} (unknown)"


def block_check8(data: bytes) -> int:
    """Trailer checksum for long (e.g. 131-byte) blocks: 8-bit sum of all
    bytes except the trailer itself. Verified on power-on ident blocks."""
    return sum(data[:-1]) & 0xFF


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def load(path: Path, gap_s: float) -> tuple[list[Transaction], list[Packet]]:
    raw = parse_saleae(path)
    txns = frame_transactions(raw, gap_s)
    pkts = extract_packets(txns)
    return txns, pkts


def cmd_summary(txns: list[Transaction], pkts: list[Packet]) -> None:
    sizes = Counter(len(t) for t in txns)
    print(f"transactions: {len(txns)}")
    print(f"sizes: {dict(sorted(sizes.items()))}")
    p4 = [p for p in pkts if len(p.raw) == 4 and not p.is_idle]
    bad = [p for p in p4 if not p.check_ok]
    print(f"non-idle 4-byte packets: {len(p4)}  check5 failures: {len(bad)}")
    for p in bad[:10]:
        print(f"  BADCK {p}")
    print()
    counts: Counter[tuple[str, int, int]] = Counter()
    for p in p4:
        counts[(p.direction, p.cmd_base, p.tag2)] += 1
    print(f"{'dir':3} {'cmd':>4} {'ack':>4} {'tag':>3} {'count':>6}")
    for (d, c, tag), n in sorted(counts.items()):
        acks = sum(1 for p in p4 if p.direction == d and p.cmd_base == c and p.tag2 == tag and p.is_ack)
        print(f"{d:3} {c:#04x} {acks:>4} {tag:>3} {n:>6}")


def cmd_dump(pkts: list[Packet], cmd: int | None, direction: str | None) -> None:
    for p in pkts:
        if len(p.raw) != 4 or p.is_idle:
            continue
        if cmd is not None and p.cmd_base != cmd:
            continue
        if direction and p.direction != direction:
            continue
        print(p)


def cmd_timeline(txns: list[Transaction], hide_idle: bool) -> None:
    """Annotated transaction-by-transaction listing."""
    prev = None
    for i, txn in enumerate(txns):
        gap = f"+{(txn.start - prev) * 1000:8.2f}ms" if prev is not None else " " * 11
        prev = txn.start
        if len(txn) != 4:
            m, s = txn.mosi, txn.miso
            print(f"[{i:4d}] {txn.start:9.6f} {gap}  LONG {len(txn)}B block")
            for name, d in (("tx", m), ("rx", s)):
                asc = "".join(chr(c) if 32 <= c < 127 else "." for c in d)
                ck = block_check8(d)
                ok = "sum8 OK" if ck == d[-1] else f"trailer {d[-1]:02x} (fixed-ack?)"
                print(f"        {name}: idx={d[:2].hex()} {ok}")
                print(f"        {name} ascii: {asc.strip('.')[:100]}")
            continue
        tx = Packet(t=txn.start, direction="tx", raw=txn.mosi)
        rx = Packet(t=txn.start, direction="rx", raw=txn.miso)
        if hide_idle and tx.is_idle and rx.is_idle:
            continue
        txs = "--         " if tx.is_idle else tx.hex()
        rxs = "--         " if rx.is_idle else rx.hex()
        print(f"[{i:4d}] {txn.start:9.6f} {gap}  tx {txs}  rx {rxs}   "
              f"tx: {annotate(tx):<38} rx: {annotate(rx)}")


def cmd_export_replay(txns: list[Transaction], out: Path, end_index: int | None) -> None:
    """Export body-side transactions (with expected lens responses) as JSON
    for replay on the Raspberry Pi (see gf_body_replay.py)."""
    import json

    sel = txns[:end_index] if end_index else txns
    t0 = sel[0].start
    entries = [
        {
            "t": round(txn.start - t0, 6),
            "tx": txn.mosi.hex(),
            "rx_expected": txn.miso.hex(),
        }
        for txn in sel
    ]
    out.write_text(json.dumps({
        "source": "power-on capture replay",
        "transactions": entries,
    }, indent=1))
    print(f"wrote {len(entries)} transactions to {out} "
          f"(span {sel[-1].start - t0:.3f}s)")


def cmd_focus(pkts: list[Packet], csv_out: Path | None, plot_out: Path | None) -> None:
    """Extract focus-ring delta packets and integrate to a position trace."""
    deltas = [
        p for p in pkts
        if len(p.raw) == 4
        and p.direction == "rx"
        and p.cmd_base == 0x0C
        and not p.is_ack
        and p.tag2 == 2
    ]
    if not deltas:
        print("no focus ring packets (rx cmd 0x0c tag2=2) found")
        return

    rows = []
    pos = 0
    for p in deltas:
        pos += p.payload_s16
        rows.append((p.t, p.payload_s16, pos))

    total_cw = sum(d for _, d, _ in rows if d > 0)
    total_ccw = sum(d for _, d, _ in rows if d < 0)
    positions = [r[2] for r in rows]
    print(f"focus ring delta packets: {len(rows)}")
    print(f"time span: {rows[0][0]:.3f}s .. {rows[-1][0]:.3f}s")
    print(f"delta range: {min(d for _, d, _ in rows)} .. {max(d for _, d, _ in rows)}")
    print(f"total CW counts: +{total_cw}   total CCW counts: {total_ccw}   net: {positions[-1]}")
    print(f"integrated position range: {min(positions)} .. {max(positions)}")

    # direction reversals in the integrated trace
    reversals = 0
    last_sign = 0
    for _, d, _ in rows:
        s = (d > 0) - (d < 0)
        if s and last_sign and s != last_sign:
            reversals += 1
        if s:
            last_sign = s
    print(f"direction reversals: {reversals}")

    print("\nfirst 20 packets:")
    for p in deltas[:20]:
        print(f"  {p}")

    if csv_out:
        with csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "delta", "position"])
            w.writerows(rows)
        print(f"\nwrote {csv_out}")

    if plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r[0] for r in rows]
        ds = [r[1] for r in rows]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
        ax1.step(ts, positions, where="post", lw=1)
        ax1.set_ylabel("integrated position (counts)")
        ax1.set_title("Focus ring — integrated position and per-packet delta")
        ax1.grid(alpha=0.3)
        ax2.plot(ts, ds, ".", ms=3)
        ax2.set_ylabel("delta per packet (counts)")
        ax2.set_xlabel("time (s)")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_out, dpi=120)
        print(f"wrote {plot_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP_S,
                    help="transaction framing gap in seconds (default 50us)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary", help="transaction/packet statistics")
    p_sum.add_argument("capture", type=Path)

    p_dump = sub.add_parser("dump", help="print decoded packets")
    p_dump.add_argument("capture", type=Path)
    p_dump.add_argument("--cmd", type=lambda s: int(s, 16), help="filter by base command (hex)")
    p_dump.add_argument("--dir", choices=["tx", "rx"], help="filter by direction")

    p_tl = sub.add_parser("timeline", help="annotated transaction listing")
    p_tl.add_argument("capture", type=Path)
    p_tl.add_argument("--hide-idle", action="store_true", help="skip all-idle transactions")

    p_foc = sub.add_parser("focus", help="extract focus ring deltas + position trace")
    p_foc.add_argument("capture", type=Path)
    p_foc.add_argument("--csv", type=Path, help="write time/delta/position CSV")
    p_foc.add_argument("--plot", type=Path, help="write PNG plot")

    p_exp = sub.add_parser("export-replay", help="export tx sequence as JSON for Pi replay")
    p_exp.add_argument("capture", type=Path)
    p_exp.add_argument("-o", "--out", type=Path, required=True)
    p_exp.add_argument("--end-index", type=int, help="stop before this transaction index")

    args = ap.parse_args(argv)
    txns, pkts = load(args.capture, args.gap)

    if args.command == "summary":
        cmd_summary(txns, pkts)
    elif args.command == "timeline":
        cmd_timeline(txns, args.hide_idle)
    elif args.command == "dump":
        cmd_dump(pkts, args.cmd, args.dir)
    elif args.command == "focus":
        cmd_focus(pkts, args.csv, args.plot)
    elif args.command == "export-replay":
        cmd_export_replay(txns, args.out, args.end_index)


if __name__ == "__main__":
    main()
