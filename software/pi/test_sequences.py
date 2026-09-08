#!/usr/bin/env python3
"""Check the packet builders against real captured body traffic.

Pure logic — no SPI, no GPIO, no lens. Every expected byte string below was
lifted from a capture, so this is a regression test against the ground truth
rather than against our own encoder.

Sources:
  startup_shutdown.txt          GF250, OIS switch ON: startup, use, clean
                                shutdown.
  startup_shutdown_no_ois.txt   the same, OIS switch OFF.
  startup_shutdown_focus_full_range.txt   OIS off, AF range switch at full
                                range instead of 5m-infinity.
  focus_ring_back_forth.txt     deliberate focus-ring rotation.
  Transaction indexes are into the 50us-gap framing of those files.
Usage:
    python3 test_sequences.py
"""

from __future__ import annotations

import sys

import gf_controller as gf


class FakeSession:
    """Just enough BodySession to drive the sequence builders: the rolling
    transport counter, seeded wherever the capture happened to be."""

    def __init__(self, counter: int):
        self.counter = counter

    def next_counter(self) -> int:
        n = self.counter
        self.counter = 8 + ((self.counter + 1) & 0x07)
        return n


def hexes(seq) -> list[str]:
    return [p.hex(" ") for p in seq]


FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if got == want:
        print(f"  ok   {name}")
        return
    FAILURES.append(name)
    print(f"  FAIL {name}")
    for i in range(max(len(got), len(want))):
        g = str(got[i]) if i < len(got) else "-"
        w = str(want[i]) if i < len(want) else "-"
        print(f"         [{i}] got {g:<24} want {w:<24}"
              f"{'' if g == w else '   <<<'}")


# ---------------------------------------------------------------------------
# Shutdown phase A: OIS off (startup_shutdown.txt txn 585-590,
# t=4.9588). Counter is at 8 when the body starts the sequence.
# ---------------------------------------------------------------------------

def test_ois_off_sequence() -> None:
    sess = FakeSession(counter=8)
    seq = gf.staged_sequence(sess, [gf.OIS_OFF])
    check("phase A: 0x20 OIS off + execute", hexes(seq), [
        "00 00 20 04",   # staged: OIS off
        "08 10 80 22",   # transport(8)
        "00 00 3f c6",   # execute
        "09 00 a0 1e",   # ack of the 0x20 echo
        "00 00 00 00",
        "08 00 bf d8",   # ACK_BF
    ])


# ---------------------------------------------------------------------------
# Shutdown phase B: park (txn 599-606, t=4.9732). Counter is at 0x0c.
# ---------------------------------------------------------------------------

def test_park_sequence() -> None:
    sess = FakeSession(counter=0x0C)
    seq = gf.staged_sequence(sess, [gf.channel(gf.PARK_CHANNEL), gf.PARK])
    check("phase B: 0x28 channel 0x8001 + 0x10 park + execute", hexes(seq), [
        "80 01 28 24",   # staged: channel select 0x8001
        "0c 10 80 02",   # transport(0x0c)
        "00 01 10 22",   # staged: park
        "0d 00 a8 1e",   # ack of the 0x28 echo
        "00 00 3f c6",   # execute
        "0e 00 90 04",   # ack of the 0x10 echo
        "00 00 00 00",
        "08 00 bf d8",   # ACK_BF
    ])


# ---------------------------------------------------------------------------
# The same builder has to reproduce the body's four-stage focus drive
# (txn 117-126, t=1.9634, counter at 9) and two-stage iris write
# (txn 221-226, t=2.2674, counter at 0x0d). Both run past the execute ack
# into pipelined traffic, so only the sequence proper is compared.
# ---------------------------------------------------------------------------

def test_focus_drive_sequence() -> None:
    sess = FakeSession(counter=9)
    stages = [gf.channel(0x02),
              gf.pkt(0x7F, 0xFF, 0x15),             # envelope 32767
              gf.pkt(0x01, 0x03, 0x15, tag2=1),     # speed 259
              gf.pkt(0x00, 0xDF, 0x15, tag2=2)]     # target +223
    seq = gf.staged_sequence(sess, stages)
    check("body focus drive (4 stages)", hexes(seq)[:10], [
        "80 02 28 06", "09 10 80 2a", "7f ff 15 10", "0a 00 a8 06",
        "01 03 15 42", "0b 00 95 00", "00 df 15 9c", "0c 00 95 4a",
        "00 00 3f c6", "0d 00 95 92",
    ])


def test_iris_write_sequence() -> None:
    sess = FakeSession(counter=0x0D)
    # channel 0x8004 carries tag2=1 here, unlike the park's tag2=0
    stages = [gf.channel(0x04, tag2=1), gf.iris_setpoint(4)]
    seq = gf.staged_sequence(sess, stages)
    check("body iris write (2 stages)", hexes(seq)[:6], [
        "80 04 28 4a", "0d 10 80 0a", "10 c0 18 ae", "0e 00 a8 68",
        "00 00 3f c6", "0f 00 98 ae",
    ])


# ---------------------------------------------------------------------------
# The refactor must not change what command_iris/command_focus put on the
# wire: these are the byte strings the hand-rolled versions produced.
# ---------------------------------------------------------------------------

def test_command_iris_unchanged() -> None:
    sess = FakeSession(counter=8)
    seq = gf.staged_sequence(sess, [gf.iris_setpoint(4)])
    check("command_iris shape (regression)", hexes(seq), [
        "10 c0 18 ae", "08 10 80 22", "00 00 3f c6", "09 00 98 be",
        "00 00 00 00", "08 00 bf d8",
    ])


def test_command_focus_unchanged() -> None:
    sess = FakeSession(counter=8)
    envelope = gf.pkt(0x7F, 0xFF, 0x15)
    staged = gf.pkt(0x00, 0xDF, 0x15, tag2=2)
    seq = gf.staged_sequence(sess, [envelope, gf.FOCUS_SPEED_288, staged])
    check("command_focus shape (regression)", hexes(seq), [
        "7f ff 15 10", "08 10 80 22", "01 20 15 40", "09 00 95 30",
        "00 df 15 9c", "0a 00 95 7a", "00 00 3f c6", "0b 00 95 82",
        "00 00 00 00", "08 00 bf d8",
    ])


# ---------------------------------------------------------------------------
# Shutdown phase D: the state readout the body runs both after
# identification (txn 105-116) and just before cutting power (txn 649-660).
# ---------------------------------------------------------------------------

def test_state_readout_requests() -> None:
    check("phase D: state readout requests", hexes(gf.STATE_READOUT), [
        "00 00 0f c0",   # latch
        "00 01 09 04",   # 0x09 sub-read
        "00 01 08 42",   # focus position
        "00 01 08 82",   # iris state
        "00 00 09 e8",   # 0x09 tag-3 read
    ])


def test_ois_packets() -> None:
    """The one packet that differs between the two shutdown captures: the
    0x20 the body sends after identification, payload tracking the switch
    (t=2.0624 with OIS on, t=1.5544 with it off)."""
    check("0x20 OIS packets", [gf.OIS_ON.hex(" "), gf.OIS_OFF.hex(" ")],
          ["00 01 20 24", "00 00 20 04"])


# ---------------------------------------------------------------------------
# Status bits, read off the b1 walk through both shutdowns. The same lens
# with its OIS switch flipped shifts the whole walk c0/c3/c1 -> 80/83/81,
# so bit 6 is the OIS switch and the low bits are unaffected by it.
#
#   OIS on : c0 running (t=4.9436), c3 parking (t=4.9829), c1 parked (t=5.1029)
#   OIS off: 80 running (t=4.6076), 83 parking (t=4.6875), 81 parked (t=4.7675)
# ---------------------------------------------------------------------------

def test_status_bits() -> None:
    samples = [
        # (packet, ois, busy, not-ready)
        (bytes.fromhex("01c0082e"), True, False, False),   # ois on, running
        (bytes.fromhex("11c30814"), True, True, True),     # ois on, parking
        (bytes.fromhex("01c1080e"), True, False, True),    # ois on, parked
        (bytes.fromhex("0180082c"), False, False, False),  # ois off, running
        (bytes.fromhex("0183080e"), False, True, True),    # ois off, parking
        (bytes.fromhex("0181080c"), False, False, True),   # ois off, parked
    ]
    got = [(gf.is_status(p), gf.ois_active(p), gf.status_busy(p),
            gf.status_disabled(p)) for p, _, _, _ in samples]
    check("status bit decode", got,
          [(True, ois, busy, nr) for _, ois, busy, nr in samples])


# ---------------------------------------------------------------------------
# b0 bit 0 is the AF range switch. Flipping it to full range shifts every
# status packet 01/11 -> 00/10 and changes nothing else in the session
# (startup_shutdown_focus_full_range.txt vs startup_shutdown_no_ois.txt,
# same lens, same body, OIS off in both).
# ---------------------------------------------------------------------------

def test_af_range_bit() -> None:
    limited = [bytes.fromhex(h) for h in
               ("0180082c", "11800830", "0183080e", "11990828")]
    full = [bytes.fromhex(h) for h in
            ("00800824", "10800828", "00830806", "10990820")]
    got = [gf.af_range_limited(p) for p in limited + full]
    check("AF range switch bit", got, [True] * 4 + [False] * 4)


def test_detail_pending_bit() -> None:
    """b0 bit 4 is a separate pending flag from the b1 bit-3 ring flag: the
    body answers it with 00 00 0c 72, not the 0c b2 ring readout."""
    ring_turn = bytes.fromhex("01080830")     # focus_ring_back_forth.txt
    detail = bytes.fromhex("11c00832")        # startup_shutdown.txt t=4.0638
    got = [(gf.detail_pending(ring_turn), bool(ring_turn[1] & 0x08)),
           (gf.detail_pending(detail), bool(detail[1] & 0x08))]
    check("detail vs ring pending bits", got, [(False, True), (True, False)])


def test_readout_payload_decode() -> None:
    """The lens replies the body read at t=5.2018/5.2025 before power-off."""
    focus = bytes.fromhex("f74c0864")     # -2228, tag2=1
    iris = bytes.fromhex("000008a2")      # index 0, tag2=2
    got = (gf.valid_pkt(focus), focus[3] >> 6,
           int.from_bytes(focus[:2], "big", signed=True),
           gf.valid_pkt(iris), iris[3] >> 6, iris[0] & 0x1F)
    check("phase D payload decode", got, (True, 1, -2228, True, 2, 0))


def main() -> int:
    print("packet builders vs captured body traffic:")
    for fn in (test_ois_off_sequence, test_park_sequence,
               test_focus_drive_sequence, test_iris_write_sequence,
               test_command_iris_unchanged, test_command_focus_unchanged,
               test_state_readout_requests, test_ois_packets,
               test_status_bits, test_af_range_bit, test_detail_pending_bit,
               test_readout_payload_decode):
        fn()
    if FAILURES:
        print(f"\n{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("\nall sequences match the capture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
