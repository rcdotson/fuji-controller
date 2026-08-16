#!/usr/bin/env python3
"""Replicate the Fuji GFX body's power-on sequence to bring a G-mount lens
into the idle state, driven by a Raspberry Pi 5 as SPI master.

Phase 1 (startup): replays the deterministic prefix of the body-side
transactions captured in startup_replay.json (exported from power_on.txt by
fuji_spi.py) with the original timing, comparing live responses against the
capture. Only the first --replay-end transactions are replayed: the capture's
tail is state-dependent idle traffic, and canned ACKs there desync a live
lens (it then answers everything with the c3 3c a5 5a resync word).
Phase 2 (idle): synthesizes the body's polling loop adaptively — 40ms status
bursts (0x09 poll every third), ACKs computed from the lens's actual packets
(cmd | 0x80, matching tag2), and 0x0c follow-up polls when the lens status
flags a pending focus/aperture ring readout. A lens beaconing c3 3c a5 5a
has dropped to its bootloader (a desync does this); the recovery is a
quiet-bus reboot — SCLK low, ~1.4s silence, startup prefix again. (The
fw-update magic handshake is NOT a recovery: the lens acks it but that is
the entry into firmware-update mode; see attempt_resync.)

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
    python3 gf_body_replay.py --startup-retries 5   # more tries at identification
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
REBOOT_QUIET_S = 0         # bus quiet for a beaconing lens to boot its app
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
FOCUS_POLL = pkt(0x00, 0x00, 0x0C, tag2=2)   # 00 00 0c b2
APERTURE_POLL = pkt(0x00, 0x00, 0x0C)        # 00 00 0c 30
# Bootloader beacon: after a reset or transport desync the lens drops to its
# bootloader and answers every transaction with this word (also the fw-update
# handshake word). Not a valid check5 packet, so it never enters the ACK path.
MAGIC_WORD = bytes.fromhex("c33ca55a")
MAGIC_REPLY = bytes.fromhex("a55a3cc3")      # body's side of the handshake
PKT_2824 = pkt(0x80, 0x20, 0x28)             # 80 20 28 24, follows the magic


def transport(n: int) -> bytes:
    return pkt(n, 0x10, 0x80)                # n 10 80 xx


def valid_pkt(rx: bytes) -> bool:
    """True if rx is a 4-byte packet with a correct check5 tail."""
    return (len(rx) == 4 and (rx[3] & 1) == 0
            and (rx[3] >> 1) & 0x1F == check5(rx[0], rx[1], rx[2], rx[3] >> 6))


def ack_for(rx: bytes) -> bytes:
    """Build the body ACK for a lens packet: cmd high bit set, carrying the
    tag2 of the packet being acknowledged (lens 0x08/tag2=0 -> 08 00 88 32;
    the 0x09 idle response is tag2=2, so its ACK is 08 00 89 b8)."""
    return pkt(0x08, 0x00, 0x80 | (rx[2] & 0x7F), tag2=rx[3] >> 6)


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
        # Open in mode 0 (CPOL=0) so SCLK is held at idle-LOW through the
        # settle window; arm() flips to mode 3, producing a single low->high
        # edge on SCLK right before the first packet. This probes whether the
        # lens anchors its first-contact timeout to that rising edge ("body
        # just enabled its bus") rather than to lens power-on.
        # NB: some SPI controllers only latch CPOL at the first transfer;
        # verify the edge timing on the analyzer.
        self.spi.mode = 0

    def arm(self) -> None:
        """Raise SCLK to its mode-3 idle-high level (ready to talk)."""
        if self.spi:
            self.spi.mode = SPI_MODE

    def disarm(self) -> None:
        """Drop SCLK back to idle-low (bus disabled, as before power-on)."""
        if self.spi:
            self.spi.mode = 0

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
        self.t0 = time.perf_counter()       # session start; transcript clock
        self.phase_t0 = self.t0             # current phase start; replay clock
        self.counter = 8  # rolling transport counter, cycles 0x8..0xf
        self.transcript = None
        if transcript_path:
            self.transcript = transcript_path.open("w")
            self.transcript.write("name\ttype\tstart_time\tduration\tmosi\tmiso\n")

    def now(self) -> float:
        """Seconds since the session started (monotonic across retries)."""
        return time.perf_counter() - self.t0

    def start_phase(self) -> None:
        self.phase_t0 = time.perf_counter()

    def phase_now(self) -> float:
        """Seconds since the current phase started; the replay time base."""
        return time.perf_counter() - self.phase_t0

    def wait_until(self, t: float) -> None:
        # phase-relative; sleep for the bulk, busy-wait the last 500us
        while True:
            dt = t - self.phase_now()
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
    """Pull the ASCII identity out of a 131-byte identification block.

    A lens sitting in its bootloader answers the ident read with the beacon
    pattern instead (c3 3c a5 5a repeating, byte-aligned or 4-bit-shifted),
    whose printable bytes ("Z 3 U <") would otherwise pass for an identity —
    reject any block carrying it.
    """
    if len(rx) < 131 or not any(rx[2:]):
        return None
    if MAGIC_WORD in rx or bytes.fromhex("0c33ca55a0") in rx:
        return None
    text = "".join(chr(c) if 32 <= c < 127 else " " for c in rx[2:0x52])
    return " ".join(text.split()) or None


def next_counter_after(replay: list[dict]) -> int:
    """Infer the transport counter the body would use after the replayed
    prefix, so the synthesized idle continues the capture's 0x8..0xf cycle.
    Transports (n 10 80) always consume a counter; body ACKs with b0 in
    9..f do too (b0=8 is ambiguous with the fixed-b0 ACK form, so a prefix
    ending on one of those can be off by one — the observed cutoffs are not)."""
    last, seen = 7, False
    for e in replay:
        tx = bytes.fromhex(e["tx"])
        if len(tx) != 4:
            continue
        if tx[1] == 0x10 and tx[2] == 0x80 and 0x8 <= tx[0] <= 0xF:
            last, seen = tx[0], True
        elif tx[1] == 0x00 and (tx[2] & 0xF0) == 0x80 and 0x9 <= tx[0] <= 0xF:
            last, seen = tx[0], True
    return 8 + ((last + 1 - 8) & 0x7) if seen else 8


def run_startup(sess: BodySession, replay: list[dict],
                abort_txn: int | None = None,
                expect_ident: str | None = None) -> bool:
    sess.start_phase()

    matches = mismatches = 0
    lens_identity = None
    aborted = False
    for i, entry in enumerate(replay):
        if abort_txn and i > abort_txn and lens_identity is None:
            # Lens has reset (it beacons within ms of dying) and won't hear
            # the rest of the replay. Bail out now so the retry can be timed
            # to its ~1.34s reboot instead of grinding through all 164 txns.
            print(f"  no identification by txn {abort_txn} — aborting attempt "
                  f"at t={sess.phase_now():.3f}s (lens likely reset)")
            aborted = True
            break
        tx = bytes.fromhex(entry["tx"])
        expected = bytes.fromhex(entry["rx_expected"])
        if sess.link.dry_run:
            sess.link.queue_dry_response(expected)
        sess.wait_until(entry["t"])
        rx = sess.xfer(tx)

        if len(tx) > 4:
            ident = decode_ident(rx)
            if ident and lens_identity is None:
                # The bootloader beacon's printable bytes ("< Z 3 U") also
                # decode to a non-empty string, so require the expected model
                # code before trusting it.
                if expect_ident and expect_ident not in ident:
                    print(f"  [{i}] ident block failed check "
                          f"(no '{expect_ident}'): {ident}")
                else:
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

    if not aborted:
        print(f"startup replay done: {matches} responses matched capture, "
              f"{mismatches} differed")
    if lens_identity is None and not sess.link.dry_run:
        print("no lens identification received")
    else:
        print(f"lens id: {lens_identity}")
    return lens_identity is not None or sess.link.dry_run


def run_startup_with_retry(sess: BodySession, replay: list[dict], settle: float,
                           retries: int, retry_delay: float,
                           abort_txn: int | None,
                           expect_ident: str | None) -> bool:
    """Replay the startup sequence, retrying if the lens never identifies.

    SCLK is held LOW through the settle window (SpiLink opens in mode 0);
    arm() raises it to the mode-3 idle level immediately before the first
    packet, so the lens sees a single low->high SCLK edge with data starting
    right after. Each retry drops the line back low (disarm) for the
    quiet-bus gap and raises it again, giving the lens a fresh edge per
    attempt, without a power cycle. The rolling transport counter is
    restarted per attempt.

    Retries are timed to the lens's reboot cycle: when a lens dies it
    beacons within ms and takes ~1.34s to boot back into its listening
    window (measured: reset at t~=0.025-0.029, first live packet again at
    t~=1.364). A failed attempt aborts early (--abort-txn) so the reset
    moment is known to within a few ms, then retry_delay (~1.4s default)
    puts the next replay's first packet just after the reboot completes.
    """
    attempts = retries + 1
    print(f"settling {settle:.2f}s with SCLK held low...")
    time.sleep(settle)
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(f"startup attempt {attempt} of {attempts}: SCLK low "
                  f"{retry_delay:.2f}s (lens reboot window), then re-raising")
            sess.counter = 8
            sess.link.disarm()
            time.sleep(retry_delay)
        print("raising SCLK, starting replay")
        sess.link.arm()
        ok = run_startup(sess, replay, abort_txn, expect_ident)
        if ok:
            return True
    if not sess.link.dry_run:
        print(f"WARNING: no lens identification after {attempts} attempt(s) "
              "— check wiring/power")
    return False


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


def attempt_resync(sess: BodySession) -> bool:
    """Perform the magic handshake from the GF110 firmware-update capture
    (body sends a5 5a 3c c3, then 80 20 28 24; the lens echo-acks with
    a5 5a 3c c3 and reports 00 f3 03 e8).

    NOTE: tested on hardware 2026-08-16 — a beaconing lens acks this
    reliably, but it is the ENTRY into firmware-update mode, not a transport
    resync: the lens keeps beaconing at normal traffic afterwards, including
    a full startup replay. Kept for future fw-update work; the working
    recovery from beacon state is a quiet-bus reboot (see run_idle)."""
    ok = False
    for j, tx in enumerate((MAGIC_REPLY, PKT_2824, transport(0x08),
                            IDLE_PKT, IDLE_PKT)):
        if j:
            time.sleep(INTRA_BURST_GAP_S)
        rx = sess.xfer(tx)
        if rx == MAGIC_REPLY or (j >= 2 and valid_pkt(rx) and any(rx)):
            ok = True
    if ok:
        sess.counter = 9  # the handshake's transport(8) consumed counter 8
    return ok


def run_idle(sess: BodySession, duration: float,
             replay: list[dict] | None = None) -> None:
    """Adaptive idle loop.

    Each 40ms burst starts with the status poll + transport packet (plus the
    0x09 poll every third burst), then continues slot by slot, reacting to
    what the lens actually sends: every valid non-ACK lens packet is ACKed
    with its own cmd and tag2 (not a canned byte), and a status packet whose
    b1 pending bits flag a ring readout gets the matching 0x0c follow-up
    poll in the same burst. The burst ends after two quiet trailing slots
    (or the slot cap), so its length breathes with the lens's traffic the
    way the real body's bursts do.

    The magic word means the lens has dropped to its bootloader (it beacons
    c3 3c a5 5a until either a fw-update handshake or a reboot). Recovery
    mirrors what works at power-on: SCLK low, ~1.4s of bus quiet so the
    bootloader times out into the app (measured boot ~1.34s), then the
    startup prefix again, then resume polling.
    """
    print(f"entering idle loop for {duration:.0f}s "
          "(adaptive bursts @40ms, 0x09 poll every third)...")
    deadline = time.perf_counter() + duration
    sess.start_phase()
    burst_i = 0
    last_status = None
    magic_mode = False
    resync_cooldown = 0  # bursts to wait before re-attempting the handshake
    next_burst = IDLE_PERIOD_S

    while time.perf_counter() < deadline:
        sess.wait_until(next_burst)
        queue = [STATUS_POLL, transport(sess.next_counter())]
        if burst_i % 3 == 0:
            queue.append(POLL_09)

        polled: set[bytes] = set()
        slots = quiet = 0
        while (queue or quiet < 2) and slots < 14:
            tx = queue.pop(0) if queue else IDLE_PKT
            if slots:
                time.sleep(INTRA_BURST_GAP_S)
            rx = sess.xfer(tx)
            slots += 1

            if rx == MAGIC_WORD:
                if not magic_mode:
                    magic_mode = True
                    print(f"  t={sess.now():8.3f} lens dropped to bootloader "
                          "(beacon) — transport lost")
                if replay is not None and resync_cooldown == 0:
                    print(f"  t={sess.now():8.3f} quiet-rebooting lens: SCLK "
                          f"low, {REBOOT_QUIET_S:.1f}s silence, then startup")
                    sess.link.disarm()
                    time.sleep(REBOOT_QUIET_S)
                    sess.link.arm()
                    if run_startup(sess, replay):
                        magic_mode = False
                        sess.counter = next_counter_after(replay)
                        # run_startup reset the phase clock; realign the
                        # burst schedule to it
                        next_burst = sess.phase_now()
                        burst_i = 0
                    else:
                        resync_cooldown = 25  # ~1s of polling between tries
                break  # rest of this burst is moot either way
            if not (valid_pkt(rx) and any(rx)):
                if not queue:
                    quiet += 1
                continue
            if magic_mode:
                magic_mode = False
                print(f"  t={sess.now():8.3f} lens left resync state")
            if rx[2] & 0x80:
                continue  # ACK from the lens (incl. n 00 80 transport acks)

            quiet = 0
            cmd = rx[2] & 0x7F
            queue.append(ack_for(rx))
            if cmd == 0x08:
                status = describe_status(rx)
                if status and status != last_status:
                    print(f"  t={sess.now():8.3f} lens status: {status}")
                    last_status = status
                pend = rx[1] & 0x7F
                if pend & 0x08 and FOCUS_POLL not in polled:
                    polled.add(FOCUS_POLL)
                    queue.append(FOCUS_POLL)
                if pend & 0x10 and APERTURE_POLL not in polled:
                    polled.add(APERTURE_POLL)
                    queue.append(APERTURE_POLL)
            elif cmd == 0x0C:
                # readout tag2 mirrors the poll's tag2: 2 = focus, 0 = aperture
                kind = {2: "focus", 0: "aperture"}.get(rx[3] >> 6, "ring?")
                print(f"  t={sess.now():8.3f} {kind} ring: b0={rx[0]:02x} "
                      f"b1={rx[1]:02x} (raw {rx.hex(' ')})")
            elif cmd == 0x09:
                if rx[1]:
                    print(f"  t={sess.now():8.3f} 0x09 rate: "
                          f"b0={rx[0]:02x} b1={rx[1]:02x}")
            elif cmd == 0x03:
                print(f"  t={sess.now():8.3f} lens error/desync report "
                      f"({rx.hex(' ')})")

        burst_i += 1
        if resync_cooldown:
            resync_cooldown -= 1
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
    ap.add_argument("--startup-retries", type=int, default=5,
                    help="extra attempts at the startup sequence if the lens "
                         "never identifies (default 5 — aborted attempts are "
                         "cheap; 0 to give up on first fail)")
    ap.add_argument("--retry-delay", type=float, default=1.4,
                    help="quiet seconds before each retry, timed so the replay "
                         "lands in the lens's post-reboot listening window "
                         "(lens boots in ~1.34s; default 1.4)")
    ap.add_argument("--abort-txn", type=int, default=20,
                    help="abort a startup attempt early if no identification "
                         "by this transaction, so the retry stays synced to "
                         "the lens reboot (0 disables early abort)")
    ap.add_argument("--expect-ident", default="",
                    help="substring a real identification must contain (e.g. "
                         "LR107A), rejecting bootloader-beacon garbage that "
                         "decodes as an ident (default: check disabled)")
    ap.add_argument("--replay-end", type=int, default=64,
                    help="replay only the first N captured transactions — the "
                         "capture's tail is state-dependent idle traffic that "
                         "desyncs a live lens (default 64, the end of the "
                         "deterministic config prefix; 0 = full capture)")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--transcript", type=Path,
                    help="write Saleae-style TSV of the session (byte-accurate "
                         "tx/rx, analyzable with fuji_spi.py)")
    ap.add_argument("--dry-run", action="store_true",
                    help="no SPI hardware; lens responses simulated from capture")
    args = ap.parse_args()

    replay = json.loads(args.replay.read_text())["transactions"]
    total = len(replay)
    if args.replay_end > 0:
        replay = replay[:args.replay_end]
    print(f"replaying {len(replay)} of {total} startup transactions "
          f"from {args.replay}")

    link = SpiLink(args.bus, args.device, args.dry_run)
    sess = BodySession(link, args.transcript)
    try:
        ok = run_startup_with_retry(sess, replay, args.settle,
                                    max(0, args.startup_retries),
                                    args.retry_delay,
                                    args.abort_txn or None,
                                    args.expect_ident or None)
        if not ok:
            print("aborting before idle loop (no lens response)")
            sys.exit(1)
        sess.counter = next_counter_after(replay)
        run_idle(sess, args.idle_seconds, replay)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        sess.close()
        if args.transcript:
            print(f"transcript written to {args.transcript}")


if __name__ == "__main__":
    main()
