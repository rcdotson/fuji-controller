#!/usr/bin/env python3
"""Replicate the Fuji GFX body's power-on sequence to bring a G-mount lens
into the idle state, driven by a Raspberry Pi 5 as SPI master.

Phase 1 (startup): replays the deterministic prefix of the body-side
transactions captured in startup_replay.json (exported from power_on.txt by
fuji_spi.py) with the original timing, comparing live responses against the
capture. Only the first --replay-end transactions are replayed: the capture's
tail is state-dependent idle traffic, and canned ACKs there desync a live
lens (it then answers everything with the c3 3c a5 5a resync word).
Phase 2 (idle): synthesizes the body's polling loop adaptively as request
quads — every request (status, 0x09, 0x0c readout) framed as
[request, transport(n), idle, ack], the invariant structure the real body
uses (see focus_ring_back_forth.txt); ACKs are computed from the lens's
actual packets (cmd | 0x80, matching tag2). A lens streaming c3 3c a5 5a
has lost transport sync and wants the reset dialogue (counterpart marker,
0x28 session reset, ACKs including its 0x03 error report — see
transport_reset); if the marker persists after that, the startup prefix is
re-run as escalation.

Phase 3 (shutdown): replays the body's power-off sequence, captured in
startup_shutdown.txt (t=4.958-5.203) — OIS off (0x20 payload 0), park via
0x28 channel 0x8001 + 0x10, wait for the status busy bit to clear (~129ms),
then read the final focus/iris back before power is cut. It runs on exit and
on the `2` key, so the lens is never simply de-energized mid-motion.

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

During the idle session (tty only), keys drive the lens:
    iris:  `]` stop down a third-stop, `[` open up, `o` wide open (1),
           `c` fully closed (22)
    focus: `m`/`n` step the motor +/-2 counts, `.`/`,` +/-50, `>`/`<` +/-500
    power: `1` lens power on, `2` off (GPIO6 -> external high-side switch;
           raised automatically before the settle window at startup, driven
           low again when the script exits). `2` and script exit both park
           the lens first — see Phase 3.
    `q` quits
Iris setpoints are staged with 0x18 and latched with the 3f execute, each
followed by a feedback poll (00 01 08 82) reading back the landed index.
Focus moves use the captured 10-slot 0x15 sequence (envelope, speed 288,
absolute signed 16-bit target, execute). After every accepted move the
position is actively polled with 00 01 08 42 (the tag-1 sibling of the
iris feedback poll, seen in the ring-service captures) once per burst
until two consecutive readings agree, printing each change and a final
"focus settled at N"; the settled position seeds the next move. The
first focus move starts from position 0 unless feedback has been seen —
expect a jump toward the infinity end on the first press.

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
IRIS_FEEDBACK = pkt(0x00, 0x01, 0x08, tag2=2)  # 00 01 08 82: iris state poll
FOCUS_POS_POLL = pkt(0x00, 0x01, 0x08, tag2=1)  # 00 01 08 42: focus position
SYNC_EXECUTE = pkt(0x00, 0x00, 0x3F, tag2=3)   # 00 00 3f c6: execute/latch
ACK_BF = pkt(0x08, 0x00, 0xBF, tag2=3)         # 08 00 bf d8

# --- power-down vocabulary (ground truth: startup_shutdown.txt) -------------
# The body's shutdown is four phases: OIS off (0x20 payload 0), the park
# command on channel 0x8001, a wait for the status busy bit to clear, then a
# state snapshot before power is cut.
#
# 0x20 carries the OIS state, not a general actuator enable: the same GF250
# with its OIS switch flipped sends payload 1 (startup_shutdown.txt t=2.0624)
# vs payload 0 (startup_shutdown_no_ois.txt t=1.5544) at the same point in
# startup, tracking the b1 bit-6 the lens reports. Shutdown sends payload 0
# either way.
OIS_ON = pkt(0x00, 0x01, 0x20)                 # 00 01 20 24
OIS_OFF = pkt(0x00, 0x00, 0x20)                # 00 00 20 04
PARK = pkt(0x00, 0x01, 0x10)                   # 00 01 10 22: park/shutdown
STATE_LATCH = pkt(0x00, 0x00, 0x0F, tag2=3)    # 00 00 0f c0: opens a readout
POLL_09_SUB = pkt(0x00, 0x01, 0x09)            # 00 01 09 04: 0x09 sub-read
POLL_09_T3 = pkt(0x00, 0x00, 0x09, tag2=3)     # 00 00 09 e8: 0x09 tag-3 read

# The five-request block the body runs both after identification (t=1.959 in
# startup_shutdown.txt) and immediately before cutting power (t=5.200) — a
# state snapshot on the way in and on the way out, byte-identical both times.
STATE_READOUT = [STATE_LATCH, POLL_09_SUB, FOCUS_POS_POLL, IRIS_FEEDBACK,
                 POLL_09_T3]


def channel(n: int, tag2: int = 0) -> bytes:
    """0x28 channel selector: the payload picks which subsystem the staged
    command that follows addresses. Observed: 0x8001 park (shutdown,
    tag2=0), 0x8002 focus motor (tag2=0, precedes a 0x15 drive), 0x8004 iris
    (tag2=1, precedes a 0x18 setpoint), 0x8020 transport reset. tag2 is not
    constant across channels, so pass it explicitly outside the park path."""
    return pkt(0x80, n, 0x28, tag2=tag2)


PARK_CHANNEL = 0x01


def iris_setpoint(index: int) -> bytes:
    """0x18 staged iris setpoint: index 1 (wide open) .. 22 (fully closed),
    third-stop steps; encoding per protocol README Aperture Drive section
    (BE16 payload = 0x1000 + (index-1)*0x40, e.g. 10 00 18 a8 = index 1)."""
    v = 0x1000 + (index - 1) * 0x40
    return pkt(v >> 8, v & 0xFF, 0x18, tag2=2)
# Transport resync marker: when the lens loses transport sync (protocol
# violation, or fresh out of reset) it streams this word — a classic line-sync
# pattern of bit-complement pairs — until the body answers with the
# counterpart and completes the reset dialogue (see transport_reset). The
# fw-update capture uses the same exchange before its block transfers; it is
# a generic transport reset, not update-specific. Not a valid check5 packet,
# so it never enters the ACK path.
MAGIC_WORD = bytes.fromhex("c33ca55a")
MAGIC_REPLY = bytes.fromhex("a55a3cc3")      # body's counterpart marker
PKT_2824 = pkt(0x80, 0x20, 0x28)             # 80 20 28 24: session reset


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


def ack_staged(n: int, staged: bytes) -> bytes:
    """ACK of a staged command's echo, inside a drive sequence.

    Same cmd|0x80 / matching-tag2 rule as ack_for, but b0 carries the rolling
    transport counter instead of the fixed 0x08 (the counter-carrying ACK
    path). Reproduces every observed sequence ACK: 0d 00 a8 1e for the
    80 01 28 24 channel select, 0e 00 90 04 for the 00 01 10 22 park,
    09 00 a0 1e for the 00 00 20 04 disable, and the 0x95/0x98 forms the
    focus and iris sequences already used."""
    return pkt(n, 0x00, 0x80 | (staged[2] & 0x7F), tag2=staged[3] >> 6)


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
# Lens power (external high-side switch driven by a GPIO, active high)
# ---------------------------------------------------------------------------

class LensPower:
    """Drives the external lens-power circuit via a GPIO (default GPIO6).

    Uses lgpio directly: gpiozero's default chip lookup fails on this
    kernel's renumbered gpiochips, so the RP1 header bank is found by its
    54-line signature. The pin is claimed LOW (power off) at start; close()
    drives it low again, so exiting the script cuts lens power."""

    def __init__(self, gpio: int):
        self.gpio = gpio
        self.state = False
        self.h = None
        if gpio < 0:
            return
        try:
            import lgpio  # noqa: PLC0415 -- only needed on the Pi
        except ImportError:
            print("lens power: python3-lgpio not available — disabled")
            return
        self._lgpio = lgpio
        for chip in range(17):
            try:
                h = lgpio.gpiochip_open(chip)
            except lgpio.error:
                continue
            if lgpio.gpio_get_chip_info(h)[1] >= 54:  # RP1 header bank
                lgpio.gpio_claim_output(h, gpio, 0)
                self.h = h
                return
            lgpio.gpiochip_close(h)
        print("lens power: no 54-line gpiochip found — disabled")

    @property
    def enabled(self) -> bool:
        return self.h is not None

    def set(self, on: bool) -> None:
        if self.h is not None:
            self._lgpio.gpio_write(self.h, self.gpio, 1 if on else 0)
            self.state = on

    def close(self) -> None:
        if self.h is not None:
            self._lgpio.gpio_write(self.h, self.gpio, 0)
            self._lgpio.gpio_free(self.h, self.gpio)
            self._lgpio.gpiochip_close(self.h)
            self.h = None


# ---------------------------------------------------------------------------
# Keyboard (non-blocking single-key input during the idle session)
# ---------------------------------------------------------------------------

class Keyboard:
    """Raw-mode stdin poller; inert when stdin is not a tty (pipes, dry runs
    under automation). Call restore() before the process exits."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.saved = None
        if self.enabled:
            import termios  # noqa: PLC0415 -- POSIX-only, matches deployment
            import tty      # noqa: PLC0415
            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        import select  # noqa: PLC0415
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def restore(self) -> None:
        if self.saved is not None:
            import termios  # noqa: PLC0415
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None


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

def is_status(rx: bytes) -> bool:
    """True for a lens tag-0 0x08 status response (not an ack)."""
    return (len(rx) == 4 and (rx[2] & 0x7F) == 0x08 and not (rx[2] & 0x80)
            and (rx[3] >> 6) == 0)


def status_busy(rx: bytes) -> bool:
    """b1 bit 1: an actuator is moving. Set during the iris move at t=2.30
    and for the 129ms the park takes at t=4.98 in startup_shutdown.txt."""
    return bool(rx[1] & 0x02)


def status_disabled(rx: bytes) -> bool:
    """b1 bit 0: actuators not ready. Set through the lens's boot, clear once
    it is running, and set again after the park — but it also flickers during
    the startup config block (startup_shutdown_no_ois.txt t=1.49-1.57), so
    treat it as advisory rather than a clean latch."""
    return bool(rx[1] & 0x01)


def af_range_limited(rx: bytes) -> bool:
    """b0 bit 0: the AF range switch is at 5m-infinity rather than full.

    Set in all 171 status packets of the two 5m-infinity captures and clear
    in all 84 of startup_shutdown_focus_full_range.txt — same lens and body,
    switch flipped. The body sends nothing in response to it: the request
    vocabulary is identical across those captures, so this is purely a
    lens-to-body report."""
    return bool(rx[0] & 0x01)


def detail_pending(rx: bytes) -> bool:
    """b0 bit 4: a lens-detail readout is waiting on 0x0c tag2=1.

    Distinct from the b1 bit-3 focus-ring flag: across the three
    startup/shutdown captures this bit is set 20 times and the body answers
    each with 00 00 0c 72, while focus_ring_back_forth.txt's 44 ring events
    set b1 bit 3 with this bit clear. The tag2=1 payload (0x0040-0x004c) is
    not decoded and this engine does not issue the poll."""
    return bool(rx[0] & 0x10)


def ois_active(rx: bytes) -> bool:
    """b1 bit 6: the lens's OIS switch is on.

    Set in every status packet of startup_shutdown.txt and clear in every one
    of startup_shutdown_no_ois.txt — same lens and body, switch flipped —
    including the first packet after boot, before the body has sent anything.
    Clear on a GF45, which has no OIS at all."""
    return bool(rx[1] & 0x40)


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
        names.append("busy")
    if pend & 0x01:
        names.append("not-ready")
    if rx[1] & 0x40:
        names.append("ois")
    if rx[0] & 0x10:
        names.append("detail")
    if rx[0] & 0x01:
        names.append("af-limit")
    return f"b0={rx[0]:02x} b1={rx[1]:02x} pending={'+'.join(names) or 'none'}"


def transport_reset(sess: BodySession) -> bool:
    """Complete the lens's transport-reset dialogue.

    A lens streaming the resync marker wants the full exchange the captured
    body performs (fw-update capture, but the mechanism is generic): the
    counterpart marker, the 0x28 session-reset, a fresh transport packet,
    and ACKs for everything it answers — including the 0x03 error report
    (e.g. 00 f3 03 e8) that earlier recovery attempts left unacknowledged,
    which kept the lens in marker state. Every ACK is framed with its own
    transport packet, matching the per-request framing the real body uses
    everywhere. Success = the lens stopped streaming the marker and spoke
    at least one valid packet."""
    queue = [MAGIC_REPLY, PKT_2824, transport(0x08), IDLE_PKT,
             pkt(0x08, 0x00, 0xA8)]  # 08 00 a8 36: ack of the lens's a8 reply
    n = 9  # dialogue transport counters restart at 8; transport(8) used above
    marker_free = 0
    got_valid = False
    slots = busy = 0
    while (queue or marker_free < 3) and slots < 24:
        tx = queue.pop(0) if queue else IDLE_PKT
        if slots:
            time.sleep(INTRA_BURST_GAP_S)
        rx = sess.xfer(tx)
        slots += 1
        if rx == MAGIC_WORD:
            marker_free = 0
            continue
        marker_free += 1
        if valid_pkt(rx) and any(rx):
            got_valid = True
            if rx[2] & 0x80:
                # busy flag: repeat the same packet until the ack is clean
                # (attempt_6 showed 08 10 a8 06 mid-dialogue — the lens
                # asking us to wait, per the captured body's retry behavior)
                if rx[1] & 0x10 and busy < 6:
                    busy += 1
                    queue.insert(0, tx)
            else:
                queue.extend([transport(n), ack_for(rx)])
                n = 8 + ((n + 1 - 8) & 0x7)
                if (rx[2] & 0x7F) == 0x03:
                    print(f"  t={sess.now():8.3f} lens error report "
                          f"{rx.hex(' ')} (code {rx[1]:02x}) — acked")
    if got_valid and marker_free >= 3:
        sess.counter = n
        return True
    return False


def run_sequence(sess: BodySession, seq: list[bytes]) -> list[bytes] | None:
    """Send a staged control sequence slot by slot, honoring busy flags
    (repeat the same packet until the ack is clean). Returns the responses,
    or None if the lens dropped to the resync marker mid-sequence."""
    responses: list[bytes] = []
    busy = 0
    i = 0
    while i < len(seq):
        tx = seq[i]
        time.sleep(INTRA_BURST_GAP_S)
        rx = sess.xfer(tx)
        if rx == MAGIC_WORD:
            return None
        responses.append(rx)
        if valid_pkt(rx) and any(rx) and rx[2] & 0x80:
            if rx[1] & 0x10 and busy < 8:
                busy += 1
                continue  # busy flag: repeat the same packet
            if rx[1] & ~0x10:
                # unknown flag bits in an ack are the lens objecting (e.g.
                # 0a 08 95 42 right before the marker drop in
                # failed_at_about_3min.tsv); pressing on desyncs the
                # transport, so abort the command instead
                print(f"  t={sess.now():8.3f} lens flagged the sequence "
                      f"({rx.hex(' ')}) — aborting command")
                return None
        i += 1
    return responses


def staged_sequence(sess: BodySession, stages: list[bytes]) -> list[bytes]:
    """Build the body's staged-command sequence for any number of stages.

    Every control write in the captures — iris, focus, the 0x20 enable and
    the 0x28+0x10 park — is the same template: the stages go out on the even
    slots, one transport frame follows the first stage, and each stage's ACK
    lands two slots after it, so the ACKs trail the stages by one. The 3f
    execute occupies the slot after the last stage, and the sequence closes
    on an idle and the bf execute-ack:

        stage0, transport(n), stage1, ack(stage0), ... ,
        execute, ack(stage_last), idle, ACK_BF

    Verified slot-for-slot against startup_shutdown.txt: 6 slots for the
    one-stage 0x20 disable (t=4.9588), 8 for the two-stage park (t=4.9732),
    and it reproduces the 6-slot iris and 10-slot focus sequences this file
    previously spelled out by hand.
    """
    seq = [stages[0], transport(sess.next_counter())]
    for i in range(1, len(stages)):
        seq.append(stages[i])
        seq.append(ack_staged(sess.next_counter(), stages[i - 1]))
    seq.append(SYNC_EXECUTE)
    seq.append(ack_staged(sess.next_counter(), stages[-1]))
    seq.extend([IDLE_PKT, ACK_BF])
    return seq


def staged_command(sess: BodySession, stages: list[bytes]) -> bool:
    """Run a staged command sequence. True when the lens echoed one of the
    staged packets or the execute — its documented accept signals."""
    responses = run_sequence(sess, staged_sequence(sess, stages))
    if responses is None:
        return False
    return any(rx in stages or (valid_pkt(rx) and (rx[2] & 0x7F) == 0x3F)
               for rx in responses)


def command_iris(sess: BodySession, index: int) -> bool:
    """Stage and latch an iris setpoint (first proven control write).

    Sequence per the README's Aperture Drive notes and the captured 0x15
    motor-drive pattern: stage the 0x18 setpoint and latch it with the 3f
    execute, acked via the counter-carrying 0x98 path."""
    return staged_command(sess, [iris_setpoint(index)])


FOCUS_SPEED_288 = pkt(0x01, 0x20, 0x15, tag2=1)  # 01 20 15 40: manual speed


def command_focus(sess: BodySession, target: int,
                  prev: int | None) -> bool:
    """Drive the focus motor to an absolute position (signed 16-bit counts).

    Three stages, from the manual-focus bursts in focus_ring_back_forth.txt:
    tag0 envelope (move budget ~= 2500 + 2.2 x |travel|, fitted from 44
    captured sequences), tag1 speed (288, the manual-focus constant), and the
    tag2 absolute target."""
    delta = abs(target - prev) if prev is not None else 0
    env = min(20000, 2500 + (delta * 11) // 5)
    tb = target.to_bytes(2, "big", signed=True)
    envelope = pkt(env >> 8, env & 0xFF, 0x15)
    staged = pkt(tb[0], tb[1], 0x15, tag2=2)
    return staged_command(sess, [envelope, FOCUS_SPEED_288, staged])


# ---------------------------------------------------------------------------
# Enable / shutdown (mirrors the body's power-on and power-off sequences)
# ---------------------------------------------------------------------------

def request_quad(sess: BodySession, req: bytes) -> tuple[bytes | None, bool]:
    """Issue one request quad and return (payload, hit_marker).

    The standalone form of the quad run_idle builds inline: request,
    transport frame, idle, ack of whatever came back. payload is the lens's
    non-ack response, or None if it only acked."""
    payload = None
    plan: list[bytes | None] = [req, transport(sess.next_counter()),
                                IDLE_PKT, None]
    retries = 0
    j = 0
    while j < len(plan):
        tx = plan[j]
        if tx is None:
            tx = ack_for(payload) if payload else IDLE_PKT
        time.sleep(INTRA_BURST_GAP_S)
        rx = sess.xfer(tx)
        if rx == MAGIC_WORD:
            return None, True
        if valid_pkt(rx) and any(rx):
            if rx[2] & 0x80:
                if rx[1] & 0x10 and retries < 8:
                    retries += 1   # busy: repeat the same packet
                    continue
            else:
                payload = rx
        j += 1
    return payload, False


def sync_ois(sess: BodySession) -> str:
    """Send the 0x20 the body sends after identification, matching the lens.

    The payload is not ours to choose: it mirrors the OIS switch the lens
    reports in status b1 bit 6 (payload 1 with the switch on, 0 with it off —
    the only packet that differs between startup_shutdown.txt and
    startup_shutdown_no_ois.txt). Sending payload 1 to a lens whose switch is
    off would be the body overriding the user's switch, so the status is read
    first and nothing is sent if it cannot be read.

    Returns 'on', 'off', 'rejected' or 'unknown'."""
    payload, magic = request_quad(sess, STATUS_POLL)
    if magic or payload is None or not is_status(payload):
        return "unknown"
    on = ois_active(payload)
    if not staged_command(sess, [OIS_ON if on else OIS_OFF]):
        return "rejected"
    return "on" if on else "off"


def read_lens_state(sess: BodySession) -> dict:
    """Run the body's five-request state readout (STATE_READOUT).

    Returns {'focus': int|None, 'iris': int|None, 'reachable': bool}. Issued
    as five quads rather than the body's tighter pipelined form; the
    request/response pairs are identical on the wire."""
    out: dict = {"focus": None, "iris": None, "reachable": True}
    for req in STATE_READOUT:
        payload, magic = request_quad(sess, req)
        if magic:
            out["reachable"] = False
            return out
        if payload is None or (payload[2] & 0x7F) != 0x08:
            continue
        tag = payload[3] >> 6
        if tag == 1:
            pos = int.from_bytes(payload[:2], "big", signed=True)
            if pos != 32767:      # encoder out-of-range sentinel
                out["focus"] = pos
        elif tag == 2:
            out["iris"] = payload[0] & 0x1F
    return out


def run_shutdown(sess: BodySession, timeout: float = 1.5,
                 grace_bursts: int = 2, on_poll=None) -> dict:
    """Shut the lens down the way the body does, before power is removed.

    Replays the four phases captured in startup_shutdown.txt t=4.958-5.203:

      A  0x20 payload 0 + execute        disable the actuators   (t=4.9588)
      B  0x0f latch, 0x09 sub-read, then
         0x28 channel 0x8001 + 0x10 payload 1 + execute   park   (t=4.9706)
      C  poll status at the burst period until the b1 busy bit clears
         (129ms in the capture), then grace_bursts more            (t=4.98+)
      D  the five-request state readout, so the parked focus and iris
         positions are known before the rail drops               (t=5.1996)

    on_poll, if given, is called once per phase-C status poll; returning True
    abandons the wait (the caller wants to cut power now).

    Returns {'status': 'ok'|'timeout'|'rejected'|'unreachable'|'aborted',
             'focus': int|None, 'iris': int|None, 'wait_s': float}.
    The caller must cut power afterwards whatever the status says — a failed
    dialogue is a reason to log, not a reason to leave the lens energized."""
    result: dict = {"status": "ok", "focus": None, "iris": None,
                    "wait_s": 0.0}

    def unreachable() -> dict:
        result["status"] = "unreachable"
        print(f"  t={sess.now():8.3f} shutdown: lens stopped answering — "
              "cutting power without finishing the sequence")
        return result

    print(f"  t={sess.now():8.3f} shutdown: OIS off")
    if not staged_command(sess, [OIS_OFF]):
        # the park is what protects the mechanism, so press on regardless
        print(f"  t={sess.now():8.3f} shutdown: disable not acknowledged "
              "— continuing to the park")

    time.sleep(INTRA_BURST_GAP_S)
    for req in (STATE_LATCH, POLL_09_SUB):
        _, magic = request_quad(sess, req)
        if magic:
            return unreachable()
    print(f"  t={sess.now():8.3f} shutdown: parking (0x28 channel "
          f"{PARK_CHANNEL:#04x} + 0x10)")
    if not staged_command(sess, [channel(PARK_CHANNEL), PARK]):
        # nothing to wait for: skip phase C rather than burn the timeout,
        # but still read the state back before the rail drops
        result["status"] = "rejected"
        print(f"  t={sess.now():8.3f} shutdown: park not acknowledged")
        state = read_lens_state(sess)
        if not state["reachable"]:
            return unreachable()
        result["focus"], result["iris"] = state["focus"], state["iris"]
        return result

    # Phase C: the park is asynchronous — the lens raises the busy bit and
    # drops it when it has finished retracting.
    t_park = time.perf_counter()
    deadline = t_park + timeout
    next_poll = t_park
    saw_busy = False
    parked = False
    polls = 0
    while time.perf_counter() < deadline:
        next_poll += IDLE_PERIOD_S
        dt = next_poll - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        if on_poll is not None and on_poll():
            result["status"] = "aborted"
            result["wait_s"] = time.perf_counter() - t_park
            return result
        payload, magic = request_quad(sess, STATUS_POLL)
        polls += 1
        if magic:
            return unreachable()
        if payload is None or not is_status(payload):
            continue
        if status_busy(payload):
            saw_busy = True
        elif saw_busy or polls >= 5:
            # busy cleared, or the lens never raised it (nothing to retract)
            parked = True
            break
    result["wait_s"] = time.perf_counter() - t_park
    if parked:
        print(f"  t={sess.now():8.3f} shutdown: parked after "
              f"{result['wait_s'] * 1000:.0f}ms ({polls} polls)")
    else:
        result["status"] = "timeout"
        print(f"  t={sess.now():8.3f} shutdown: WARNING park did not "
              f"complete within {timeout:.1f}s — reading out anyway")

    for _ in range(max(0, grace_bursts)):
        next_poll += IDLE_PERIOD_S
        dt = next_poll - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        _, magic = request_quad(sess, STATUS_POLL)
        if magic:
            return unreachable()

    state = read_lens_state(sess)
    if not state["reachable"]:
        return unreachable()
    result["focus"], result["iris"] = state["focus"], state["iris"]
    print(f"  t={sess.now():8.3f} shutdown: final state focus="
          f"{result['focus']} iris={result['iris']} — safe to cut power")
    return result


def run_idle(sess: BodySession, duration: float,
             replay: list[dict] | None = None,
             kb: Keyboard | None = None,
             power: LensPower | None = None) -> None:
    """Adaptive idle loop built from request quads.

    The real body frames EVERY request the same way (ground truth:
    focus_ring_back_forth.txt ring-service bursts):

        tx <request>      rx (previous traffic)
        tx transport(n)   rx <lens ack of request>
        tx 00 00 00 00    rx <response payload>
        tx <ack payload>  rx <lens transport-ack of n>

    A 40ms burst is a status quad, a 0x09 quad every third burst, and one
    quad per 0x0c readout the status packet's pending bits request. Sending
    requests without their own transport frame is a transport violation
    (the lens answers with error 0x26 and drops to the resync marker) —
    that was the pre-quad engine's instability.

    The resync marker (c3 3c a5 5a) means the lens has lost transport sync
    — its app is fine and waiting, so quiet time does nothing (verified:
    3x 1.4s silences changed nothing). Recovery is protocol-level:
    transport_reset completes the reset dialogue; a persistent marker
    escalates to re-running the startup prefix.
    """
    print(f"entering idle loop for {duration:.0f}s "
          "(request quads @40ms, 0x09 every third burst)...")
    if kb and kb.enabled:
        print("  iris:  ] stop down   [ open up   o wide open   c closed\n"
              "  focus: n/m -/+2   ,/. -/+50   </> -/+500   (- near, + far)\n"
              "  power: 1 on   2 off      q quits")
    deadline = time.perf_counter() + duration
    sess.start_phase()
    burst_i = 0
    last_status = None
    last_09: bytes | None = None
    magic_mode = False
    resync_cooldown = 0  # bursts to wait before re-attempting recovery
    reset_fails = 0      # consecutive failed reset dialogues before re-init
    iris_target: int | None = None
    pending_iris = False
    want_feedback = False
    focus_target: int | None = None
    focus_pos: int | None = None
    pending_focus = False
    focus_poll_left = 0        # bursts of position polling after a move
    focus_settle_prev: int | None = None
    next_burst = IDLE_PERIOD_S

    def handle(rx: bytes) -> list[bytes]:
        """Decode a lens payload; return follow-up requests it asks for."""
        nonlocal last_status
        cmd = rx[2] & 0x7F
        reqs: list[bytes] = []
        if cmd == 0x08:
            if rx[3] >> 6 == 2:
                # iris feedback (response to 00 01 08 82): b0 low 5 bits are
                # the iris index, top 3 are flags (README Aperture Drive)
                print(f"  t={sess.now():8.3f} iris state: "
                      f"index={rx[0] & 0x1F} flags={rx[0] >> 5:03b} "
                      f"aux={rx[1]:02x} (raw {rx.hex(' ')})")
                return reqs
            if rx[3] >> 6 == 1:
                # focus position feedback (tag 1, signed BE16; 32767 is the
                # encoder's out-of-range/inactive sentinel)
                nonlocal focus_pos
                pos = int.from_bytes(rx[:2], "big", signed=True)
                if pos != 32767:
                    if pos != focus_pos:
                        print(f"  t={sess.now():8.3f} focus position: {pos} "
                              f"(raw {rx.hex(' ')})")
                    focus_pos = pos
                return reqs
            status = describe_status(rx)
            if status and status != last_status:
                print(f"  t={sess.now():8.3f} lens status: {status}")
                last_status = status
            pend = rx[1] & 0x7F
            if pend & 0x08:
                reqs.append(FOCUS_POLL)
            if pend & 0x10:
                reqs.append(APERTURE_POLL)
        elif cmd == 0x0C:
            # readout tag2 mirrors the poll's tag2: 2 = focus, 0 = aperture;
            # value is a signed 16-bit delta (ff ff = -1 in the captures)
            kind = {2: "focus", 0: "aperture"}.get(rx[3] >> 6, "ring?")
            delta = int.from_bytes(rx[:2], "big", signed=True)
            print(f"  t={sess.now():8.3f} {kind} ring: delta={delta:+d} "
                  f"(raw {rx.hex(' ')})")
        elif cmd == 0x09:
            # near-constant value (0x18/0x19 on the GF250, probably not a
            # rotation rate); log only when it changes to avoid clutter
            nonlocal last_09
            if rx[:2] != last_09:
                print(f"  t={sess.now():8.3f} 0x09 value changed: "
                      f"b0={rx[0]:02x} b1={rx[1]:02x}")
                last_09 = rx[:2]
        elif cmd == 0x03:
            print(f"  t={sess.now():8.3f} lens error/desync report "
                  f"({rx.hex(' ')}, code {rx[1]:02x})")
        return reqs

    while time.perf_counter() < deadline:
        sess.wait_until(next_burst)

        key = kb.poll() if kb else None
        if key == "q":
            print("  'q' pressed — ending session")
            break
        if key in ("1", "2"):
            if power and power.enabled:
                on = key == "1"
                if not on and not magic_mode:
                    # park the lens before the rail drops, as the body does
                    run_shutdown(sess)
                power.set(on)
                print(f"  t={sess.now():8.3f} lens power "
                      f"{'ON' if on else 'OFF'} (GPIO{power.gpio})")
                if on:
                    print("  (lens boots in ~1.4s; the marker recovery "
                          "will re-init it)")
            else:
                print("  lens power control not available")
        if key in ("[", "]", "o", "c"):
            cur = iris_target or 1
            new = {"o": 1, "c": 22,
                   "]": min(22, cur + 1),
                   "[": max(1, cur - 1)}[key]
            if new != iris_target:
                iris_target = new
                pending_iris = True
        if key in ("n", "m", ",", ".", "<", ">"):
            # base the first move on live position feedback if we have it
            base = focus_target if focus_target is not None else \
                (focus_pos if focus_pos is not None else 0)
            step = {"n": -2, "m": 2, ",": -50, ".": 50,
                    "<": -500, ">": 500}[key]
            new = max(-32768, min(32767, base + step))
            if new != focus_target:
                focus_target = new
                pending_focus = True

        requests = [STATUS_POLL]
        if burst_i % 3 == 0:
            requests.append(POLL_09)
        if want_feedback:
            requests.append(IRIS_FEEDBACK)
            want_feedback = False
        if focus_poll_left:
            requests.append(FOCUS_POS_POLL)
        issued: set[bytes] = set(requests)
        hit_magic = False
        quads = 0
        first_slot = True

        while requests and quads < 5 and not hit_magic:
            req = requests.pop(0)
            quads += 1
            payload = None
            frame = transport(sess.next_counter())
            plan = [req, frame, IDLE_PKT, None]  # None = ack slot
            retries = 0
            j = 0
            while j < len(plan):
                tx = plan[j]
                if tx is None:
                    tx = ack_for(payload) if payload else IDLE_PKT
                if not first_slot:
                    time.sleep(INTRA_BURST_GAP_S)
                first_slot = False
                rx = sess.xfer(tx)
                if rx == MAGIC_WORD:
                    hit_magic = True
                    break
                if valid_pkt(rx) and any(rx):
                    if magic_mode:
                        magic_mode = False
                        print(f"  t={sess.now():8.3f} lens left resync state")
                    if rx[2] & 0x80:
                        # b1 bit 0x10 on an ACK is the lens's busy/not-ready
                        # flag; the captured body repeats the same packet
                        # until the ack comes back clean
                        if rx[1] & 0x10 and retries < 8:
                            retries += 1
                            continue
                    else:
                        payload = rx
                j += 1
            if payload:
                for r in handle(payload):
                    if r not in issued:
                        issued.add(r)
                        requests.append(r)

        if focus_poll_left and not pending_focus:
            focus_poll_left -= 1
            settled = (focus_pos is not None
                       and focus_pos == focus_settle_prev)
            if settled or focus_poll_left == 0:
                print(f"  t={sess.now():8.3f} focus settled at "
                      f"{focus_pos if focus_pos is not None else 'unknown'} "
                      f"(target {focus_target})")
                focus_poll_left = 0
            focus_settle_prev = focus_pos

        if pending_iris and not hit_magic and not magic_mode:
            # control writes ride at the end of a burst, after the polls,
            # like the captured body's 0x15 drive sequences do
            pending_iris = False
            time.sleep(INTRA_BURST_GAP_S)
            staged = iris_setpoint(iris_target)
            print(f"  t={sess.now():8.3f} commanding iris to index "
                  f"{iris_target} ({staged.hex(' ')})")
            if command_iris(sess, iris_target):
                print(f"  t={sess.now():8.3f} iris command accepted")
                want_feedback = True
            else:
                print(f"  t={sess.now():8.3f} iris command not acknowledged")

        if pending_focus and not hit_magic and not magic_mode:
            pending_focus = False
            time.sleep(INTRA_BURST_GAP_S)
            print(f"  t={sess.now():8.3f} commanding focus to {focus_target} "
                  f"(from {focus_pos if focus_pos is not None else 'unknown'})")
            if command_focus(sess, focus_target, focus_pos):
                print(f"  t={sess.now():8.3f} focus command accepted")
                focus_poll_left = 12   # track position until it settles
                focus_settle_prev = None
            else:
                print(f"  t={sess.now():8.3f} focus command not acknowledged")

        if hit_magic:
            if not magic_mode:
                magic_mode = True
                print(f"  t={sess.now():8.3f} lens streaming resync marker "
                      "— transport lost")
            if resync_cooldown == 0:
                time.sleep(INTRA_BURST_GAP_S)
                if transport_reset(sess):
                    reset_fails = 0
                    if replay is not None:
                        # the dialogue leaves the lens in its post-reset
                        # state awaiting init; polling it there just
                        # re-markers it (the 402-reset loop in
                        # failed_at_about_3min.tsv) — re-init first
                        print(f"  t={sess.now():8.3f} transport reset "
                              "complete — re-initializing")
                        time.sleep(INTRA_BURST_GAP_S)
                        if run_startup(sess, replay):
                            magic_mode = False
                            sess.counter = next_counter_after(replay)
                        else:
                            resync_cooldown = 2
                        # run_startup reset the phase clock; realign
                        next_burst = sess.phase_now()
                        burst_i = 0
                    else:
                        print(f"  t={sess.now():8.3f} transport reset "
                              "complete — resuming polling")
                        magic_mode = False
                else:
                    reset_fails += 1
                    if reset_fails <= 2 or replay is None:
                        # give the lens a couple of bursts, then retry the
                        # dialogue before reaching for a full re-init
                        resync_cooldown = 2 if reset_fails <= 2 else 25
                    else:
                        reset_fails = 0
                        print(f"  t={sess.now():8.3f} marker persists — "
                              "re-running startup prefix")
                        time.sleep(INTRA_BURST_GAP_S)
                        ok = run_startup(sess, replay)
                        # run_startup reset the phase clock; realign the
                        # burst schedule EITHER WAY (a stale schedule after
                        # a failed re-init stalled attempt_6 for 2.1s)
                        next_burst = sess.phase_now()
                        burst_i = 0
                        if ok:
                            magic_mode = False
                            sess.counter = next_counter_after(replay)
                        else:
                            resync_cooldown = 25  # ~1s between attempts

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
    ap.add_argument("--replay-end", type=int, default=20,
                    help="replay only the first N captured transactions from startup_replay.json — the "
                         "capture's tail is state-dependent idle traffic that "
                         "desyncs a live lens (default 20, the end of the "
                         "deterministic config prefix; 0 = full capture)")
    ap.add_argument("--idle-seconds", type=float, default=10.0)
    ap.add_argument("--transcript", type=Path,
                    help="write Saleae-style TSV of the session (byte-accurate "
                         "tx/rx, analyzable with fuji_spi.py)")
    ap.add_argument("--power-gpio", type=int, default=17,
                    help="GPIO driving the external lens-power switch "
                         "(default 6; -1 disables). Raised before the settle "
                         "window so power->first-packet timing is "
                         "deterministic; keys 1/2 toggle it in-session")
    ap.add_argument("--no-ois-sync", action="store_true",
                    help="skip the 0x20 OIS-state packet the body sends after "
                         "identification (payload mirrors the lens's own OIS "
                         "switch)")
    ap.add_argument("--shutdown-timeout", type=float, default=1.5,
                    help="seconds to wait for the park to complete before "
                         "cutting power anyway (default 1.5)")
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
    kb = Keyboard()
    power = LensPower(-1 if args.dry_run else args.power_gpio)
    lens_up = False
    try:
        if power.enabled:
            print(f"lens power ON (GPIO{power.gpio} high); settle window "
                  "provides the boot delay")
            power.set(True)
        ok = run_startup_with_retry(sess, replay, args.settle,
                                    max(0, args.startup_retries),
                                    args.retry_delay,
                                    args.abort_txn or None,
                                    args.expect_ident or None)
        if not ok:
            print("aborting before idle loop (no lens response)")
            sys.exit(1)
        sess.counter = next_counter_after(replay)
        lens_up = True
        if not args.no_ois_sync:
            print(f"OIS sync: {sync_ois(sess)}")
        run_idle(sess, args.idle_seconds, replay, kb, power)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        kb.restore()
        # A lens that is still energized gets the body's park sequence before
        # the rail drops; power.state is False only if the session already
        # cut it with the '2' key (which parks on its way down).
        if lens_up and (power.state or not power.enabled):
            print("shutting the lens down...")
            try:
                run_shutdown(sess, timeout=args.shutdown_timeout)
            except Exception as exc:      # never block the power cut
                print(f"  shutdown sequence failed: {exc}")
        if power.enabled:
            print("lens power OFF (script exit)")
        power.close()
        sess.close()
        if args.transcript:
            print(f"transcript written to {args.transcript}")


if __name__ == "__main__":
    main()
