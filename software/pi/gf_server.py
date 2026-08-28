#!/usr/bin/env python3
"""Headless lens server: drives a G-mount lens over SPI, takes orders over UART.

Runs unattended on the Pi (systemd, see gf-server.service). On start it claims
the SPI bus and the lens-power GPIO but leaves the lens POWERED OFF and
un-initialized — nothing is driven until a `SET POWER ON` arrives on the serial
link. Power-on runs the same startup replay + idle engine as gf_controller.py
(imported from it, so the protocol logic has one home); the idle loop then runs
continuously, servicing serial commands at burst boundaries and emitting
unsolicited status events as the lens reports them.

Serial link (default /dev/serial0, 115200 8N1, no flow control):

    ASCII lines, terminated with LF or CRLF, case-insensitive, <= 128 chars.
    Every command produces exactly one reply line, `OK ...` or `ERR ...`.
    Unsolicited `EVT ...` lines may appear at any time between replies.

    ->  SET POWER ON        <-  OK POWER ON      (accepted; boot is async,
                                                  wait for EVT STATE READY)
    ->  SET POWER OFF       <-  OK POWER OFF
    ->  SET FOCUS -1200     <-  OK FOCUS -1200   (absolute motor counts,
                                                  signed 16-bit; move is async)
    ->  SET IRIS 7          <-  OK IRIS 7        (third-stop index, 1 = wide
                                                  open .. 22 = fully closed)
    ->  GET POWER           <-  OK POWER ON      (ON only once the lens is up)
    ->  GET FOCUS           <-  OK FOCUS -1198   (last position feedback)
    ->  GET IRIS            <-  OK IRIS 7        (last index feedback)
    ->  GET STATE           <-  OK STATE READY   (OFF|STARTING|READY|RESYNC)
    ->  PING                <-  OK PONG
    ->  HELP                <-  OK HELP SET POWER ON|OFF; ...

    ERR codes: SYNTAX (unparseable), RANGE (value out of bounds),
               NOT_READY (lens is off or still booting), NO_FEEDBACK (value
               not yet known), NO_POWER_GPIO (power switch unavailable).
    A command the lens itself refuses is accepted with OK and then reported
    as EVT ERROR <WHAT>_REJECTED, since acceptance is only known a burst later.

    EVT lines: STATE <name>, POWER ON|OFF, FOCUS <pos>, FOCUS_SETTLED <pos>,
               IRIS <index>, RING FOCUS|APERTURE <delta>, ERROR <text>.
               Suppress them with --no-events.

Wiring and lens-power notes: see gf_controller.py's module docstring. The UART
adds:

    Pi GPIO14 (TXD, phys pin 8)  -> host RX
    Pi GPIO15 (RXD, phys pin 10) -> host TX
    Pi GND    (phys pin 6)       -> host GND

/dev/serial0 is the primary UART; free it from the login console first
(`sudo raspi-config` -> Interface -> Serial: login shell NO, hardware YES, or
drop `console=serial0,115200` from /boot/firmware/cmdline.txt and add
`enable_uart=1` to config.txt). Needs python3-serial, python3-spidev,
python3-lgpio.

Usage:
    python3 gf_server.py                       # /dev/serial0 @ 115200
    python3 gf_server.py --port - --dry-run    # protocol test on stdin/stdout
    python3 gf_server.py --baud 9600 --no-events
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gf_controller as gf  # noqa: E402 -- needs the path fix above

IRIS_MIN, IRIS_MAX = 1, 22
FOCUS_MIN, FOCUS_MAX = -32768, 32767
MAX_LINE = 128
FOCUS_SETTLE_BURSTS = 12   # position polls granted after a commanded move
RING_POLL_BURSTS = 3       # position polls after the ring is turned by hand


# ---------------------------------------------------------------------------
# Command transport: the UART, or stdin/stdout for bench testing
# ---------------------------------------------------------------------------

class CommandPort:
    """Line-oriented command link. Writes are serialized with a lock so the
    reader thread and the engine can both emit without interleaving."""

    def __init__(self, port: str, baud: int):
        self.lock = threading.Lock()
        self.ser = None
        self.stdio = port == "-"
        if self.stdio:
            # protocol lines own the real stdout; every diagnostic print in
            # this process (ours and gf_controller's) is pushed to stderr
            self.out = sys.stdout
            sys.stdout = sys.stderr
            return
        import serial  # noqa: PLC0415 -- only needed with a real UART
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def write_line(self, text: str) -> None:
        with self.lock:
            try:
                if self.ser is not None:
                    self.ser.write((text + "\r\n").encode("ascii", "replace"))
                else:
                    self.out.write(text + "\n")
                    self.out.flush()
            except OSError as exc:
                print(f"serial write failed: {exc}")

    def read_loop(self, q: "queue.Queue[str]", stop: threading.Event) -> None:
        """Feed complete lines into q until stop is set (daemon thread)."""
        buf = bytearray()
        dropping = False   # swallowing the tail of an overlong line
        while not stop.is_set():
            try:
                if self.ser is None:
                    line = sys.stdin.readline()
                    if not line:          # EOF on a piped stdin: done taking
                        stop.set()        # orders, ask the engine to wind up
                        return
                    q.put(line.strip())
                    continue
                chunk = self.ser.read(1)
                if not chunk:
                    continue
                pending = self.ser.in_waiting
                if pending:
                    chunk += self.ser.read(pending)
            except OSError as exc:
                print(f"serial read failed: {exc}")
                time.sleep(0.5)
                continue
            buf.extend(chunk)
            while b"\n" in buf:
                raw, _, rest = bytes(buf).partition(b"\n")
                buf = bytearray(rest)
                if dropping:              # tail of a line already rejected
                    dropping = False
                    continue
                q.put(raw.decode("ascii", "replace").strip())
            if len(buf) > MAX_LINE:       # noise or a runaway sender: drop it
                buf.clear()
                if not dropping:
                    dropping = True
                    q.put("\x00overlong")

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()


class NullPower:
    """Stand-in for LensPower under --dry-run: reports enabled and remembers
    the requested state so the full state machine can be exercised off-Pi."""

    gpio = -1
    enabled = True

    def __init__(self) -> None:
        self.state = False

    def set(self, on: bool) -> None:
        self.state = on

    def close(self) -> None:
        self.state = False


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class LensServer:
    """Owns the SPI session, the lens power pin, and the command queue.

    The engine is a state machine stepped from run(): OFF idles cheaply,
    STARTING replays the body's power-on sequence (abortable by SET POWER
    OFF between attempts), READY runs gf_controller's request-quad burst
    loop, and RESYNC is READY while the lens is being coaxed back from the
    transport marker.
    """

    def __init__(self, sess: gf.BodySession, power: gf.LensPower,
                 replay: list[dict], port: CommandPort, args) -> None:
        self.sess = sess
        self.power = power
        self.replay = replay
        self.port = port
        self.args = args
        self.events = not args.no_events
        self.commands: "queue.Queue[str]" = queue.Queue()
        self.stop = threading.Event()

        self.state = "OFF"
        self.abort_startup = False

        # lens state, all unknown until feedback arrives
        self.focus_pos: int | None = None
        self.focus_target: int | None = None
        self.iris_index: int | None = None
        self.iris_target: int = 1
        self.pending_focus = False
        self.pending_iris = False
        self.want_iris_feedback = False
        self.focus_poll_left = 0
        self.focus_settle_prev: int | None = None

        # burst bookkeeping (mirrors gf_controller.run_idle)
        self.burst_i = 0
        self.next_burst = 0.0
        self.last_status: str | None = None
        self.last_09: bytes | None = None
        self.magic_mode = False
        self.resync_cooldown = 0
        self.reset_fails = 0

    # -- output ------------------------------------------------------------

    def reply(self, text: str) -> None:
        self.port.write_line(text)

    def event(self, text: str) -> None:
        if self.events:
            self.port.write_line(f"EVT {text}")

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            print(f"  t={self.sess.now():8.3f} state -> {state}")
            self.event(f"STATE {state}")

    # -- command handling --------------------------------------------------

    def drain_commands(self) -> None:
        while True:
            try:
                line = self.commands.get_nowait()
            except queue.Empty:
                return
            self.handle(line)

    def handle(self, line: str) -> None:
        if not line:
            return
        if line.startswith("\x00"):
            self.reply("ERR SYNTAX line too long")
            return
        parts = line.upper().split()
        if not parts:
            return
        verb = parts[0]

        if verb == "PING":
            self.reply("OK PONG")
        elif verb == "HELP":
            self.reply("OK HELP SET POWER ON|OFF; SET FOCUS <-32768..32767>; "
                       f"SET IRIS <{IRIS_MIN}..{IRIS_MAX}>; "
                       "GET POWER|FOCUS|IRIS|STATE; PING")
        elif verb == "GET" and len(parts) == 2:
            self.handle_get(parts[1])
        elif verb == "SET" and len(parts) == 3:
            self.handle_set(parts[1], parts[2])
        else:
            self.reply(f"ERR SYNTAX {line[:40]!r}")

    def handle_get(self, what: str) -> None:
        if what == "POWER":
            # ON from the moment the rail is raised, so a host polling after
            # SET POWER ON sees ON while the lens is still booting; STATE
            # distinguishes STARTING from READY
            self.reply(f"OK POWER {'OFF' if self.state == 'OFF' else 'ON'}")
        elif what == "STATE":
            self.reply(f"OK STATE {self.state}")
        elif what == "FOCUS":
            if self.focus_pos is None:
                self.reply("ERR NO_FEEDBACK focus position unknown")
            else:
                self.reply(f"OK FOCUS {self.focus_pos}")
        elif what == "IRIS":
            if self.iris_index is None:
                self.reply("ERR NO_FEEDBACK iris index unknown")
            else:
                self.reply(f"OK IRIS {self.iris_index}")
        else:
            self.reply(f"ERR SYNTAX unknown GET {what}")

    def handle_set(self, what: str, value: str) -> None:
        if what == "POWER":
            self.set_power(value)
            return
        if what not in ("FOCUS", "IRIS"):
            self.reply(f"ERR SYNTAX unknown SET {what}")
            return
        try:
            n = int(value, 0)
        except ValueError:
            self.reply(f"ERR SYNTAX {value!r} is not an integer")
            return
        if self.state in ("OFF", "STARTING"):
            self.reply("ERR NOT_READY lens is "
                       + ("off" if self.state == "OFF" else "starting"))
            return
        if what == "FOCUS":
            if not FOCUS_MIN <= n <= FOCUS_MAX:
                self.reply(f"ERR RANGE focus {FOCUS_MIN}..{FOCUS_MAX}")
                return
            self.focus_target = n
            self.pending_focus = True
            self.reply(f"OK FOCUS {n}")
        else:
            if not IRIS_MIN <= n <= IRIS_MAX:
                self.reply(f"ERR RANGE iris {IRIS_MIN}..{IRIS_MAX}")
                return
            self.iris_target = n
            self.pending_iris = True
            self.reply(f"OK IRIS {n}")

    def set_power(self, value: str) -> None:
        if value not in ("ON", "OFF"):
            self.reply(f"ERR SYNTAX POWER {value!r} (want ON or OFF)")
            return
        if value == "ON":
            if not self.power.enabled:
                self.reply("ERR NO_POWER_GPIO lens power switch unavailable")
                return
            if self.state != "OFF":
                self.reply("OK POWER ON")   # already on or coming up
                return
            self.reply("OK POWER ON")
            self.abort_startup = False
            self.set_state("STARTING")
            self.event("POWER ON")
        else:
            self.reply("OK POWER OFF")
            if self.state == "STARTING":
                self.abort_startup = True   # picked up between replay attempts
            self.power_down()

    def power_down(self) -> None:
        """Cut lens power and forget everything the lens told us: it reboots
        from scratch, so cached focus/iris feedback is no longer true."""
        self.power.set(False)
        self.sess.link.disarm()
        self.focus_pos = self.focus_target = self.iris_index = None
        self.pending_focus = self.pending_iris = False
        self.want_iris_feedback = False
        self.focus_poll_left = 0
        self.last_status = self.last_09 = None
        self.magic_mode = False
        self.resync_cooldown = self.reset_fails = 0
        if self.state != "OFF":
            self.event("POWER OFF")
        self.set_state("OFF")

    # -- phase 1: bring the lens up ----------------------------------------

    def sleep_abortable(self, seconds: float) -> bool:
        """Sleep in slices, servicing commands; False if power-off arrived."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.drain_commands()
            if self.abort_startup or self.stop.is_set():
                return False
            time.sleep(min(0.02, max(0.0, end - time.perf_counter())))
        return True

    def do_startup(self) -> None:
        """Replay the body's power-on sequence, retrying on the lens's reboot
        cadence (gf_controller.run_startup_with_retry, unrolled so commands
        are serviced and SET POWER OFF can abort between attempts)."""
        args = self.args
        print(f"lens power ON (GPIO{self.power.gpio}); settling "
              f"{args.settle:.2f}s with SCLK held low...")
        self.power.set(True)
        self.sess.link.disarm()
        if not self.sleep_abortable(args.settle):
            return

        attempts = max(0, args.startup_retries) + 1
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                print(f"startup attempt {attempt} of {attempts}: SCLK low "
                      f"{args.retry_delay:.2f}s (lens reboot window)")
                self.sess.counter = 8
                self.sess.link.disarm()
                if not self.sleep_abortable(args.retry_delay):
                    return
            self.sess.link.arm()
            if gf.run_startup(self.sess, self.replay, args.abort_txn or None,
                              args.expect_ident or None):
                self.sess.counter = gf.next_counter_after(self.replay)
                self.enter_idle()
                return
            self.drain_commands()
            if self.abort_startup or self.stop.is_set():
                return

        print("WARNING: no lens identification — cutting power")
        self.event("ERROR STARTUP_FAILED no lens identification")
        self.power_down()

    def enter_idle(self) -> None:
        """Hand over to the burst loop, asking for one round of feedback so
        GET FOCUS / GET IRIS answer as soon as the lens is up."""
        self.sess.start_phase()
        self.next_burst = gf.IDLE_PERIOD_S
        self.burst_i = 0
        self.want_iris_feedback = True
        self.focus_poll_left = RING_POLL_BURSTS
        self.focus_settle_prev = None
        self.set_state("READY")

    # -- phase 2: idle bursts ----------------------------------------------

    def decode(self, rx: bytes) -> list[bytes]:
        """Decode one lens payload, emit any state change, and return the
        follow-up requests it asks for (same contract as run_idle.handle)."""
        cmd = rx[2] & 0x7F
        reqs: list[bytes] = []
        if cmd == 0x08:
            tag = rx[3] >> 6
            if tag == 2:                       # iris state feedback
                index = rx[0] & 0x1F
                if index != self.iris_index:
                    self.iris_index = index
                    self.event(f"IRIS {index}")
                print(f"  t={self.sess.now():8.3f} iris state: index={index} "
                      f"flags={rx[0] >> 5:03b} (raw {rx.hex(' ')})")
                return reqs
            if tag == 1:                       # focus position feedback
                pos = int.from_bytes(rx[:2], "big", signed=True)
                if pos != 32767 and pos != self.focus_pos:
                    self.focus_pos = pos
                    self.event(f"FOCUS {pos}")
                    print(f"  t={self.sess.now():8.3f} focus position: {pos}")
                return reqs
            status = gf.describe_status(rx)
            if status and status != self.last_status:
                print(f"  t={self.sess.now():8.3f} lens status: {status}")
                self.last_status = status
            pend = rx[1] & 0x7F
            if pend & 0x08:
                reqs.append(gf.FOCUS_POLL)
            if pend & 0x10:
                reqs.append(gf.APERTURE_POLL)
        elif cmd == 0x0C:                      # ring rotation readout
            kind = {2: "FOCUS", 0: "APERTURE"}.get(rx[3] >> 6, "UNKNOWN")
            delta = int.from_bytes(rx[:2], "big", signed=True)
            self.event(f"RING {kind} {delta:+d}")
            # the lens services its own rings, so chase the new value to keep
            # GET FOCUS / GET IRIS honest
            if kind == "FOCUS":
                self.focus_poll_left = max(self.focus_poll_left,
                                           RING_POLL_BURSTS)
                # require a fresh reading before calling it settled, or the
                # stale pre-turn position would match and end the polling
                self.focus_settle_prev = None
            elif kind == "APERTURE":
                self.want_iris_feedback = True
        elif cmd == 0x09:
            if rx[:2] != self.last_09:
                print(f"  t={self.sess.now():8.3f} 0x09 value changed: "
                      f"b0={rx[0]:02x} b1={rx[1]:02x}")
                self.last_09 = rx[:2]
        elif cmd == 0x03:
            print(f"  t={self.sess.now():8.3f} lens error/desync report "
                  f"({rx.hex(' ')}, code {rx[1]:02x})")
            self.event(f"ERROR LENS_REPORT {rx.hex()}")
        return reqs

    def run_quads(self, requests: list[bytes]) -> bool:
        """Run one burst's worth of request quads. True if the lens dropped
        to the resync marker mid-burst.

        Quad shape (the invariant framing the real body uses for every
        request): request, transport(n), idle, ack-of-the-response.
        """
        sess = self.sess
        issued: set[bytes] = set(requests)
        first_slot = True
        quads = 0
        while requests and quads < 5:
            req = requests.pop(0)
            quads += 1
            payload = None
            plan = [req, gf.transport(sess.next_counter()), gf.IDLE_PKT, None]
            retries = 0
            j = 0
            while j < len(plan):
                tx = plan[j]
                if tx is None:
                    tx = gf.ack_for(payload) if payload else gf.IDLE_PKT
                if not first_slot:
                    time.sleep(gf.INTRA_BURST_GAP_S)
                first_slot = False
                rx = sess.xfer(tx)
                if rx == gf.MAGIC_WORD:
                    return True
                if gf.valid_pkt(rx) and any(rx):
                    if self.magic_mode:
                        self.magic_mode = False
                        print(f"  t={sess.now():8.3f} lens left resync state")
                        self.set_state("READY")
                    if rx[2] & 0x80:
                        # busy flag on an ack: repeat the packet, as the
                        # captured body does, until it comes back clean
                        if rx[1] & 0x10 and retries < 8:
                            retries += 1
                            continue
                    else:
                        payload = rx
                j += 1
            if payload:
                for r in self.decode(payload):
                    if r not in issued:
                        issued.add(r)
                        requests.append(r)
        return False

    def burst(self) -> None:
        sess = self.sess
        sess.wait_until(self.next_burst)
        self.drain_commands()
        if self.state == "OFF" or self.stop.is_set():
            return

        requests = [gf.STATUS_POLL]
        if self.burst_i % 3 == 0:
            requests.append(gf.POLL_09)
        if self.want_iris_feedback:
            requests.append(gf.IRIS_FEEDBACK)
            self.want_iris_feedback = False
        if self.focus_poll_left:
            requests.append(gf.FOCUS_POS_POLL)

        hit_magic = self.run_quads(requests)

        if self.focus_poll_left and not self.pending_focus:
            self.focus_poll_left -= 1
            settled = (self.focus_pos is not None
                       and self.focus_pos == self.focus_settle_prev)
            if settled or self.focus_poll_left == 0:
                self.focus_poll_left = 0
                if self.focus_pos is not None:
                    print(f"  t={sess.now():8.3f} focus settled at "
                          f"{self.focus_pos} (target {self.focus_target})")
                    self.event(f"FOCUS_SETTLED {self.focus_pos}")
            self.focus_settle_prev = self.focus_pos

        quiet = not hit_magic and not self.magic_mode
        if self.pending_iris and quiet:
            # control writes ride at the end of a burst, after the polls,
            # like the captured body's drive sequences do
            self.pending_iris = False
            time.sleep(gf.INTRA_BURST_GAP_S)
            print(f"  t={sess.now():8.3f} commanding iris index "
                  f"{self.iris_target}")
            if gf.command_iris(sess, self.iris_target):
                self.want_iris_feedback = True
            else:
                self.event(f"ERROR IRIS_REJECTED {self.iris_target}")
        elif self.pending_iris:
            self.pending_iris = False
            self.event(f"ERROR IRIS_DROPPED {self.iris_target} lens resyncing")

        if self.pending_focus and quiet:
            self.pending_focus = False
            time.sleep(gf.INTRA_BURST_GAP_S)
            print(f"  t={sess.now():8.3f} commanding focus to "
                  f"{self.focus_target} (from {self.focus_pos})")
            if gf.command_focus(sess, self.focus_target, self.focus_pos):
                self.focus_poll_left = FOCUS_SETTLE_BURSTS
                self.focus_settle_prev = None
            else:
                self.event(f"ERROR FOCUS_REJECTED {self.focus_target}")
        elif self.pending_focus:
            self.pending_focus = False
            self.event(f"ERROR FOCUS_DROPPED {self.focus_target} lens resyncing")

        if hit_magic:
            self.recover()

        self.burst_i += 1
        if self.resync_cooldown:
            self.resync_cooldown -= 1
        self.next_burst += gf.IDLE_PERIOD_S

    def recover(self) -> None:
        """Marker recovery, per run_idle: complete the transport-reset
        dialogue, re-initialize, and escalate to the startup prefix if the
        marker persists."""
        sess = self.sess
        if not self.magic_mode:
            self.magic_mode = True
            print(f"  t={sess.now():8.3f} lens streaming resync marker "
                  "— transport lost")
            self.set_state("RESYNC")
        if self.resync_cooldown:
            return
        time.sleep(gf.INTRA_BURST_GAP_S)
        if gf.transport_reset(sess):
            self.reset_fails = 0
            # the dialogue leaves the lens awaiting init; polling it there
            # just re-markers it, so re-run the startup prefix first
            print(f"  t={sess.now():8.3f} transport reset complete "
                  "— re-initializing")
            time.sleep(gf.INTRA_BURST_GAP_S)
            if gf.run_startup(sess, self.replay):
                self.magic_mode = False
                sess.counter = gf.next_counter_after(self.replay)
                self.set_state("READY")
            else:
                self.resync_cooldown = 2
        else:
            self.reset_fails += 1
            if self.reset_fails <= 2:
                self.resync_cooldown = 2
            else:
                self.reset_fails = 0
                print(f"  t={sess.now():8.3f} marker persists — re-running "
                      "startup prefix")
                time.sleep(gf.INTRA_BURST_GAP_S)
                if gf.run_startup(sess, self.replay):
                    self.magic_mode = False
                    sess.counter = gf.next_counter_after(self.replay)
                    self.set_state("READY")
                else:
                    self.resync_cooldown = 25   # ~1s between attempts
        # run_startup reset the phase clock either way; realign the schedule
        self.next_burst = sess.phase_now()
        self.burst_i = 0

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:
        # banner + initial state, so a host can tell the Pi rebooted
        self.event("READY gf-server 1.0")
        self.event("STATE OFF")
        print("server ready — lens powered off, waiting for commands")
        while not self.stop.is_set():
            self.drain_commands()
            if self.stop.is_set():
                break
            if self.state == "STARTING":
                self.do_startup()
            elif self.state in ("READY", "RESYNC"):
                self.burst()
            else:
                time.sleep(0.02)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/serial0",
                    help="command UART (default /dev/serial0; '-' = stdio)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--replay", type=Path,
                    default=Path(__file__).parent / "startup_replay.json")
    ap.add_argument("--replay-end", type=int, default=64,
                    help="replay only the first N captured transactions "
                         "(default 64, the deterministic config prefix)")
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--settle", type=float, default=1.4,
                    help="bus silence after lens power-on (default 1.4)")
    ap.add_argument("--startup-retries", type=int, default=5)
    ap.add_argument("--retry-delay", type=float, default=1.4)
    ap.add_argument("--abort-txn", type=int, default=20,
                    help="abort a startup attempt if no identification by "
                         "this transaction (0 disables)")
    ap.add_argument("--expect-ident", default="",
                    help="substring a real identification must contain")
    ap.add_argument("--power-gpio", type=int, default=17,
                    help="GPIO driving the lens-power switch (default 17)")
    ap.add_argument("--no-events", action="store_true",
                    help="reply to commands only; no unsolicited EVT lines")
    ap.add_argument("--transcript", type=Path,
                    help="write a Saleae-style TSV of the SPI session")
    ap.add_argument("--dry-run", action="store_true",
                    help="no SPI/GPIO hardware; exercise the serial protocol")
    args = ap.parse_args()

    replay = json.loads(args.replay.read_text())["transactions"]
    if args.replay_end > 0:
        replay = replay[:args.replay_end]

    port = CommandPort(args.port, args.baud)
    link = gf.SpiLink(args.bus, args.device, args.dry_run)
    sess = gf.BodySession(link, args.transcript)
    power = NullPower() if args.dry_run else gf.LensPower(args.power_gpio)
    server = LensServer(sess, power, replay, port, args)

    reader = threading.Thread(target=port.read_loop,
                              args=(server.commands, server.stop), daemon=True)
    reader.start()

    def shutdown(_sig, _frame):
        server.stop.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        print("shutting down — cutting lens power")
        server.power_down()
        power.close()
        sess.close()
        port.close()


if __name__ == "__main__":
    main()
