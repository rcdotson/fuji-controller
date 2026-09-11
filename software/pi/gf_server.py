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
    ->  SET POWER OFF       <-  OK POWER OFF     (accepted; the lens is
                                                  parked first and the rail
                                                  drops ~250ms later, at
                                                  EVT STATE OFF)
    ->  SET POWER OFF FORCE <-  OK POWER OFF     (cut the rail now, no park)
    ->  SET FOCUS -1200     <-  OK FOCUS -1200   (absolute motor counts,
                                                  signed 16-bit; move is async)
    ->  SET IRIS 7          <-  OK IRIS 7        (third-stop index, 1 = wide
                                                  open .. 22 = fully closed)
    ->  GET POWER           <-  OK POWER ON      (ON only once the lens is up)
    ->  GET FOCUS           <-  OK FOCUS -1198   (last position feedback)
    ->  GET IRIS            <-  OK IRIS 7        (last index feedback)
    ->  GET STATE           <-  OK STATE READY   (OFF|STARTING|READY|RESYNC
                                                  |STOPPING|FAULT)
    ->  PING                <-  OK PONG
    ->  HELP                <-  OK HELP SET POWER ON|OFF; ...

    ERR codes: SYNTAX (unparseable), RANGE (value out of bounds),
               NOT_READY (lens is off or still booting), NO_FEEDBACK (value
               not yet known), NO_POWER_GPIO (power switch unavailable).
    A command the lens itself refuses is accepted with OK and then reported
    as EVT ERROR <WHAT>_REJECTED, since acceptance is only known a burst later.

    EVT lines: STATE <name>, POWER ON|OFF, FOCUS <pos>, FOCUS_SETTLED <pos>,
               IRIS <index>, RING FOCUS|APERTURE <delta>,
               PARKED <focus> <iris>, RESYNC_POWER_CYCLE, ERROR <text>.
               Suppress them with --no-events.

Shutdown replays the body's own power-off sequence (startup_shutdown.txt):
OIS off, park on 0x28 channel 0x8001, wait for the status busy
bit to clear, read the final focus/iris back, then drop the rail. SIGTERM
takes the same path, so `systemctl stop gf-server` and a reboot park the lens
properly — the unit allows 15s for it.

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
DRIVE_HOLD_BURSTS = 25     # longest a staged drive waits for the lens to settle


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
    loop, RESYNC is READY while the lens is being coaxed back from the
    transport marker (see recover), and FAULT is the rail down after that
    recovery ladder ran out — only SET POWER ON leaves it.
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
        self.abort_shutdown = False

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
        self.drive_held = 0

        # burst bookkeeping (mirrors gf_controller.run_idle)
        self.burst_i = 0
        self.next_burst = 0.0
        self.last_status: str | None = None
        self.last_status_rx: bytes | None = None
        self.last_09: bytes | None = None
        self.magic_mode = False
        self.resync_cooldown = 0
        # recovery ladder: reset-dialogue failures, then re-init failures,
        # then power cycles. Each rung only escalates when the one below it
        # has genuinely run out of attempts.
        self.dialogue_fails = 0
        self.reinit_fails = 0
        self.power_cycles = 0

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
            self.reply("OK HELP SET POWER ON|OFF [FORCE]; "
                       "SET FOCUS <-32768..32767>; "
                       f"SET IRIS <{IRIS_MIN}..{IRIS_MAX}>; "
                       "GET POWER|FOCUS|IRIS|STATE; PING")
        elif verb == "GET" and len(parts) == 2:
            self.handle_get(parts[1])
        elif verb == "SET" and len(parts) == 3:
            self.handle_set(parts[1], parts[2])
        elif (verb == "SET" and len(parts) == 4
              and parts[1] == "POWER" and parts[3] == "FORCE"):
            self.set_power(parts[2], force=True)
        else:
            self.reply(f"ERR SYNTAX {line[:40]!r}")

    def handle_get(self, what: str) -> None:
        if what == "POWER":
            # ON from the moment the rail is raised, so a host polling after
            # SET POWER ON sees ON while the lens is still booting; STATE
            # distinguishes STARTING from READY
            self.reply(f"OK POWER {'OFF' if self.state in ('OFF', 'FAULT') else 'ON'}")
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
        if self.state in ("OFF", "STARTING", "STOPPING", "FAULT"):
            self.reply("ERR NOT_READY lens is "
                       + {"OFF": "off", "STARTING": "starting",
                          "STOPPING": "shutting down",
                          "FAULT": "in a fault state; SET POWER ON to retry"
                          }[self.state])
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

    def set_power(self, value: str, force: bool = False) -> None:
        """SET POWER ON|OFF, and the SET POWER OFF FORCE escape hatch.

        A plain OFF parks the lens first (the body's shutdown sequence) and
        drops the rail ~250ms later; FORCE cuts immediately, for a lens that
        is wedged or a host that needs the rail down now."""
        if value not in ("ON", "OFF"):
            self.reply(f"ERR SYNTAX POWER {value!r} (want ON or OFF)")
            return

        if value == "ON":
            if force:
                self.reply("ERR SYNTAX FORCE applies to SET POWER OFF only")
                return
            if not self.power.enabled:
                self.reply("ERR NO_POWER_GPIO lens power switch unavailable")
                return
            if self.state == "STOPPING":
                self.reply("ERR NOT_READY lens is shutting down")
                return
            if self.state not in ("OFF", "FAULT"):
                self.reply("OK POWER ON")   # already on or coming up
                return
            # FAULT means the rail is already down and the recovery ladder
            # gave up; an explicit SET POWER ON is the operator retrying, so
            # clear the ladder and start from cold
            self.dialogue_fails = self.reinit_fails = self.power_cycles = 0
            self.reply("OK POWER ON")
            self.abort_startup = False
            self.set_state("STARTING")
            self.event("POWER ON")
            return

        self.reply("OK POWER OFF")
        if force:
            if self.state not in ("OFF",):
                self.event("ERROR SHUTDOWN_SKIPPED forced")
            self.abort_shutdown = True      # cuts a park already in progress
            self.power_down()
        elif self.state == "READY":
            self.set_state("STOPPING")      # run() picks it up next iteration
        elif self.state == "STOPPING":
            pass                            # already parking
        elif self.state == "STARTING":
            self.abort_startup = True       # picked up between replay attempts
            self.event("ERROR SHUTDOWN_SKIPPED lens was still booting")
            self.power_down()
        elif self.state == "RESYNC":
            # no working transport to shut down over; the dialogue would only
            # answer with the resync marker
            self.event("ERROR SHUTDOWN_SKIPPED lens lost transport sync")
            self.power_down()
        else:
            self.power_down()               # already off

    def power_down(self) -> None:
        """Cut lens power and forget everything the lens told us: it reboots
        from scratch, so cached focus/iris feedback is no longer true."""
        self.power.set(False)
        self.sess.link.disarm()
        self.forget_lens_state()
        self.focus_target = None            # deliberate power off: no intent
        self.magic_mode = False
        self.resync_cooldown = 0
        # an operator-driven power down is a clean slate for the ladder too
        self.dialogue_fails = self.reinit_fails = self.power_cycles = 0
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

    def attempt_startup(self, attempts: int, reboot_first: bool = False,
                        label: str = "startup") -> bool | None:
        """Replay the startup prefix up to `attempts` times, dropping SCLK low
        for the lens's reboot window between tries.

        That SCLK low->high edge is the part that matters: a cold boot
        routinely fails the first replay and succeeds on the second, once the
        lens has had its ~1.34s reboot window. Recovery needs it exactly as
        much, so both callers come through here rather than calling
        gf.run_startup bare — which also kept expect_ident's bootloader-beacon
        rejection out of the recovery path.

        Returns True if the lens identified, False if it never did, and None
        if a SET POWER OFF or shutdown arrived and the caller should stand
        down. `reboot_first` gives the first attempt a reboot window too, for
        callers whose link is already armed (i.e. mid-session recovery).
        """
        args = self.args
        for attempt in range(1, attempts + 1):
            if attempt > 1 or reboot_first:
                print(f"{label} attempt {attempt} of {attempts}: SCLK low "
                      f"{args.retry_delay:.2f}s (lens reboot window)")
                self.sess.counter = 8
                self.sess.link.disarm()
                if not self.sleep_abortable(args.retry_delay):
                    return None
            self.sess.link.arm()
            if gf.run_startup(self.sess, self.replay, args.abort_txn or None,
                              args.expect_ident or None):
                self.sess.counter = gf.next_counter_after(self.replay)
                return True
            self.drain_commands()
            if self.abort_startup or self.stop.is_set():
                return None
        return False

    def do_startup(self) -> None:
        """Bring the lens up from cold: raise the rail, let it settle, then
        replay the body's power-on sequence (gf_controller.run_startup_with_retry,
        unrolled so commands are serviced and SET POWER OFF can abort between
        attempts)."""
        args = self.args
        print(f"lens power ON (GPIO{self.power.gpio}); settling "
              f"{args.settle:.2f}s with SCLK held low...")
        self.power.set(True)
        self.sess.link.disarm()
        if not self.sleep_abortable(args.settle):
            return

        ok = self.attempt_startup(max(0, args.startup_retries) + 1)
        if ok is None:
            return
        if ok:
            self.enter_idle()
            return

        print("WARNING: no lens identification — cutting power")
        self.event("ERROR STARTUP_FAILED no lens identification")
        self.power_down()

    def sync_ois(self) -> None:
        """Send the 0x20 the body sends after identification, with the payload
        matching the lens's own OIS switch (gf_controller.sync_ois)."""
        if self.args.no_ois_sync:
            return

        result = gf.sync_ois(self.sess)
        if result in ("on", "off"):
            print(f"  t={self.sess.now():8.3f} OIS {result} (switch position)")
            self.event(f"OIS {result.upper()}")
        elif result == "rejected":
            self.event("ERROR OIS_REJECTED 0x20 not acknowledged")
        else:
            self.event("ERROR OIS_UNKNOWN could not read the switch position")

    def actuators_busy(self) -> bool:
        """True while the last status said an actuator is moving or not yet
        ready — the window in which a fresh staged drive desyncs the lens."""
        rx = self.last_status_rx
        return rx is not None and (gf.status_busy(rx) or gf.status_disabled(rx))

    def enter_idle(self) -> None:
        """Hand over to the burst loop, asking for one round of feedback so
        GET FOCUS / GET IRIS answer as soon as the lens is up."""
        self.sync_ois()
        self.sess.start_phase()
        self.next_burst = gf.IDLE_PERIOD_S
        self.burst_i = 0
        self.want_iris_feedback = True
        self.focus_poll_left = RING_POLL_BURSTS
        self.focus_settle_prev = None
        # the lens is up: the recovery ladder starts from the bottom again
        self.dialogue_fails = self.reinit_fails = self.power_cycles = 0
        self.set_state("READY")

    # -- phase 3: shut the lens down ---------------------------------------

    def do_shutdown(self) -> None:
        """Run the body's park sequence, then drop the rail.

        Blocks for as long as the park takes (~250ms in the capture, bounded
        by --shutdown-timeout), but keeps servicing commands between status
        polls so GET still answers and SET POWER OFF FORCE can cut it short."""

        def on_poll() -> bool:
            self.drain_commands()
            return self.abort_shutdown

        self.abort_shutdown = False
        try:
            info = gf.run_shutdown(self.sess,
                                   timeout=self.args.shutdown_timeout,
                                   on_poll=on_poll)
        except Exception as exc:            # never leave the lens energized
            print(f"shutdown sequence raised: {exc}")
            self.event(f"ERROR SHUTDOWN_FAILED {exc}")
            self.power_down()
            return

        status = info["status"]
        problem = {
            "timeout": "SHUTDOWN_TIMEOUT park did not finish in "
                       f"{self.args.shutdown_timeout:.1f}s",
            "rejected": "SHUTDOWN_REJECTED lens did not accept the park",
            "unreachable": "SHUTDOWN_UNREACHABLE lens stopped answering",
            "aborted": "SHUTDOWN_ABORTED forced off mid-park",
        }.get(status)
        if problem:
            self.event(f"ERROR {problem}")
        # the readout runs on every path that reached it, so report whatever
        # positions came back even when the park itself went badly
        if info["focus"] is not None:
            self.event(f"FOCUS {info['focus']}")
        if info["iris"] is not None:
            self.event(f"IRIS {info['iris']}")
        if status == "ok":
            focus = "?" if info["focus"] is None else info["focus"]
            iris = "?" if info["iris"] is None else info["iris"]
            self.event(f"PARKED {focus} {iris}")
        self.power_down()

    def stop_lens(self) -> None:
        """Wind the lens down on the way out (SIGTERM, systemd stop, unwind).
        Always ends with the rail low."""
        if self.state in ("READY", "STOPPING"):
            print("parking the lens before exit")
            self.set_state("STOPPING")
            self.do_shutdown()
        else:
            if self.state == "RESYNC":
                self.event("ERROR SHUTDOWN_SKIPPED lens lost transport sync")
            elif self.state == "STARTING":
                self.event("ERROR SHUTDOWN_SKIPPED lens was still booting")
            self.power_down()

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
            self.last_status_rx = rx
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

        # settle tracking runs even with a command pending: a pending drive
        # now waits for the move to land, so freezing the countdown here
        # would mean it never lands and the drive never goes out
        if self.focus_poll_left:
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
        # A staged 0x15/0x18 drive started while the previous one is still
        # running is what desyncs the lens: it flags the sequence
        # (0d 08 95 5a), run_sequence aborts, and the half-open quad drops it
        # to the resync marker. SSAv2's AGL loop re-commands focus several
        # times a second, so hold the newest target and issue it on the first
        # burst after the move lands — latest target wins, nothing is dropped.
        moving = self.focus_poll_left > 0 or self.actuators_busy()
        if moving and (self.pending_focus or self.pending_iris):
            # ...but never hold one forever: a ring being turned by hand keeps
            # refreshing focus_poll_left, and the host's target still has to
            # land eventually
            self.drive_held += 1
            if self.drive_held > DRIVE_HOLD_BURSTS:
                print(f"  t={sess.now():8.3f} lens still busy after "
                      f"{self.drive_held} bursts — issuing the held command")
                moving = False
        else:
            self.drive_held = 0

        if self.pending_iris and quiet and not moving:
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
        elif self.pending_iris and not quiet:
            self.pending_iris = False
            self.event(f"ERROR IRIS_DROPPED {self.iris_target} lens resyncing")

        if self.pending_focus and quiet and not moving:
            self.pending_focus = False
            time.sleep(gf.INTRA_BURST_GAP_S)
            print(f"  t={sess.now():8.3f} commanding focus to "
                  f"{self.focus_target} (from {self.focus_pos})")
            if gf.command_focus(sess, self.focus_target, self.focus_pos):
                self.focus_poll_left = FOCUS_SETTLE_BURSTS
                self.focus_settle_prev = None
            else:
                self.event(f"ERROR FOCUS_REJECTED {self.focus_target}")
        elif self.pending_focus and not quiet:
            self.pending_focus = False
            self.event(f"ERROR FOCUS_DROPPED {self.focus_target} lens resyncing")

        if hit_magic:
            self.recover()

        self.burst_i += 1
        if self.resync_cooldown:
            self.resync_cooldown -= 1
        self.next_burst += gf.IDLE_PERIOD_S

    def recover(self) -> None:
        """Coax the lens back from the resync marker, escalating until it
        comes back or the ladder runs out.

        The rungs, in order, each tried --resync-retries times:

          1. the transport-reset dialogue (the lens lost transport sync)
          2. a re-init through attempt_startup, which drops SCLK low for the
             lens's reboot window first — the step a bare run_startup skips,
             and the one that makes a cold boot's second attempt succeed
          3. a power cycle, because the marker does not only mean "please
             reset the transport": decode_ident calls the same word the
             bootloader beacon, and a lens that has reset into its bootloader
             has no application to resync, so nothing on the bus reaches it
          4. STATE FAULT with the rail down, rather than hammering the bus
             forever — which is what the old ladder did, because reset_fails
             was zeroed on every dialogue "success" and never counted a
             failed re-init, so the escalation was unreachable.
        """
        sess = self.sess
        if not self.magic_mode:
            self.magic_mode = True
            print(f"  t={sess.now():8.3f} lens streaming resync marker "
                  "— transport lost")
            self.set_state("RESYNC")
        if self.resync_cooldown:
            return

        tries = max(1, self.args.resync_retries)
        time.sleep(gf.INTRA_BURST_GAP_S)

        # the dialogue rung also stands down once the re-inits it feeds have
        # run out: a dialogue that keeps reporting success while the re-init
        # keeps failing would otherwise loop here forever, which is the exact
        # shape of the bug this ladder replaces
        if self.dialogue_fails < tries and self.reinit_fails < tries:
            if gf.transport_reset(sess):
                self.dialogue_fails = 0
                # the dialogue leaves the lens awaiting init; polling it there
                # just re-markers it, so re-run the startup prefix first
                print(f"  t={sess.now():8.3f} transport reset complete "
                      "— re-initializing")
                time.sleep(gf.INTRA_BURST_GAP_S)
                if self.reinit(1):
                    return
                # give the lens a couple of bursts before the next rung
                self.resync_cooldown = 2
                return
            else:
                self.dialogue_fails += 1
                self.resync_cooldown = 2
                return

        # the dialogue is not getting us anywhere: re-initialize properly,
        # with the SCLK-low reboot window in front of each attempt
        if self.reinit_fails < tries:
            print(f"  t={sess.now():8.3f} marker persists — re-running the "
                  "startup prefix with a reboot window")
            if self.reinit(tries):
                return
            self.resync_cooldown = 25       # ~1s before the next rung
            return

        # nothing on the bus reaches it; do what the operator does by hand
        if self.power_cycles < max(0, self.args.max_power_cycles):
            self.power_cycles += 1
            if self.power_cycle():
                return
            self.resync_cooldown = 25
            return

        print(f"  t={sess.now():8.3f} lens unrecoverable after "
              f"{self.power_cycles} power cycle(s) — cutting power")
        self.event("ERROR RESYNC_FAILED lens did not come back")
        self.power.set(False)
        self.sess.link.disarm()
        self.forget_lens_state()
        self.set_state("FAULT")

    def reinit(self, attempts: int) -> bool:
        """Re-run the startup prefix mid-session and, if the lens identifies,
        put the burst loop back to work. Counts its own failures so the
        ladder in recover() can move on."""
        ok = self.attempt_startup(attempts, reboot_first=True, label="re-init")
        # attempt_startup reset the phase clock either way; realign the
        # schedule now, or a stale one stalls the next burst for seconds
        self.next_burst = self.sess.phase_now()
        self.burst_i = 0
        if ok:
            self.magic_mode = False
            self.dialogue_fails = self.reinit_fails = 0
            self.enter_idle()               # also re-sends the 0x20 OIS state
            self.restore_targets()
            return True
        if ok is None:
            return True                     # aborted by the host; stop here
        self.reinit_fails += 1
        return False

    def power_cycle(self) -> bool:
        """Drop the rail, wait, and bring the lens up from cold — the recovery
        the operator does by hand when the lens will not resync.

        True if the lens came back or the host took the lens off us mid-cycle;
        False only if the cycle genuinely failed to revive it."""
        args = self.args
        if not self.power.enabled:
            self.event("ERROR RESYNC_NO_POWER_GPIO cannot power cycle")
            return False
        print(f"  t={self.sess.now():8.3f} power cycling the lens "
              f"({self.power_cycles} of {args.max_power_cycles}) — rail down "
              f"for {args.power_cycle_delay:.1f}s")
        self.event("RESYNC_POWER_CYCLE")

        self.power.set(False)
        self.sess.link.disarm()
        self.forget_lens_state()
        if not self.sleep_abortable(args.power_cycle_delay) or self.handed_off():
            return True                     # host cut in; it owns the lens now

        print(f"lens power ON (GPIO{self.power.gpio}); settling "
              f"{args.settle:.2f}s with SCLK held low...")
        self.power.set(True)
        if not self.sleep_abortable(args.settle) or self.handed_off():
            return True

        ok = self.attempt_startup(max(0, args.startup_retries) + 1,
                                  label="post-power-cycle startup")
        self.next_burst = self.sess.phase_now()
        self.burst_i = 0
        if ok:
            self.magic_mode = False
            self.dialogue_fails = self.reinit_fails = 0
            self.enter_idle()
            self.restore_targets()
            return True
        return ok is None                   # None = aborted, not a failure

    def handed_off(self) -> bool:
        """True once a command has taken the lens out of recovery — SET POWER
        OFF while we were in RESYNC drops the rail and goes to OFF, and the
        cycle must not quietly power it back up underneath that."""
        return self.state not in ("RESYNC", "READY") or self.stop.is_set()

    def forget_lens_state(self) -> None:
        """Drop everything the lens told us. It reboots from scratch across a
        power cycle, so cached focus/iris feedback is no longer true.

        The host's *targets* are intent, not feedback, so they survive — see
        restore_targets, which puts the lens back where it was asked to be
        once it comes back."""
        self.focus_pos = self.iris_index = None
        self.pending_focus = self.pending_iris = False
        self.want_iris_feedback = False
        self.focus_poll_left = 0
        self.focus_settle_prev = None
        self.drive_held = 0
        self.last_status = self.last_09 = self.last_status_rx = None

    def restore_targets(self) -> None:
        """Re-apply the host's last focus/iris after the lens rebooted.

        A recovered lens comes up at its own defaults, and the host will not
        necessarily re-send: SSAv2's AGL loop suppresses a repeat command
        through its own deadband, so without this the lens would sit at the
        wrong focus and nothing upstream would notice."""
        if self.focus_target is not None:
            self.pending_focus = True
        self.pending_iris = True
        print(f"  t={self.sess.now():8.3f} restoring focus "
              f"{self.focus_target} / iris {self.iris_target} after the reboot")

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
            elif self.state == "STOPPING":
                self.do_shutdown()
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
    ap.add_argument("--replay-end", type=int, default=20,
                    help="replay only the first N captured transactions "
                         "(default 20, the deterministic config prefix)")
    ap.add_argument("--bus", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0,
                    help="bus silence after lens power-on (default 0)")
    ap.add_argument("--startup-retries", type=int, default=5)
    ap.add_argument("--retry-delay", type=float, default=1.4)
    ap.add_argument("--resync-retries", type=int, default=2,
                    help="reset-dialogue attempts before re-initializing, and "
                         "re-init attempts before power cycling (default 2)")
    ap.add_argument("--power-cycle-delay", type=float, default=3.0,
                    help="seconds the rail stays down in an automatic resync "
                         "power cycle (default 3.0)")
    ap.add_argument("--max-power-cycles", type=int, default=2,
                    help="automatic power cycles before giving up and going "
                         "to STATE FAULT (default 2)")
    ap.add_argument("--abort-txn", type=int, default=20,
                    help="abort a startup attempt if no identification by "
                         "this transaction (0 disables)")
    ap.add_argument("--expect-ident", default="",
                    help="substring a real identification must contain")
    ap.add_argument("--power-gpio", type=int, default=17,
                    help="GPIO driving the lens-power switch (default 17)")
    ap.add_argument("--no-ois-sync", action="store_true",
                    help="skip the 0x20 OIS-state packet after startup")
    ap.add_argument("--shutdown-timeout", type=float, default=0.5,
                    help="seconds to wait for the park to complete before "
                         "cutting power anyway (default 0.5)")
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
        print("server exiting")
        server.stop_lens()
        power.close()
        sess.close()
        port.close()


if __name__ == "__main__":
    main()
