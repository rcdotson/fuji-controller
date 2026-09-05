# Pi lens controller

Two programs share one protocol implementation:

- **`gf_controller.py`** — interactive bench tool. Powers the lens, replays the
  body's startup sequence, then runs a keyboard-driven idle session. All the
  packet building, startup replay, idle-burst engine and resync recovery lives
  here.
- **`gf_server.py`** — headless server for unattended use. Imports the engine
  from `gf_controller.py` and drives it from an ASCII command link on
  `/dev/serial0` instead of the keyboard. **Starts with the lens powered off
  and un-initialized**; nothing is driven until a `SET POWER ON` arrives.

## Serial protocol

115200 8N1, no flow control. Lines terminated with LF or CRLF, case
insensitive, 128 characters max. Every command gets exactly one reply line.

| Command | Reply | Notes |
| --- | --- | --- |
| `SET POWER ON` | `OK POWER ON` | Accepted, not complete — the rail comes up, then the startup replay runs (~2–10 s). Wait for `EVT STATE READY`. |
| `SET POWER OFF` | `OK POWER OFF` | Cuts the rail immediately, aborting a startup in progress. Cached focus/iris feedback is discarded (the lens reboots). |
| `SET FOCUS <n>` | `OK FOCUS <n>` | Absolute motor position, signed 16-bit (−32768…32767). The move is asynchronous — track `EVT FOCUS` / `EVT FOCUS_SETTLED`. |
| `SET IRIS <n>` | `OK IRIS <n>` | Third-stop index, 1 = wide open … 22 = fully closed. Confirmed by `EVT IRIS <n>` from the lens's own feedback poll. |
| `GET POWER` | `OK POWER ON`\|`OFF` | `ON` from the moment the rail is raised, including while booting. |
| `GET FOCUS` | `OK FOCUS <pos>` | Last position feedback. `ERR NO_FEEDBACK` before the first reading. |
| `GET IRIS` | `OK IRIS <index>` | Last index feedback. `ERR NO_FEEDBACK` before the first reading. |
| `GET STATE` | `OK STATE <name>` | `OFF`, `STARTING`, `READY` or `RESYNC`. |
| `PING` | `OK PONG` | |
| `HELP` | `OK HELP ...` | One-line grammar reminder. |

Error replies are `ERR <CODE> <detail>`, with `CODE` one of `SYNTAX` (garbled
or unknown command), `RANGE` (value out of bounds), `NOT_READY` (lens off or
still booting), `NO_FEEDBACK` (value not yet known) and `NO_POWER_GPIO` (the
power switch could not be claimed).

Whether the *lens* accepts a `SET FOCUS`/`SET IRIS` is only known a burst
later, so those are answered `OK` on validation and a refusal arrives
afterwards as `EVT ERROR FOCUS_REJECTED <n>` / `EVT ERROR IRIS_REJECTED <n>`.

Unsolicited events (suppress with `--no-events`):

```
EVT READY gf-server 1.0        server booted, lens off
EVT STATE OFF|STARTING|READY|RESYNC
EVT POWER ON|OFF
EVT FOCUS <pos>                position changed (commanded move or ring)
EVT FOCUS_SETTLED <pos>        motor stopped
EVT IRIS <index>               iris landed on a new index
EVT RING FOCUS <+/-delta>      user turned the focus ring
EVT RING APERTURE <+/-delta>   user turned the aperture ring
EVT ERROR <detail>             startup failed, lens rejected a command, ...
```

`RESYNC` means the lens dropped transport sync and is streaming the resync
marker; the server runs the reset dialogue and re-initializes on its own, and
returns to `READY` when the lens answers again. Focus/iris commands issued
during a resync are refused with an `EVT ERROR ..._DROPPED` rather than being
queued for an unbounded time — reissue them after `EVT STATE READY`.

### Example session

```
-> SET POWER ON
<- OK POWER ON
<- EVT STATE STARTING
<- EVT POWER ON
<- EVT STATE READY
<- EVT IRIS 1
-> SET IRIS 10
<- OK IRIS 10
<- EVT IRIS 10
-> SET FOCUS -1500
<- OK FOCUS -1500
<- EVT FOCUS -1204
<- EVT FOCUS -1500
<- EVT FOCUS_SETTLED -1500
-> GET FOCUS
<- OK FOCUS -1500
-> SET POWER OFF
<- OK POWER OFF
<- EVT POWER OFF
<- EVT STATE OFF
```

## Pi setup

Dependencies: `sudo apt install python3-spidev python3-lgpio python3-serial`.

Enable SPI (`dtparam=spi=on` in `/boot/firmware/config.txt`) and free the UART
from the login console — `sudo raspi-config` → Interface Options → Serial Port
→ login shell **No**, hardware **Yes** (equivalently: drop
`console=serial0,115200` from `/boot/firmware/cmdline.txt` and add
`enable_uart=1` to `config.txt`). Reboot.

Wiring is in `gf_controller.py`'s module docstring; the server adds only the
UART:

```
Pi GPIO14 (TXD, phys pin 8)  -> host RX
Pi GPIO15 (RXD, phys pin 10) -> host TX
Pi GND    (phys pin 6)       -> host GND
```

### Run at startup

```sh
sudo cp gf-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now gf-server
journalctl -fu gf-server        # live protocol log
```

Edit `ExecStart`/`WorkingDirectory`/`User` in the unit if the repo is not at
`/home/pi/fuji-controller`.

### Trying it without hardware

```sh
python3 gf_server.py --port - --dry-run --settle 0.2
```

Speaks the protocol on stdin/stdout with the SPI link simulated from the
capture; all diagnostics go to stderr so the command stream stays clean.



### Focus Distance Model

Initial estimate with some noisy data - needs further confirmation.

 a lens's focus extension is f²/d, so counts should be linear in 1/d, and panel 2 shows it is:


```
counts(d) = -381.9 + 28426 / d        d in metres    R² = 0.957, RMSE 37 counts
d(counts)  = 28426 / (counts + 381.9)
```

Infinity asymptote ≈ −382 counts
