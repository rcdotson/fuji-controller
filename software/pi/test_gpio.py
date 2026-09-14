#!/usr/bin/env python3
"""Standalone test for the lens-power GPIO switch.

Loads LensPower/Keyboard from gf_controller and toggles the pins directly:
press `1` to drive them high (lens power on), `2` for low (off), `q` to quit.
The current board has two power-control lines (GPIO17 and GPIO27) and
LensPower drives every line it owns identically, so both move together.
The pins are driven low on exit.

Usage:
    python3 test_gpio.py            # GPIO17 + GPIO27 (the board default)
    python3 test_gpio.py 17         # a single line
    python3 test_gpio.py 17,27      # any explicit set
"""

import sys
import time

from gf_controller import DEFAULT_POWER_GPIOS, Keyboard, LensPower, \
    parse_power_gpios


def main() -> None:
    gpios = (parse_power_gpios(" ".join(sys.argv[1:])) if len(sys.argv) > 1
             else list(DEFAULT_POWER_GPIOS))
    if not gpios:
        sys.exit("no GPIOs to test")
    power = LensPower(gpios)
    if not power.enabled:
        sys.exit("could not claim the GPIOs — see message above")

    kb = Keyboard()
    if not kb.enabled:
        power.close()
        sys.exit("stdin is not a tty — run from an interactive terminal")

    print(f"{power.label} claimed, driven LOW.  1 = high  2 = low  q = quit")
    try:
        while True:
            key = kb.poll()
            if key == "q":
                break
            if key in ("1", "2"):
                on = key == "1"
                power.set(on)
                print(f"{power.label} -> {'HIGH' if on else 'LOW'}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        kb.restore()
        power.close()
        print(f"\n{power.label} driven low and released")


if __name__ == "__main__":
    main()
