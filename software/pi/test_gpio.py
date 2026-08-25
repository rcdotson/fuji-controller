#!/usr/bin/env python3
"""Standalone test for the lens-power GPIO switch.

Loads LensPower/Keyboard from gf_body_replay and toggles the pin directly:
press `1` to drive the pin high (lens power on), `2` for low (off),
`q` to quit. The pin is driven low on exit.

Usage:
    python3 test_gpio.py           # GPIO6 (the lens-power default)
    python3 test_gpio.py 17        # any other BCM pin
"""

import sys
import time

from gf_body_replay import Keyboard, LensPower


def main() -> None:
    gpio = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    power = LensPower(gpio)
    if not power.enabled:
        sys.exit("could not claim the GPIO — see message above")

    kb = Keyboard()
    if not kb.enabled:
        power.close()
        sys.exit("stdin is not a tty — run from an interactive terminal")

    print(f"GPIO{gpio} claimed, driven LOW.  1 = high  2 = low  q = quit")
    try:
        while True:
            key = kb.poll()
            if key == "q":
                break
            if key in ("1", "2"):
                on = key == "1"
                power.set(on)
                print(f"GPIO{gpio} -> {'HIGH' if on else 'LOW'}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        kb.restore()
        power.close()
        print(f"\nGPIO{gpio} driven low and released")


if __name__ == "__main__":
    main()
