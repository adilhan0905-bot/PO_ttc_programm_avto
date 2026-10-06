from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class CanMessage:
    arbitration_id: int
    danger_code: int
    speed_kmh: float
    ttc_s: float


class CanBusAdapter:
    """Educational CAN adapter. By default it simulates CAN transmission."""

    def __init__(self, enabled: bool = True, simulate: bool = True) -> None:
        self.enabled = enabled
        self.simulate = simulate
        self._bus = None
        if enabled and not simulate:
            try:
                import can  # type: ignore
                self._bus = can.Bus(interface="socketcan", channel="can0", bitrate=500000)
            except Exception as exc:  # pragma: no cover - platform specific
                logging.warning("CAN initialization failed: %s", exc)
                self._bus = None

    def send_critical(self, speed_kmh: float, ttc_s: float) -> bool:
        if not self.enabled:
            return False
        msg = CanMessage(0x321, 2, speed_kmh, ttc_s)
        if self.simulate or self._bus is None:
            logging.info("CAN SIMULATION id=0x%03X danger=%s speed=%.1f ttc=%.2f", msg.arbitration_id, msg.danger_code, speed_kmh, ttc_s)
            return True
        try:  # pragma: no cover - depends on physical CAN hardware
            import can  # type: ignore
            payload = bytearray(8)
            payload[0] = msg.danger_code
            payload[1:3] = int(max(0, min(65535, speed_kmh * 10))).to_bytes(2, "big")
            payload[3:5] = int(max(0, min(65535, ttc_s * 100))).to_bytes(2, "big")
            self._bus.send(can.Message(arbitration_id=msg.arbitration_id, data=payload, is_extended_id=False))
            return True
        except Exception as exc:
            logging.error("CAN send failed: %s", exc)
            return False
