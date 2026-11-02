"""Bind CUDA thread-local device; do not introduce GPU synchronization."""
import logging
import threading
import time

logger = logging.getLogger(__name__)
_local = threading.local()


def guarded_transfer(cuda, device, transfer, session, sources, destinations, lengths):
    if not (len(sources) == len(destinations) == len(lengths)):
        raise ValueError("KV transfer pointer/length counts differ")
    if not sources:
        return 0
    device = int(device)
    before = cuda.current_device()
    started = time.monotonic()
    result = None
    try:
        with cuda.device(device):
            if cuda.current_device() != device:
                raise RuntimeError("NVLink caller CUDA device guard failed")
            seen = getattr(_local, "seen", set())
            key = (before, device)
            if key not in seen:
                logger.info("KV_DEVICE_GUARD thread=%s previous=%s selected=%s",
                            threading.get_ident(), before, device)
                _local.seen = seen | {key}
            result = transfer(session, sources, destinations, lengths)
            return result
    finally:
        elapsed = time.monotonic() - started
        if elapsed >= 1.0 or result != 0:
            logger.warning(
                "KV_COPY_CALL device=%s session=%s fragments=%s bytes=%s "
                "elapsed_s=%.6f result=%s",
                device, session, len(lengths), sum(lengths), elapsed, result)
