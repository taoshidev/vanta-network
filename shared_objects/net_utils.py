"""External IP resolution, cached for the life of the process.

Replaces bittensor.utils.networking.get_external_ip (removed in bittensor v11) with a
small standalone implementation so this repo has no dependency on bittensor's internal
networking utilities.
"""
import urllib.request

from shared_objects.log import logger

_EXTERNAL_IP_SERVICES = (
    "https://checkip.amazonaws.com",
    "https://api.ipify.org",
)

_external_ip: str | None = None


def get_external_ip() -> str:
    global _external_ip
    if _external_ip is not None:
        return _external_ip

    last_error = None
    for url in _EXTERNAL_IP_SERVICES:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                ip = resp.read().decode().strip()
            if ip:
                _external_ip = ip
                return _external_ip
        except Exception as e:
            last_error = e
            logger.warning(f"get_external_ip: {url} failed: {e}")

    raise RuntimeError(f"Unable to determine external IP from any service: {last_error}")
