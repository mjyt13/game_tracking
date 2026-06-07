"""
Определение сетевых адресов и регистрация mDNS-имени.
Зависимость zeroconf опциональна: при её отсутствии mDNS просто не запускается,
сервер остаётся доступен по IP.
"""

import socket


def _udp_route_ip() -> str | None:
    """IP интерфейса маршрута наружу (UDP-сокет данные не шлёт)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _ip_priority(ip: str) -> int:
    """
    Ранг адреса как «настоящего LAN». Меньше — выше приоритет.
    192.168.x / 10.x — типичные домашние/офисные сети; 172.x — часто
    виртуальные адаптеры (Docker, WSL, Hyper-V), их ставим ниже.
    """
    if ip.startswith("192.168."):
        return 0
    if ip.startswith("10."):
        return 1
    if ip.startswith("172."):
        return 3   # вероятно виртуальный — ниже реальных LAN
    return 2


def get_all_ips() -> list[str]:
    """
    Все IPv4-адреса машины, кроме loopback (127.x) и APIPA (169.254.x),
    отсортированные по вероятности быть реальным LAN-адресом (первым — лучший).
    Полезно при нескольких адаптерах (Wi-Fi + Ethernet + виртуальные).
    """
    found: set[str] = set()

    route_ip = _udp_route_ip()
    if route_ip:
        found.add(route_ip)
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.add(ip)
    except OSError:
        pass

    ips = [
        ip for ip in found
        if not ip.startswith("127.") and not ip.startswith("169.254.")
    ]
    ips.sort(key=_ip_priority)
    return ips


def get_primary_ip() -> str | None:
    """Лучший кандидат на реальный LAN-адрес (первый из get_all_ips)."""
    ips = get_all_ips()
    return ips[0] if ips else None


def start_mdns(hostname: str, port: int, ip: str | None):
    """
    Зарегистрировать mDNS-сервис, чтобы устройства обращались по '<hostname>.local'.
    Возвращает (zeroconf, service_info) для последующей отмены, либо (None, None),
    если zeroconf не установлен или регистрация не удалась.
    """
    if not ip:
        return None, None
    try:
        from zeroconf import Zeroconf, ServiceInfo
    except ImportError:
        return None, None

    try:
        info = ServiceInfo(
            "_http._tcp.local.",
            f"{hostname}._http._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=port,
            properties={"path": "/"},
            server=f"{hostname}.local.",
        )
        zc = Zeroconf()
        zc.register_service(info)
        return zc, info
    except Exception:
        return None, None


def stop_mdns(zc, info) -> None:
    """Отменить регистрацию mDNS-сервиса."""
    if zc is None:
        return
    try:
        if info is not None:
            zc.unregister_service(info)
        zc.close()
    except Exception:
        pass
