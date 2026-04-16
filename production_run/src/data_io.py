from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlencode

import pandas as pd
import requests


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dig(obj, path: str, default=None):
    """Read nested dictionary values by dot path."""
    if not path:
        return default
    parts = path.split(".")
    cur = obj
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _build_url(base_url: str, ip: str, from_ms: int, to_ms: int, request_style: str) -> str:
    """Build endpoint URL using configured request style."""
    base_url = base_url.rstrip("/")
    from_s = from_ms // 1000
    to_s = to_ms // 1000

    if request_style == "path_milliseconds":
        return f"{base_url}/{ip}/{from_ms}/{to_ms}"
    if request_style == "path_seconds":
        return f"{base_url}/{ip}/{from_s}/{to_s}"
    if request_style == "query_milliseconds":
        return f"{base_url}/{ip}?{urlencode({'from': from_ms, 'to': to_ms})}"
    if request_style == "query_seconds":
        return f"{base_url}/{ip}?{urlencode({'from': from_s, 'to': to_s})}"

    raise ValueError(
        "Unsupported request_style. Use one of: "
        "path_milliseconds, path_seconds, query_milliseconds, query_seconds"
    )


def _fetch_iops(
    ip: str,
    from_ms: int,
    to_ms: int,
    base_url: str,
    headers: Dict[str, str],
    timeout: int,
    request_style: str,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    url = _build_url(base_url, ip, from_ms, to_ms, request_style)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    message_key = mapping.get("message_key", "message")
    time_key = mapping.get("time_key", "time_str")
    read_key = mapping.get("read_key", "read_data")
    write_key = mapping.get("write_key", "write_data")
    ip_key = mapping.get("ip_key", "ip")

    rows: List[dict] = []
    for item in payload.get(message_key, []):
        ts_raw = _dig(item, time_key, 0)
        ts_int = int(_to_float(ts_raw, 0))
        ts = datetime.fromtimestamp(ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int, tz=timezone.utc)

        read_iops = _to_float(_dig(item, read_key, 0.0))
        write_iops = _to_float(_dig(item, write_key, 0.0))
        rows.append(
            {
                "timestamp": ts,
                "ip": _dig(payload, ip_key, ip),
                "read_iops": read_iops,
                "write_iops": write_iops,
                "total_iops": read_iops + write_iops,
            }
        )

    return pd.DataFrame(rows)


def _fetch_network(
    ip: str,
    from_ms: int,
    to_ms: int,
    base_url: str,
    headers: Dict[str, str],
    timeout: int,
    mapping: Dict[str, str],
    request_style: str,
) -> pd.DataFrame:
    url = _build_url(base_url, ip, from_ms, to_ms, request_style)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    message_key = mapping.get("message_key", "message")
    time_key = mapping.get("time_key", "time_str")
    in_key = mapping.get("in_key", "in_data")
    out_key = mapping.get("out_key", "out_data")
    ip_key = mapping.get("ip_key", "ip")
    single_key = mapping.get("single_key", "")

    rows: List[dict] = []
    for item in payload.get(message_key, []):
        ts_raw = _dig(item, time_key, 0)
        ts_int = int(_to_float(ts_raw, 0))
        ts = datetime.fromtimestamp(ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int, tz=timezone.utc)

        in_raw = _dig(item, in_key)
        out_raw = _dig(item, out_key)
        if single_key:
            single_raw = _dig(item, single_key)
            if single_raw is not None:
                in_raw = single_raw
                out_raw = 0.0
        if in_raw is None:
            in_raw = item.get("in_data", item.get("in_bandwidth"))
        if out_raw is None:
            out_raw = item.get("out_data", item.get("out_bandwidth"))

        in_bw = _to_float(in_raw)
        out_bw = _to_float(out_raw)

        rows.append(
            {
                "timestamp": ts,
                "ip": _dig(payload, ip_key, ip),
                "in_bandwidth": in_bw,
                "out_bandwidth": out_bw,
                "total_network": in_bw + out_bw,
            }
        )

    return pd.DataFrame(rows)


def _extract_interface_bandwidth(item: dict, interface_key: str, in_key: str, out_key: str) -> tuple[float, float]:
    interfaces = _dig(item, interface_key, [])
    if not isinstance(interfaces, list) or not interfaces:
        return 0.0, 0.0

    in_total = 0.0
    out_total = 0.0
    for itf in interfaces:
        in_total += _to_float(_dig(itf, in_key, 0.0))
        out_total += _to_float(_dig(itf, out_key, 0.0))

    return in_total, out_total


def _fetch_vm_unified(
    ip: str,
    from_ms: int,
    to_ms: int,
    base_url: str,
    headers: Dict[str, str],
    timeout: int,
    request_style: str,
    mapping: Dict[str, str],
) -> pd.DataFrame:
    """Fetch vm-performance once and derive both disk IOPS and network columns."""
    url = _build_url(base_url, ip, from_ms, to_ms, request_style)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    message_key = mapping.get("message_key", "performance_metrics")
    # Primary disk IOPS source in vm-performance payload.
    time_key = mapping.get("time_key", "disk_io_summary.io_operations_data.time_str")
    read_key = mapping.get("read_key", "disk_io_summary.io_operations_data.read_data")
    write_key = mapping.get("write_key", "disk_io_summary.io_operations_data.write_data")
    # Network source in vm-performance payload.
    interface_key = mapping.get("interface_key", "interface")
    interface_in_key = mapping.get("interface_in_key", "in_bandwidth")
    interface_out_key = mapping.get("interface_out_key", "out_bandwidth")
    ip_key = mapping.get("ip_key", "ip")
    payload_ip_key = mapping.get("payload_ip_key", "lan_ip")

    rows: List[dict] = []
    metrics = payload.get(message_key, [])
    if not isinstance(metrics, list):
        metrics = []

    for item in metrics:
        ts_raw = _dig(item, time_key, None)
        if ts_raw is None:
            # Fallbacks for variant VM payloads.
            ts_raw = _dig(item, "cpu_load.time_str", 0)
        ts_int = int(_to_float(ts_raw, 0))
        ts = datetime.fromtimestamp(ts_int / 1000.0 if ts_int > 10_000_000_000 else ts_int, tz=timezone.utc)

        read_iops = _to_float(_dig(item, read_key, 0.0))
        write_iops = _to_float(_dig(item, write_key, 0.0))
        in_bw, out_bw = _extract_interface_bandwidth(item, interface_key, interface_in_key, interface_out_key)

        row_ip = _dig(item, ip_key, None) or _dig(payload, payload_ip_key, None) or ip

        rows.append(
            {
                "timestamp": ts,
                "ip": row_ip,
                "read_iops": read_iops,
                "write_iops": write_iops,
                "total_iops": read_iops + write_iops,
                "in_bandwidth": in_bw,
                "out_bandwidth": out_bw,
                "total_network": in_bw + out_bw,
            }
        )

    return pd.DataFrame(rows)


def collect_from_api(config: dict) -> pd.DataFrame:
    api_cfg = config.get("api", {})
    if not api_cfg.get("enabled", False):
        raise ValueError("API mode is disabled in config")

    vm_unified_enabled = bool(api_cfg.get("vm_unified_enabled", False))

    iops_base_url = api_cfg.get("iops_base_url", "").strip()
    network_base_url = api_cfg.get("network_base_url", "").strip()
    vm_base_url = api_cfg.get("vm_base_url", network_base_url).strip()

    if vm_unified_enabled:
        if not vm_base_url:
            raise ValueError("vm_base_url (or network_base_url) is required when vm_unified_enabled=true")
    else:
        if not iops_base_url or not network_base_url:
            raise ValueError("Both iops_base_url and network_base_url are required in API mode")

    secret = api_cfg.get("secret_key", "").strip()
    headers = {"X-SECRET-KEY": secret} if secret else {}
    timeout = int(api_cfg.get("timeout_seconds", 15))
    network_mapping = api_cfg.get("network_response", {}) or {}
    iops_mapping = api_cfg.get("iops_response", {}) or {}
    iops_request_style = api_cfg.get("iops_request_style", "path_milliseconds")
    network_request_style = api_cfg.get("network_request_style", "query_milliseconds")
    vm_mapping = api_cfg.get("vm_unified_response", {}) or {}
    vm_request_style = api_cfg.get("vm_request_style", network_request_style)

    history_hours = int(config.get("lookback", {}).get("history_hours", 240))
    to_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    from_ms = int((datetime.now(tz=timezone.utc) - timedelta(hours=history_hours)).timestamp() * 1000)

    all_rows: List[pd.DataFrame] = []
    ips: Iterable[str] = api_cfg.get("ips", [])
    for ip in ips:
        if vm_unified_enabled:
            vm_df = _fetch_vm_unified(
                ip,
                from_ms,
                to_ms,
                vm_base_url,
                headers,
                timeout,
                vm_request_style,
                vm_mapping,
            )
            all_rows.append(vm_df)
        else:
            iops_df = _fetch_iops(
                ip,
                from_ms,
                to_ms,
                iops_base_url,
                headers,
                timeout,
                iops_request_style,
                iops_mapping,
            )
            net_df = _fetch_network(
                ip,
                from_ms,
                to_ms,
                network_base_url,
                headers,
                timeout,
                network_mapping,
                network_request_style,
            )
            merged = pd.merge(iops_df, net_df, on=["timestamp", "ip"], how="inner")
            all_rows.append(merged)

    if not all_rows:
        raise ValueError("No data fetched from APIs")

    df = pd.concat(all_rows, ignore_index=True)
    return standardize_dataset(df)


def read_unified_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return standardize_dataset(df)


def standardize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "ip",
        "read_iops",
        "write_iops",
        "in_bandwidth",
        "out_bandwidth",
    }

    if "total_iops" not in df.columns:
        df["total_iops"] = df["read_iops"].astype(float) + df["write_iops"].astype(float)
    if "total_network" not in df.columns:
        df["total_network"] = df["in_bandwidth"].astype(float) + df["out_bandwidth"].astype(float)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in [
        "read_iops",
        "write_iops",
        "total_iops",
        "in_bandwidth",
        "out_bandwidth",
        "total_network",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "ip"]).sort_values(["ip", "timestamp"]).reset_index(drop=True)
    return df
