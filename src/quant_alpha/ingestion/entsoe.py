from __future__ import annotations

import os
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Iterable

import pandas as pd


class EntsoeError(RuntimeError):
    pass


@dataclass(frozen=True)
class EntsoeClient:
    token: str
    base_url: str = "https://web-api.tp.entsoe.eu/api"
    timeout_seconds: int = 60
    polite_sleep_seconds: float = 0.2

    @classmethod
    def from_env(
        cls,
        token_env: str = "ENTSOE_API_KEY",
        base_url: str = "https://web-api.tp.entsoe.eu/api",
        timeout_seconds: int = 60,
    ) -> "EntsoeClient":
        token = os.getenv(token_env)
        if not token:
            raise EntsoeError(
                f"Missing ENTSO-E API token. Set {token_env} before using data_source=entsoe."
            )
        return cls(token=token, base_url=base_url, timeout_seconds=timeout_seconds)

    def request(self, params: dict[str, str]) -> bytes:
        query = {"securityToken": self.token, **params}
        url = f"{self.base_url}?{urllib.parse.urlencode(query)}"
        context = ssl.create_default_context()
        try:
            import certifi

            context.load_verify_locations(cafile=certifi.where())
        except Exception:
            # Fall back to platform trust store if certifi is unavailable.
            pass
        with urllib.request.urlopen(url, timeout=self.timeout_seconds, context=context) as response:
            payload = response.read()
        time.sleep(self.polite_sleep_seconds)
        return payload


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in element if _strip_namespace(child.tag) == name)


def _first_text(element: ET.Element, name: str) -> str | None:
    for child in _children(element, name):
        return child.text
    return None


def _period_start(period: ET.Element) -> datetime:
    for interval in _children(period, "timeInterval"):
        start = _first_text(interval, "start")
        if start:
            return pd.Timestamp(start).to_pydatetime()
    raise EntsoeError("ENTSO-E response period is missing timeInterval/start.")


def _resolution_delta(value: str | None) -> timedelta:
    if value in {"PT15M", "PT15m"}:
        return timedelta(minutes=15)
    if value in {"PT30M", "PT30m"}:
        return timedelta(minutes=30)
    if value in {"PT60M", "PT1H", "PT60m", "PT1h"}:
        return timedelta(hours=1)
    if value in {"P1D", "P1d"}:
        return timedelta(days=1)
    raise EntsoeError(f"Unsupported ENTSO-E resolution: {value}")


def parse_entsoe_timeseries(xml_payload: bytes, value_names: tuple[str, ...]) -> pd.Series:
    root = ET.parse(BytesIO(xml_payload)).getroot()
    records: list[dict[str, object]] = []

    for timeseries in root.iter():
        if _strip_namespace(timeseries.tag) != "TimeSeries":
            continue
        for period in _children(timeseries, "Period"):
            start = _period_start(period)
            delta = _resolution_delta(_first_text(period, "resolution"))
            for point in _children(period, "Point"):
                position_text = _first_text(point, "position")
                if position_text is None:
                    continue
                value_text = next(
                    (_first_text(point, value_name) for value_name in value_names if _first_text(point, value_name)),
                    None,
                )
                if value_text is None:
                    continue
                records.append(
                    {
                        "timestamp": start + delta * (int(position_text) - 1),
                        "value": float(value_text),
                    }
                )

    if not records:
        return pd.Series(dtype="float64", name="value")

    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(None)
    return frame.groupby("timestamp")["value"].mean().sort_index()


def _period_params(start: str, end: str) -> dict[str, str]:
    start_ts = pd.Timestamp(start, tz=timezone.utc)
    end_ts = pd.Timestamp(end, tz=timezone.utc)
    return {
        "periodStart": start_ts.strftime("%Y%m%d%H%M"),
        "periodEnd": end_ts.strftime("%Y%m%d%H%M"),
    }


def _query_series(client: EntsoeClient, params: dict[str, str], value_names: tuple[str, ...]) -> pd.Series:
    try:
        payload = client.request(params)
    except Exception as exc:  # pragma: no cover - network failures vary by platform
        raise EntsoeError(f"ENTSO-E request failed for {params}: {exc}") from exc
    return parse_entsoe_timeseries(payload, value_names)


def _resample(series: pd.Series, interval: str) -> pd.Series:
    if series.empty:
        return series
    return series.resample(interval).mean().interpolate(limit_direction="both")


# psrType groups for the generation-mix features (A75 actual generation/type).
# Pumped storage (B10) is excluded to avoid generation/consumption sign
# ambiguity; wind and solar already arrive as A69 day-ahead forecasts. These are
# *realised* generation (known only up to t), so the forecast harness uses them
# as point-in-time anchors, not as t+k drivers.
_GENERATION_GROUPS: dict[str, tuple[str, ...]] = {
    "gen_nuclear": ("B14",),
    "gen_gas": ("B04",),
    "gen_coal": ("B02", "B05"),   # lignite + hard coal
    "gen_hydro": ("B11", "B12"),  # run-of-river + reservoir
}


def _safe_query(client: EntsoeClient, params: dict[str, str], value_names: tuple[str, ...]) -> pd.Series:
    """Like ``_query_series`` but returns an empty series instead of raising.

    For enrichment series (actual load, generation mix) that must not fail a
    whole zone when a single publisher is missing or slow."""
    try:
        return _query_series(client, params, value_names)
    except EntsoeError:
        return pd.Series(dtype="float64", name="value")


def _fetch_generation_mix(
    client: EntsoeClient, domain: str, period: dict[str, str], bar_interval: str
) -> dict[str, pd.Series]:
    """Actual generation per fuel group (A75/A16), resampled.

    Each group sums its psrTypes; a group with no data is omitted so a partial
    publisher still runs. Returns ``name -> series`` plus ``gen_total`` over the
    groups present."""
    out: dict[str, pd.Series] = {}
    total: pd.Series | None = None
    for name, psr_types in _GENERATION_GROUPS.items():
        merged: pd.Series | None = None
        for psr in psr_types:
            part = _safe_query(
                client,
                {
                    "documentType": "A75",
                    "processType": "A16",
                    "in_Domain": domain,
                    "psrType": psr,
                    **period,
                },
                ("quantity",),
            )
            if not part.empty:
                merged = part if merged is None else merged.add(part, fill_value=0)
        if merged is not None and not merged.empty:
            resampled = _resample(merged, bar_interval)
            out[name] = resampled
            total = resampled if total is None else total.add(resampled, fill_value=0)
    if total is not None:
        out["gen_total"] = total
    return out


def fetch_entsoe_power_market(
    markets: list[str],
    domains: dict[str, str],
    start: str,
    end: str,
    bar_interval: str,
    client: EntsoeClient,
    include_generation: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    period = _period_params(start, end)

    for market in markets:
        domain = domains.get(market)
        if not domain:
            raise EntsoeError(f"Missing ENTSO-E bidding-zone domain code for market {market}.")

        spot = _query_series(
            client,
            {
                "documentType": "A44",
                "in_Domain": domain,
                "out_Domain": domain,
                **period,
            },
            ("price.amount",),
        )
        load = _query_series(
            client,
            {
                "documentType": "A65",
                "processType": "A01",
                "outBiddingZone_Domain": domain,
                **period,
            },
            ("quantity",),
        )
        solar = _query_series(
            client,
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": domain,
                "psrType": "B16",
                **period,
            },
            ("quantity",),
        )
        wind_onshore = _query_series(
            client,
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": domain,
                "psrType": "B19",
                **period,
            },
            ("quantity",),
        )
        wind_offshore = _query_series(
            client,
            {
                "documentType": "A69",
                "processType": "A01",
                "in_Domain": domain,
                "psrType": "B18",
                **period,
            },
            ("quantity",),
        )

        # Realised load (A65/A16) — the demand-surprise anchor; non-fatal if a
        # zone does not publish it in time.
        actual_load = _safe_query(
            client,
            {"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": domain, **period},
            ("quantity",),
        )

        series_map = {
            "spot_price": _resample(spot, bar_interval),
            "load_forecast": _resample(load, bar_interval),
            "solar_forecast": _resample(solar, bar_interval),
            "wind_forecast": _resample(wind_onshore.add(wind_offshore, fill_value=0), bar_interval),
            "actual_load": _resample(actual_load, bar_interval),
        }
        if include_generation:
            for name, series in _fetch_generation_mix(client, domain, period, bar_interval).items():
                series_map[name] = series

        market_frame = pd.concat(series_map, axis=1).dropna(subset=["spot_price", "load_forecast"])

        market_frame["solar_forecast"] = market_frame["solar_forecast"].fillna(0)
        market_frame["wind_forecast"] = market_frame["wind_forecast"].fillna(0)
        market_frame["residual_load"] = (
            market_frame["load_forecast"]
            - market_frame["wind_forecast"]
            - market_frame["solar_forecast"]
        )
        gen_cols = [c for c in market_frame.columns if c.startswith("gen_")]
        market_frame[gen_cols] = market_frame[gen_cols].fillna(0)
        if "actual_load" in market_frame:
            market_frame["demand_surprise"] = market_frame["actual_load"] - market_frame["load_forecast"]
        scarcity = market_frame["residual_load"].rank(pct=True).fillna(0.5) - 0.5
        market_frame["imbalance_price"] = market_frame["spot_price"] + scarcity * 10.0
        market_frame["market"] = market
        market_frame = market_frame.reset_index().rename(columns={"index": "timestamp"})
        frames.append(market_frame)

    if not frames:
        raise EntsoeError("ENTSO-E returned no data for the requested markets.")

    combined = pd.concat(frames, ignore_index=True)
    gen_cols = sorted(c for c in combined.columns if c.startswith("gen_"))
    if gen_cols:  # zones missing a group entirely -> 0 for that group
        combined[gen_cols] = combined[gen_cols].fillna(0)
    base = [
        "timestamp",
        "market",
        "spot_price",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "residual_load",
        "imbalance_price",
    ]
    enrich = [c for c in ["actual_load", "demand_surprise", *gen_cols] if c in combined.columns]
    return combined[base + enrich]


def fetch_entsoe_cross_border(
    directed_borders,
    domains: dict[str, str],
    start: str,
    end: str,
    bar_interval: str,
    client: EntsoeClient,
) -> pd.DataFrame:
    """Directed cross-border physical flows (A11) + day-ahead NTC (A61/A01).

    ``directed_borders`` is an iterable of ``(from_zone, to_zone)``. For each, the
    physical flow ``from -> to`` and the day-ahead net transfer capacity are
    fetched and resampled to ``bar_interval``. Resilient: a border with no flow is
    skipped and a missing NTC is left NaN (flow-based-coupling zones often have no
    A61), so a partial publisher still runs. Both quantities are realised /
    ex-ante known at ``t`` — the congestion edge feature is a point-in-time anchor.
    Returns ``[timestamp, from_zone, to_zone, flow, ntc]`` (MW).
    """
    period = _period_params(start, end)
    rows: list[pd.DataFrame] = []
    for a, b in directed_borders:
        da, db = domains.get(a), domains.get(b)
        if not da or not db:
            continue
        flow = _safe_query(
            client, {"documentType": "A11", "out_Domain": da, "in_Domain": db, **period}, ("quantity",)
        )
        if flow.empty:
            continue
        series = {"flow": _resample(flow, bar_interval)}
        ntc = _safe_query(
            client,
            {"documentType": "A61", "contract_MarketAgreement.Type": "A01",
             "out_Domain": da, "in_Domain": db, **period},
            ("quantity",),
        )
        if not ntc.empty:
            series["ntc"] = _resample(ntc, bar_interval)
        frame = pd.concat(series, axis=1)
        if "ntc" not in frame:
            frame["ntc"] = float("nan")
        frame["from_zone"] = a
        frame["to_zone"] = b
        rows.append(frame.reset_index().rename(columns={"index": "timestamp"}))
    if not rows:
        raise EntsoeError("ENTSO-E returned no cross-border flow data.")
    return pd.concat(rows, ignore_index=True)[["timestamp", "from_zone", "to_zone", "flow", "ntc"]]
