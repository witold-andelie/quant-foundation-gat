from __future__ import annotations

import pandas as pd
import pytest

from quant_alpha.ingestion.entsoe import (
    EntsoeError,
    _fetch_generation_mix,
    _period_params,
    parse_entsoe_timeseries,
)


def _quantity_xml(value: bytes) -> bytes:
    return (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">\n'
        b"  <TimeSeries><Period>\n"
        b"    <timeInterval><start>2024-01-01T00:00Z</start><end>2024-01-01T03:00Z</end></timeInterval>\n"
        b"    <resolution>PT60M</resolution>\n"
        b"    <Point><position>1</position><quantity>" + value + b"</quantity></Point>\n"
        b"    <Point><position>2</position><quantity>" + value + b"</quantity></Point>\n"
        b"    <Point><position>3</position><quantity>" + value + b"</quantity></Point>\n"
        b"  </Period></TimeSeries>\n"
        b"</GL_MarketDocument>\n"
    )


class _FakeClient:
    """Returns canned XML keyed by psrType — no network."""

    def __init__(self, by_psr: dict[str, bytes]):
        self.by_psr = by_psr
        self.calls: list[dict] = []

    def request(self, params: dict) -> bytes:
        self.calls.append(params)
        return self.by_psr.get(params.get("psrType", ""), b"<empty/>")


SAMPLE_PRICE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <Period>
      <timeInterval>
        <start>2024-01-01T00:00Z</start>
        <end>2024-01-01T03:00Z</end>
      </timeInterval>
      <resolution>PT60M</resolution>
      <Point>
        <position>1</position>
        <price.amount>40.5</price.amount>
      </Point>
      <Point>
        <position>2</position>
        <price.amount>42.0</price.amount>
      </Point>
      <Point>
        <position>3</position>
        <price.amount>39.0</price.amount>
      </Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


def test_parse_entsoe_timeseries_handles_namespaced_price_points() -> None:
    series = parse_entsoe_timeseries(SAMPLE_PRICE_XML, ("price.amount",))

    assert list(series.index) == list(pd.date_range("2024-01-01", periods=3, freq="h"))
    assert series.tolist() == [40.5, 42.0, 39.0]


def test_period_params_are_utc_entsoe_format() -> None:
    params = _period_params("2024-01-01", "2024-01-02")

    assert params == {
        "periodStart": "202401010000",
        "periodEnd": "202401020000",
    }


def test_parser_rejects_unsupported_resolution() -> None:
    xml = SAMPLE_PRICE_XML.replace(b"<resolution>PT60M</resolution>", b"<resolution>P1Y</resolution>")

    with pytest.raises(EntsoeError):
        parse_entsoe_timeseries(xml, ("price.amount",))


def test_parser_reads_quantity_series() -> None:
    # load (A65) and generation (A75) series carry <quantity>, not <price.amount>.
    series = parse_entsoe_timeseries(_quantity_xml(b"123.5"), ("quantity",))
    assert series.tolist() == [123.5, 123.5, 123.5]


def test_generation_mix_aggregates_psr_groups() -> None:
    client = _FakeClient(
        {
            "B14": _quantity_xml(b"100"),  # nuclear
            "B04": _quantity_xml(b"50"),   # gas
            "B02": _quantity_xml(b"30"),   # lignite
            "B05": _quantity_xml(b"20"),   # hard coal -> coal group = 50
            "B11": _quantity_xml(b"10"),   # run-of-river
            "B12": _quantity_xml(b"5"),    # reservoir -> hydro group = 15
        }
    )
    mix = _fetch_generation_mix(client, "DOM", _period_params("2024-01-01", "2024-01-02"), "h")

    assert set(mix) == {"gen_nuclear", "gen_gas", "gen_coal", "gen_hydro", "gen_total"}
    assert mix["gen_nuclear"].mean() == 100
    assert mix["gen_coal"].mean() == 50      # B02 + B05
    assert mix["gen_hydro"].mean() == 15     # B11 + B12
    assert mix["gen_total"].mean() == 215    # 100 + 50 + 50 + 15


def test_fetch_cross_border_flows_and_ntc() -> None:
    from quant_alpha.ingestion.entsoe import fetch_entsoe_cross_border

    class _FlowClient:
        def request(self, params):
            dt = params.get("documentType")
            if dt == "A11":
                return _quantity_xml(b"500")   # physical flow MW
            if dt == "A61":
                return _quantity_xml(b"1000")  # day-ahead NTC MW
            return b"<empty/>"

    df = fetch_entsoe_cross_border(
        [("A", "B"), ("B", "A")], {"A": "X", "B": "Y"}, "2024-01-01", "2024-01-02", "h", _FlowClient()
    )
    assert list(df.columns) == ["timestamp", "from_zone", "to_zone", "flow", "ntc"]
    assert set(zip(df["from_zone"], df["to_zone"])) == {("A", "B"), ("B", "A")}
    assert (df["flow"] == 500).all() and (df["ntc"] == 1000).all()


def test_generation_mix_skips_missing_group() -> None:
    # a publisher with no nuclear data -> the group is omitted, not zero-filled,
    # and gen_total excludes it (the zone still runs on what is present).
    client = _FakeClient({"B04": _quantity_xml(b"50")})  # only gas
    mix = _fetch_generation_mix(client, "DOM", _period_params("2024-01-01", "2024-01-02"), "h")
    assert "gen_nuclear" not in mix
    assert mix["gen_gas"].mean() == 50
    assert mix["gen_total"].mean() == 50
