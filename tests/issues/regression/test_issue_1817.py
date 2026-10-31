"""`edgar_ownership(analysis_type="insiders")` dropped the insider, role and trades.

`_get_insider_transactions` read `reporting_owner`, `is_officer`, `is_director`,
`officer_title` and `transactions` from the Form 4 behind `hasattr` guards.
`Form4` has none of those attributes, so every guard was false and each entry
kept only `filing_date` and `accession_number`. `days` was accepted and then
ignored: the function took `head(limit * 2)` with no filing-date filter.

Ground truth is NVIDIA's Form 4 `0001588670-26-000014`, filed 2026-09-18.
The ownership XML names the reporting owner Kress Colette, officer, title
"EVP & Chief Financial Officer", with non-derivative dispositions of 40,747
shares (code F, tax withholding, $212.17) and open-market sales of 4,681
($218.0144), 20,369 ($219.0172) and 9,868 ($219.5932) shares. Read from
`wk-form4_1789765959.xml` on 2026-10-29. A 40-day lookback from that date
starts on 2026-09-19, so the same filing is outside it, while
`0001197647-26-000011` (filed 2026-10-09) is still inside.

A derivative-only Form 4 has no non-derivative rows. Vertex Pharmaceuticals
`0001209191-20-055264` (filed 2020-10-19, fixture
`data/ownership/VertexForm4.derivative-only.xml`) reports one award of
196.356 deferred stock units. Truncating that share count with `int` yields
196. Snowflake's joint Form 3 `data/form3.snow.nonderiv.xml` names Berkshire
Hathaway Inc and Warren E. Buffett as separate 10% owners.

GitHub Issue: https://github.com/dgunning/edgartools/issues/1817
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from edgar import set_identity

# NVIDIA Form 4, Colette Kress, filed 2026-09-18.
KRESS_ACCESSION = "0001588670-26-000014"
# A newer NVIDIA Form 4 that a 40-day window from 2026-10-29 still includes.
NEWER_ACCESSION = "0001197647-26-000011"
RECORDING_DATE = "2026-10-29"
# Numeric CIK so resolving the company does not download company_tickers.json.
NVIDIA_CIK = "1045810"

# Vertex Pharmaceuticals Form 4, Bruce I. Sachs, filed 2020-10-19.
# One derivative award, 196.356 deferred stock units, no non-derivative table.
VERTEX_XML = Path("data/ownership/VertexForm4.derivative-only.xml")
# Snowflake Form 3 filed 2020-09-18. Two reporting owners: Berkshire Hathaway
# Inc (CIK 1067983, a company) and Warren E. Buffett (CIK 315090).
JOINT_XML = Path("data/form3.snow.nonderiv.xml")
BERKSHIRE_CIK = 1067983


def _by_accession(result):
    return {row["accession_number"]: row for row in result.data["transactions"]}


def _pin_clock(monkeypatch, ownership):
    """Pin the lookback clock without requiring `_today` to exist.

    Unfixed main measures nothing: it never defines `_today`, and `days` is
    ignored. `raising=False` lets this test record and replay against that
    code. The attribute is unused there, so the pin does not change which
    filings unfixed code requests. The fix calls `_today()`.
    """
    monkeypatch.setattr(
        ownership,
        "_today",
        lambda: date.fromisoformat(RECORDING_DATE),
        raising=False,
    )


@pytest.mark.network
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_insider_transactions_report_owner_role_shares_and_honor_days(monkeypatch):
    """The Kress Form 4 names her, her title and the four share counts, and `days` drops it."""
    set_identity("Mark David Prado rqms40@users.noreply.github.com")
    from edgar.ai.mcp.tools import ownership

    _pin_clock(monkeypatch, ownership)
    # Both calls run before the assertions so a cassette recorded against the
    # broken code still contains every request the fixed code replays.
    wide = await ownership.edgar_ownership(
        identifier=NVIDIA_CIK, analysis_type="insiders", days=41, limit=12
    )
    narrow = await ownership.edgar_ownership(
        identifier=NVIDIA_CIK, analysis_type="insiders", days=40, limit=12
    )

    assert wide.success is True
    assert wide.data["analysis"] == "insider_transactions"
    kress = _by_accession(wide)[KRESS_ACCESSION]
    assert kress["filing_date"] == "2026-09-18"
    assert kress["insider_name"] == "Colette Kress"
    assert kress["role"] == "EVP & Chief Financial Officer"
    assert kress["title"] == "EVP & Chief Financial Officer"
    assert kress["is_officer"] is True
    assert kress["is_director"] is False
    assert kress["owners"][0]["insider_name"] == kress["insider_name"]
    assert kress["owners"][0]["is_officer"] is kress["is_officer"]
    assert len(kress["owners"]) == 1
    assert [trade["shares"] for trade in kress["transactions"]] == [40747, 4681, 20369, 9868]
    assert [trade["code"] for trade in kress["transactions"]] == ["F", "S", "S", "S"]
    assert [trade["type"] for trade in kress["transactions"]] == [
        "Tax Withholding",
        "Sale",
        "Sale",
        "Sale",
    ]
    assert [trade["action"] for trade in kress["transactions"]] == ["D", "D", "D", "D"]
    assert [trade["security"] for trade in kress["transactions"]] == ["Common", "Common", "Common", "Common"]
    assert [trade["price"] for trade in kress["transactions"]] == pytest.approx([212.17, 218.0144, 219.0172, 219.5932])
    assert all("derivative" not in trade for trade in kress["transactions"])
    # Plain ints and floats: the MCP payload is json.dumps'd.
    encoded = json.loads(json.dumps(wide.data))
    assert encoded["transactions"][_index(wide, KRESS_ACCESSION)]["transactions"][1]["shares"] == 4681

    assert narrow.success is True
    narrow_rows = _by_accession(narrow)
    assert KRESS_ACCESSION not in narrow_rows
    assert NEWER_ACCESSION in narrow_rows
    for row in narrow.data["transactions"]:
        assert row["filing_date"] >= "2026-09-19"
        assert row["insider_name"]


def _index(result, accession: str) -> int:
    for i, row in enumerate(result.data["transactions"]):
        if row["accession_number"] == accession:
            return i
    raise AssertionError(accession)


@pytest.fixture
def offline_owners(monkeypatch):
    """Stub the reporting-owner CIK lookup, the only fetch in this parse path."""
    import edgar.ownership.owners as owners_module

    class StubEntity:
        def __init__(self, cik):
            # Berkshire Hathaway Inc is a company, so its name is not reversed.
            self.data = type("Data", (), {"is_company": int(cik) == BERKSHIRE_CIK})()

    monkeypatch.setattr(owners_module, "Entity", StubEntity)


def test_derivative_only_filing_keeps_fractional_shares(offline_owners):
    """Vertex 0001209191-20-055264 has no non-derivative table and 196.356 units."""
    from edgar.ai.mcp.tools.ownership import _owner_fields, _reported_trades
    from edgar.ownership.forms import Form4

    form4 = Form4.parse_xml(VERTEX_XML.read_text())
    assert len(form4.non_derivative_table.transactions) == 0
    assert len(form4.derivative_table.transactions) == 1

    trades = _reported_trades(form4)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["derivative"] is True
    assert trade["code"] == "A"
    assert trade["action"] == "A"
    assert trade["type"] == "Award"
    assert trade["shares"] == 196.356
    assert isinstance(trade["shares"], float)
    assert trade["price"] == pytest.approx(222.81)
    assert trade["security"] == "Deferred Stock Units"
    assert trade["underlying"] == "Common Stock"
    assert trade["date"] == "2020-10-15"

    owner = _owner_fields(form4)
    assert owner["insider_name"] == "Bruce I Sachs"
    assert owner["is_director"] is True
    assert owner["is_officer"] is False
    assert owner["role"] == "Director"


def test_joint_filing_scalars_describe_the_first_owner_only(offline_owners):
    """Berkshire and Buffett stay separate owners; the scalars are Berkshire's."""
    from edgar.ai.mcp.tools.ownership import _owner_fields
    from edgar.ownership.forms import Form3

    form3 = Form3.parse_xml(JOINT_XML.read_text())
    assert len(form3.reporting_owners.owners) == 2
    # The combined strings are what the scalars used to copy.
    # `insider_name` joins with " / "; `position` joins with "/ ".
    assert form3.insider_name == "BERKSHIRE HATHAWAY INC / Warren E Buffett"
    assert form3.position == "10% Owner/ 10% Owner"

    fields = _owner_fields(form3)
    assert fields["insider_name"] == "BERKSHIRE HATHAWAY INC"
    assert fields["role"] == "10% Owner"
    assert fields["title"] == "10% Owner"
    assert fields["is_ten_pct_owner"] is True
    assert fields["is_officer"] is False
    assert fields["is_director"] is False
    assert fields["insider_name"] == fields["owners"][0]["insider_name"]
    assert fields["is_ten_pct_owner"] is fields["owners"][0]["is_ten_pct_owner"]
    assert " / " not in fields["role"]

    buffett = fields["owners"][1]
    assert buffett["insider_name"] == "Warren E Buffett"
    assert buffett["role"] == "10% Owner"
    assert buffett["is_ten_pct_owner"] is True
    assert buffett["is_officer"] is False


class _Filing:
    def __init__(self, accession: str):
        self.filing_date = "2026-10-01"
        self.accession_number = accession
        self.parsed = False

    def obj(self):
        self.parsed = True

        class _Owner:
            name = "Stub Owner"
            position = "Director"
            officer_title = None
            is_officer = False
            is_director = True
            is_ten_pct_owner = False
            is_other = False

        class _Owners:
            owners = [_Owner()]

        class _Trade:
            transaction_type = "Sale"
            transaction_code = "S"
            acquired_disposed = "D"
            shares = 10
            price = 1.0
            security = "Common"
            date = "2026-10-01"

        class _Rows:
            def __init__(self):
                self._rows = [_Trade()]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return self._rows[index]

        class _Table:
            has_transactions = True
            transactions = _Rows()

        class _Form:
            reporting_owners = _Owners()
            non_derivative_table = _Table()
            derivative_table = None

        return _Form()


class _Company:
    name = "NVIDIA CORP"
    cik = 1045810


@pytest.mark.asyncio
async def test_limit_stops_parsing_once_the_cap_is_filled(monkeypatch):
    """`limit` is applied to filings already inside the lookback, and it stops."""
    from edgar.ai.mcp.tools import ownership

    filings = [_Filing(f"stub-{index}") for index in range(1, 5)]
    monkeypatch.setattr(ownership, "resolve_company", lambda identifier: _Company())
    monkeypatch.setattr(ownership, "_form4s_in_lookback", lambda company, days: filings)

    result = await ownership._get_insider_transactions(NVIDIA_CIK, 41, 2)

    assert result.success is True
    assert result.data["transaction_count"] == 2
    assert [row["accession_number"] for row in result.data["transactions"]] == [
        "stub-1",
        "stub-2",
    ]
    assert [filing.parsed for filing in filings] == [True, True, False, False]


class _Column:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return list(self._values)


class _Frame:
    def __init__(self, dates):
        self._dates = list(dates)
        self.data = self
        self.filtered_window = None

    def column(self, name):
        assert name == "filing_date"
        return _Column(self._dates)

    def __len__(self):
        return len(self._dates)

    def filter(self, filing_date=None):
        self.filtered_window = filing_date
        return self


def test_lookback_paginates_only_when_the_recent_page_misses_the_cutoff(monkeypatch):
    """Older submissions files load when the recent page does not reach `days`."""
    from edgar.ai.mcp.tools import ownership

    monkeypatch.setattr(ownership, "_today", lambda: date(2026, 10, 29))
    # Newest filing 2026-10-13, oldest on this page 2026-09-18.
    recent = _Frame(["2026-10-13", "2026-09-18"])
    older = _Frame(["2026-10-13", "2026-08-01"])
    calls = []

    class Company:
        def get_filings(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("trigger_full_load"):
                return older
            return recent

    # 41 days before 2026-10-29 is 2026-09-18, which the recent page already holds.
    covered = ownership._form4s_in_lookback(Company(), 41)
    assert calls == [{"form": "4", "trigger_full_load": False}]
    assert covered.filtered_window == "2026-09-18:"

    calls.clear()
    # 60 days before 2026-10-29 is 2026-08-30, past the recent page.
    paged = ownership._form4s_in_lookback(Company(), 60)
    assert calls[0] == {"form": "4", "trigger_full_load": False}
    assert calls[1]["form"] == "4"
    assert calls[1]["trigger_full_load"] is True
    assert calls[1]["filing_date"] == "2026-08-30:"
    assert paged is older
    assert date(2026, 10, 29) - timedelta(days=60) == date(2026, 8, 30)
