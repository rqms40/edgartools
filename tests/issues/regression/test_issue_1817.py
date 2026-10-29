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

GitHub Issue: https://github.com/dgunning/edgartools/issues/1817
"""

import json
from datetime import date
from unittest.mock import patch

import pytest

from edgar import set_identity

# NVIDIA Form 4, Colette Kress, filed 2026-09-18.
KRESS_ACCESSION = "0001588670-26-000014"
# A newer NVIDIA Form 4 that a 40-day window from 2026-10-29 still includes.
NEWER_ACCESSION = "0001197647-26-000011"
RECORDING_DATE = "2026-10-29"

pytestmark = [pytest.mark.network, pytest.mark.vcr]


def _by_accession(result):
    return {row["accession_number"]: row for row in result.data["transactions"]}


@pytest.mark.asyncio
async def test_insider_transactions_report_owner_role_shares_and_honor_days():
    """The Kress Form 4 names her, her title and the four share counts, and `days` drops it."""
    set_identity("Mark David Prado rqms40@users.noreply.github.com")
    from edgar.ai.mcp.tools import ownership

    # 2026-10-29 is the day the cassette was recorded. `days` is measured from
    # `_today()`, so pin that clock for replay.
    clock = patch(
        "edgar.ai.mcp.tools.ownership._today",
        return_value=date.fromisoformat(RECORDING_DATE),
    )
    # Both calls run before the assertions so a cassette recorded against the
    # broken code still contains every request the fixed code replays.
    with clock:
        wide = await ownership.edgar_ownership(identifier="NVDA", analysis_type="insiders", days=41, limit=12)
        narrow = await ownership.edgar_ownership(identifier="NVDA", analysis_type="insiders", days=40, limit=12)

    assert wide.success is True
    assert wide.data["analysis"] == "insider_transactions"
    kress = _by_accession(wide)[KRESS_ACCESSION]
    assert kress["filing_date"] == "2026-09-18"
    assert kress["insider_name"] == "Colette Kress"
    assert kress["role"] == "EVP & Chief Financial Officer"
    assert kress["title"] == "EVP & Chief Financial Officer"
    assert kress["is_officer"] is True
    assert kress["is_director"] is False
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
