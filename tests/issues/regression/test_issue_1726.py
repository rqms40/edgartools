"""ixt-sec:exchnameen returned exchange codes outside the DEI enumeration.

GitHub Issue: https://github.com/dgunning/edgartools/issues/1726

``dei:SecurityExchangeName`` is ``dei:edgarExchangeCodeItemType``. Enumeration
tokens match exactly, including case. Checked against the SEC DEI schemas for
2021, 2021q4, 2022, 2022q4, 2023, 2024, 2025, 2026 and 2027
(https://xbrl.sec.gov/dei/2026/dei-2026.xsd and the same path in each year).

Corrected, because the old token is absent from every one of those lists and
the replacement is the token the schema does list:

- ``NYSE Arca`` -> ``NYSEArca`` (was ``NYSEARCA``). ``NYSEArca`` is in every year.
- ``NYSE Texas`` -> ``NYSETX`` (was ``NYSETXSE``). ``NYSETX`` is in the 2026 and
  2027 schemas; 2021-2025 predate that code.
- ``MIAX Pearl`` -> ``PEARL`` (was ``MIAXPEARL``). The enumeration's token is
  ``PEARL``.

Left unchanged after the same check. ``NYSE``, ``NYSEAMER``, ``NYSENAT``,
``NASDAQ``, ``CboeBZX``, ``CboeBYX``, ``CboeEDGX``, ``CboeEDGA``, ``C2``,
``CBOE``, ``CHX``, ``IEX``, ``MIAX``, ``BOX`` and ``BX`` (the ``bse`` key) are
enumeration tokens in every year above. ``nyse chicago`` still maps to
``NYSECHX``, which is not a token: the Chicago exchange is ``CHX`` through the
2025 schema, and the 2026 schema adds ``NYSETX`` for the rename of that
exchange to NYSE Texas, so which token a filer who still writes "NYSE Chicago"
should get is not settled by the schema alone. ``long term stock exchange``
(``LTSE``) and ``members exchange`` / ``memx`` (``MEMX``) are not tokens in any
year checked, and the schemas name no replacement.

Ground truth, read from SEC on 2026-10-25. JPMorgan Chase & Co. 10-Q for the
period ended 2026-06-30, accession 0001628280-26-054343, primary document
https://www.sec.gov/Archives/edgar/data/19617/000162828026054343/jpm-20260630.htm.
Facts f-39 (context c-10) and f-42 (context c-11) are
``dei:SecurityExchangeName``, format ``ixt-sec:exchnameen``, display text
"NYSE Arca, Inc." They are the cover-page exchange for the AMJB and VYLD
guarantees. The SEC extracted instance
https://www.sec.gov/Archives/edgar/data/19617/000162828026054343/jpm-20260630_htm.xml
stores ``NYSEArca`` for those two securities. The elements below are the
outer HTML of f-39 and f-42 from that primary document.
"""

import pytest

from edgar.documents import ParserConfig, parse_html
from edgar.documents.strategies.ixbrl_transforms import TransformError, apply_transform

# Outer HTML of the two dei:SecurityExchangeName facts written "NYSE Arca, Inc."
# in jpm-20260630.htm. The HTML parser lowercases the tag and the attribute
# names; the values are the ones on the filing.
_JPM_ARCA_FACTS = """\
<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><body>
<ix:nonnumeric contextref="c-10" name="dei:SecurityExchangeName" format="ixt-sec:exchnameen" id="f-39">NYSE Arca, Inc.</ix:nonnumeric>
<ix:nonnumeric contextref="c-11" name="dei:SecurityExchangeName" format="ixt-sec:exchnameen" id="f-42">NYSE Arca, Inc.</ix:nonnumeric>
</body></html>"""


@pytest.mark.fast
def test_nyse_arca_and_nyse_texas_use_dei_enumeration_tokens():
    """The two codes the report names, plus MIAX Pearl, which failed the same check."""
    assert apply_transform('ixt-sec:exchnameen', 'NYSE Arca, Inc.') == 'NYSEArca'
    assert apply_transform('ixt-sec:exchnameen', 'NYSE Texas') == 'NYSETX'
    assert apply_transform('ixt-sec:exchnameen', 'MIAX Pearl') == 'PEARL'


@pytest.mark.fast
def test_an_unknown_exchange_name_is_rejected():
    """A name with no code raises; it is not returned unchanged and not invented."""
    with pytest.raises(TransformError):
        apply_transform('ixt-sec:exchnameen', 'Not A Registered Exchange')


@pytest.mark.fast
def test_jpmorgan_10q_arca_facts_parse_to_nysearca():
    """Facts f-39 and f-42 from accession 0001628280-26-054343 parse to NYSEArca.

    The SEC extracted instance stores NYSEArca for the two cover-page securities
    these facts name (AMJB and VYLD), whose visible text is "NYSE Arca, Inc."
    """
    doc = parse_html(_JPM_ARCA_FACTS, ParserConfig(extract_xbrl=True, detect_sections=False))
    facts = [f for f in doc.metadata.xbrl_data if f.concept == 'dei:SecurityExchangeName']

    assert [(f.context_ref, f.value) for f in facts] == [
        ('c-10', 'NYSEArca'),
        ('c-11', 'NYSEArca'),
    ]
    assert all(f.metadata is None or 'format_issue' not in (f.metadata or {}) for f in facts)
