"""Manual seed supplements for papers the DOI parser can't auto-extract.

This includes:
- Papers with non-DOI URLs (conference proceedings, arxiv without doi.org links)
- Papers with incorrect/truncated DOIs in the Google Doc
- Cross-listing overrides (papers the doc explicitly places in multiple categories)
"""

# Papers whose DOIs aren't in the Google Doc text, or are incorrect there.
MANUAL_DOIS: list[dict[str, str | list[str]]] = [
    # --- Papers with non-DOI URLs in the doc ---
    {
        "doi": "10.48550/arxiv.1606.01164",
        "categories": ["point_attractor"],
        "note": "Krotov & Hopfield 2016 - Dense associative memory (NeurIPS)",
    },
    {
        "doi": "10.48550/arxiv.2008.02217",
        "categories": ["point_attractor"],
        "note": "Ramsauer et al. 2021 - Hopfield Networks is All You Need (ICLR)",
    },
    {
        "doi": "10.1002/(sici)1098-1063(1996)6:3<271::aid-hipo5>3.0.co;2-q",
        "categories": ["sequence"],
        "note": "Tsodyks et al. 1996 - Population dynamics and theta rhythm (only had PubMed link)",
    },
    {
        "doi": "10.48550/arxiv.2505.01098",
        "categories": ["general_attractor"],
        "note": "Fakhoury et al. - Models of attractor dynamics in the brain (arxiv only)",
    },
    # --- Papers with incorrect/truncated DOIs in the doc ---
    {
        "doi": "10.1101/lm.3.2-3.279",
        "categories": ["sequence"],
        "note": "Jensen & Lisman 1996 - CA3 predicts memory sequences (doc had 10.1101/lm.3.2.279)",
    },
    {
        "doi": "10.1023/a:1008820728122",
        "categories": ["sequence"],
        "note": "Gerstner & Abbott 1997 - Learning navigational maps (doc had wrong suffix)",
    },
]

# DOIs that appear in the doc but are truncated by the parser (old-style with special chars).
# These map the truncated form -> correct full DOI.
DOI_CORRECTIONS: dict[str, str] = {
    # Tsodyks 1999 - truncated at angle bracket
    "10.1002/(SICI)1098-1063(1999)9:4<481::AID-HIPO14": "10.1002/(SICI)1098-1063(1999)9:4<481::AID-HIPO14>3.0.CO;2-S",
    # Levy 1996 - truncated at angle bracket
    "10.1002/(SICI)1098-1063(1996)6:6<579::AID-HIPO3": "10.1002/(SICI)1098-1063(1996)6:6<579::AID-HIPO3>3.0.CO;2-C",
    # Hartley 2000 - truncated at angle bracket
    "10.1002/1098-1063(2000)10:4<369::AID-HIPO3": "10.1002/1098-1063(2000)10:4<369::AID-HIPO3>3.0.CO;2-0",
    # Jensen 1996 - wrong DOI in doc
    "10.1101/lm.3.2.279": "10.1101/lm.3.2-3.279",
    # Gerstner 1997 - wrong DOI in doc
    "10.1023/A:1008851126951": "10.1023/a:1008820728122",
}

# Cross-listings: papers that appear under multiple categories in the Google Doc.
# Maps DOI -> additional categories beyond the first one found by the parser.
# The parser assigns the first category; these add the rest.
CROSS_LISTINGS: dict[str, list[str]] = {
    # Cho & McClelland - listed under sequences, SR, and BTSP
    "10.64898/2025.12.25.696522": ["successor_representation", "btsp"],
    # Li & Roxin - listed under continuous_attractor and BTSP
    "10.1371/journal.pcbi.1011139": ["btsp"],
    # Spalla et al. - listed under continuous_attractor and sequences
    "10.7554/eLife.69499": ["sequence"],
    # Li et al. 2024 Cell - listed under continuous_attractor and BTSP
    "10.1016/j.cell.2024.09.041": ["btsp"],
    # Maass & Wu - listed under point_attractor and BTSP
    "10.1038/s41467-024-55563-6": ["btsp"],
    # Levenstein et al. - listed under continuous_attractor and (implicitly) BTSP
    "10.1101/2024.04.28.591528": ["btsp", "sequence"],
    # Azizi et al. - listed under continuous_attractor and sequences
    "10.3389/fncom.2013.00161": ["sequence"],
    # de Cothi & Barry - listed under SR and (arguably) bespoke
    "10.1002/hipo.23246": ["bespoke"],
    # Yiu & Leibold - listed under continuous_attractor and sequences (somewhat CAN-like)
    "10.7554/eLife.86837": ["sequence"],
}
