from asr_router.glossary import Glossary


DEFAULT = {"terms": [{"term": "Alpha Group", "aliases": ["Alpa Group"]}]}
PERJOB = {
    "terms": [
        {"term": "Chairman Zhang", "aliases": ["Mr. Zhang"]},
        {"term": "Alpha Group", "aliases": ["Alfa"]},
    ]
}


def test_merge_unions_aliases():
    g = Glossary.merged(DEFAULT, PERJOB)
    alpha = g["Alpha Group"]
    assert "Alpa Group" in alpha
    assert "Alfa" in alpha


def test_perjob_adds_new():
    g = Glossary.merged(DEFAULT, PERJOB)
    assert "Chairman Zhang" in g
    assert "Mr. Zhang" in g["Chairman Zhang"]


def test_to_prompt_text():
    g = Glossary.merged(DEFAULT, {})
    text = g.to_prompt_text()
    assert "Alpha Group" in text
    assert "Alpa Group" in text


def test_empty_glossary():
    g = Glossary.merged()
    assert g.to_prompt_text() == "(none)"


def test_none_sources_ignored():
    """`merged(None, {...}, None)` should treat Nones as no-ops."""
    g = Glossary.merged(None, DEFAULT, None)
    assert "Alpha Group" in g


def test_term_with_no_aliases():
    g = Glossary.merged({"terms": [{"term": "Example City"}]})
    assert "Example City" in g
    assert g["Example City"] == []
    assert "Example City" in g.to_prompt_text()
