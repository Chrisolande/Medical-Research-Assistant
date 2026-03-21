from medical_graph_rag.pubmed_downloader import PubMedEntrezDownloader


def test_sync_search_pubmed_empty_query_uses_default_term(monkeypatch):
    downloader = PubMedEntrezDownloader(email="test@example.com")

    class DummyHandle:
        def close(self):
            return None

    def fake_esearch(**kwargs):
        assert kwargs["term"] == "research[Title/Abstract]"
        return DummyHandle()

    monkeypatch.setattr(
        "medical_graph_rag.pubmed_downloader.Entrez.esearch", fake_esearch
    )
    monkeypatch.setattr(
        "medical_graph_rag.pubmed_downloader.Entrez.read", lambda _h: {"IdList": ["1"]}
    )
    result = downloader._sync_search_pubmed("", 10, None, None, "relevance", None)
    assert result == ["1"]


def test_parse_article_populates_core_fields():
    downloader = PubMedEntrezDownloader(email="test@example.com")
    record = {
        "MedlineCitation": {
            "PMID": "42",
            "Article": {
                "Abstract": {"AbstractText": ["line one", "line two"]},
                "Journal": {
                    "Title": "J",
                    "JournalIssue": {"PubDate": {"Year": "2024"}},
                },
                "PublicationTypeList": ["Review"],
            },
            "MeshHeadingList": [{"DescriptorName": "Concept"}],
        }
    }

    parsed = downloader._parse_article(record)
    assert parsed["pmid"] == "42"
    assert "line one line two" in parsed["abstract"]
    assert parsed["mesh_terms"] == "Concept"


def test_sync_fetch_batch_handles_exception(monkeypatch):
    downloader = PubMedEntrezDownloader(email="test@example.com")

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "medical_graph_rag.pubmed_downloader.Entrez.efetch", raise_error
    )
    result = downloader._sync_fetch_batch(["1"])
    assert result == []
