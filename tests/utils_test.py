import json
from pathlib import Path

from langchain_core.documents import Document

from medical_graph_rag.utils import (
    _clean_json_candidate,
    _extract_line_based_concepts,
    _extract_quoted_fallback,
    calculate_edge_weight,
    create_batches,
    create_text_hash,
    extract_and_parse_json,
    load_json_data,
    pretty_print_docs,
    print_filtered_content,
    save_processing_results,
)


def test_clean_json_candidate_normalizes_quotes_and_trailing_comma():
    cleaned = _clean_json_candidate(" junk {'a': 'b',} tail ")
    assert cleaned == '{"a": "b"}'


def test_extract_line_based_concepts_supports_single_and_list():
    text = '"0": "diabetes"\n1: ["heart", "lung"]'
    concepts = _extract_line_based_concepts(text)
    assert concepts["0"] == ["diabetes"]
    assert concepts["1"] == ["heart", "lung"]


def test_extract_quoted_fallback_groups_quotes():
    concepts = _extract_quoted_fallback('"a" "b" "c" "d"')
    assert concepts["0"] == ["a", "b", "c"]


def test_extract_and_parse_json_with_direct_json():
    parsed = extract_and_parse_json('{"0": ["a", "b"]}')
    assert parsed == {"0": ["a", "b"]}


def test_extract_and_parse_json_with_fallback_line_parsing():
    parsed = extract_and_parse_json('"0": "term"')
    assert parsed == {"0": ["term"]}


def test_pretty_print_docs_for_single_and_batch(capsys):
    docs = [Document(page_content="one", metadata={"id": 1})]
    pretty_print_docs(docs, wrap_width=40)
    out = capsys.readouterr().out
    assert "Document 1" in out

    pretty_print_docs([docs], wrap_width=40, queries=["q"])
    out = capsys.readouterr().out
    assert "QUERY 1 RESULTS: q" in out


def test_print_filtered_content_outputs_steps(capsys):
    print_filtered_content([1], {1: "content text"}, content_preview_length=20)
    out = capsys.readouterr().out
    assert "Step 1 - Node 1" in out


def test_load_json_data_list_and_single_list_dict(tmp_path: Path):
    p1 = tmp_path / "list.json"
    p1.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert len(load_json_data(str(p1))) == 2
    assert len(load_json_data(str(p1), max_items=1)) == 1

    p2 = tmp_path / "dict.json"
    p2.write_text(json.dumps({"docs": [{"a": 1}]}), encoding="utf-8")
    assert load_json_data(str(p2)) == [{"a": 1}]


def test_load_json_data_invalid_structure_returns_empty(tmp_path: Path):
    p = tmp_path / "invalid.json"
    p.write_text(json.dumps({"a": 1, "b": 2}), encoding="utf-8")
    assert load_json_data(str(p)) == []


def test_create_batches_and_hash_and_edge_weight():
    assert list(create_batches([1, 2, 3], 2)) == [[1, 2], [3]]
    assert create_text_hash("abc") == create_text_hash("abc")
    weight = calculate_edge_weight(0.8, ["a"], ["a", "b"], ["a"])
    assert 0 < weight <= 1


def test_save_processing_results_with_batch_details(tmp_path: Path):
    docs = [
        Document(page_content="doc1", metadata={"m": 1}),
        Document(page_content="doc2", metadata={"m": 2}),
    ]
    results = {
        "processing_summary": {"total_batches": 1},
        "all_documents": docs,
        "successful_batches": [
            {
                "batch_num": 1,
                "original_count": 2,
                "chunk_count": 2,
                "documents": docs,
            }
        ],
    }
    save_processing_results(
        results=results,
        output_dir=str(tmp_path),
        base_filename="out",
        batch_size=2,
        source_type="pmc",
        save_batch_details=True,
    )
    assert (tmp_path / "out.json").exists()
    assert (tmp_path / "batch_details" / "batch_001.json").exists()
