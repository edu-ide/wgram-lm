from wgram_lm.eval.general_answer_interface import (
    answer_kind,
    extract_answer_candidate_text,
    normalize_answer_text,
    select_candidate,
    summarize_records,
)


def test_normalize_general_answer_text_without_solving_task():
    assert normalize_answer_text("Final Answer: 008017.") == "8017"
    assert normalize_answer_text(" 408, 404 ") == "408,404"
    assert normalize_answer_text("answer: true") == "TRUE"
    assert normalize_answer_text("Green.") == "green"
    assert normalize_answer_text("EMPTY") == "EMPTY"


def test_extract_answer_candidate_text_from_common_model_completions():
    assert extract_answer_candidate_text("The answer is 8017.") == "8017"
    assert extract_answer_candidate_text("Reasoning...\nFinal answer: 8008, 8004.") == "8008, 8004"
    assert extract_answer_candidate_text("After mapping twice:\ngreen") == "green"
    assert normalize_answer_text("The final answer is FALSE.") == "FALSE"


def test_select_candidate_uses_aliases_and_keeps_oracle_index():
    selection = select_candidate(
        ["wrong", " Answer: 408, 404.", "408,404"],
        ["408,404", "404,408"],
        selection_mode="oracle",
    )

    assert selection.exact is True
    assert selection.oracle_exact is True
    assert selection.selected_index == 1
    assert selection.selection_mode == "oracle"
    assert selection.normalized_selected == "408,404"


def test_select_candidate_first_mode_separates_accuracy_from_candidate_coverage():
    selection = select_candidate(
        ["wrong", " Answer: 408, 404."],
        ["408,404"],
        selection_mode="first",
    )

    assert selection.exact is False
    assert selection.oracle_exact is True
    assert selection.selected_index == 0
    assert selection.oracle_index == 1
    assert selection.selection_mode == "first"


def test_select_candidate_falls_back_to_first_without_task_executor():
    selection = select_candidate(["blue", "red"], ["green"])

    assert selection.exact is False
    assert selection.oracle_exact is False
    assert selection.selected_index == 0
    assert selection.normalized_selected == "blue"


def test_answer_kind_covers_stage59_formats():
    assert answer_kind("7") == "single_digit"
    assert answer_kind("8017") == "integer"
    assert answer_kind("408,404") == "csv_integer_list"
    assert answer_kind("FALSE") == "boolean"
    assert answer_kind("green") == "symbolic_or_text"


def test_summarize_records_groups_by_family_and_kind():
    summary = summarize_records(
        [
            {"exact": True, "task_family": "arithmetic_chain", "answer_kind": "integer"},
            {"exact": False, "task_family": "arithmetic_chain", "answer_kind": "integer"},
            {"exact": True, "task_family": "list_transform", "answer_kind": "csv_integer_list"},
        ]
    )

    assert summary["accuracy"] == 2 / 3
    assert summary["by_family"]["arithmetic_chain"]["accuracy"] == 0.5
    assert summary["by_kind"]["csv_integer_list"]["accuracy"] == 1.0
