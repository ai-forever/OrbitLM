"""Budget forcing: which responses get a second call, what is inserted, and how a response is split."""
from __future__ import annotations

import pytest

from inference.budget import Budget, Completion, Response, generate_responses, make_budget, split_response
from orbit_verl.data.response_info import STOP_THINKING_TEXT, STOP_THINKING_TEXT_RU, STOP_THINKING_TEXTS

CLOSE, OPEN, END = 9, 8, 7
EN_IDS, RU_IDS = (101, 102, CLOSE), (201, 202, CLOSE)
BUDGET = Budget(think_tokens=4, answer_tokens=3, close_id=CLOSE,
                stop_ids_by_text={STOP_THINKING_TEXTS["en"]: EN_IDS, STOP_THINKING_TEXTS["ru"]: RU_IDS})
PROMPT = [1, 2, 3]


def run(engine, prompts, langs, budget=BUDGET, max_model_len=100, max_new_tokens=None):
    return generate_responses(prompts, langs, engine, max_new_tokens=max_new_tokens, max_model_len=max_model_len,
                              budget=budget)


def test_a_response_that_ends_within_the_budget_gets_one_call(scripted_engine):
    engine = scripted_engine(lambda p, limit, call: Completion((5, CLOSE, 6, END), "stop"))
    [response] = run(engine, [PROMPT], ["en"])
    assert len(engine.calls) == 1
    assert engine.calls[0][1] == [BUDGET.think_tokens]
    assert not response.forced_close and not response.continued
    assert response.token_ids == [5, CLOSE, 6, END]
    assert response.finish_reason == "stop"


@pytest.mark.parametrize("lang, inserted", [("en", EN_IDS), ("ru", RU_IDS), (None, EN_IDS), ("de", EN_IDS)])
def test_an_open_block_at_the_budget_is_closed_with_the_sentence_of_its_language(scripted_engine, lang, inserted):
    def script(prompt, limit, call):
        return Completion((5,) * limit, "length") if call == 0 else Completion((42, END), "stop")

    engine = scripted_engine(script)
    [response] = run(engine, [PROMPT], [lang])
    second_prompts, second_limits = engine.calls[1]
    assert second_prompts == [PROMPT + [5, 5, 5, 5] + list(inserted)]
    assert second_limits == [BUDGET.answer_tokens]
    assert response.forced_close and not response.continued
    assert response.inserted_ids == inserted
    assert response.token_ids == [5, 5, 5, 5, *inserted, 42, END]
    assert response.generated_tokens == 6
    assert response.finish_reason == "stop"


def test_a_closed_block_cut_while_answering_is_continued_without_an_insert(scripted_engine):
    def script(prompt, limit, call):
        return Completion((5, CLOSE, 6, 6), "length") if call == 0 else Completion((6, END), "stop")

    engine = scripted_engine(script)
    [response] = run(engine, [PROMPT], ["ru"])
    assert engine.calls[1][0] == [PROMPT + [5, CLOSE, 6, 6]]
    assert response.continued and not response.forced_close
    assert response.token_ids == [5, CLOSE, 6, 6, 6, END]


def test_only_the_responses_cut_at_the_budget_go_to_the_second_call_in_order(scripted_engine):
    firsts = {10: Completion((5, END), "stop"), 20: Completion((5,) * 4, "length"), 30: Completion((CLOSE, 6, 6, 6), "length")}

    def script(prompt, limit, call):
        return firsts[prompt[0]] if call == 0 else Completion((prompt[0], END), "stop")

    engine = scripted_engine(script)
    responses = run(engine, [[10], [20], [30]], ["en", "ru", "en"])
    assert [p[0] for p in engine.calls[1][0]] == [20, 30]
    assert engine.calls[1][0][0] == [20, 5, 5, 5, 5, *RU_IDS]
    assert [r.second is None for r in responses] == [True, False, False]
    assert [r.forced_close for r in responses] == [False, True, False]


def test_the_answer_call_never_runs_past_the_context(scripted_engine):
    def script(prompt, limit, call):
        return Completion((5,) * limit, "length")

    engine = scripted_engine(script)
    # 3 prompt ids + 4 thinking ids + 3 inserted ids leave 1 of 11 for the answer.
    run(engine, [PROMPT], ["en"], max_model_len=11)
    assert engine.calls[1][1] == [1]
    engine = scripted_engine(script)
    [response] = run(engine, [PROMPT], ["en"], max_model_len=10)
    assert len(engine.calls) == 1
    assert response.second is None and not response.forced_close


def test_without_a_budget_there_is_one_call_limited_by_the_context(scripted_engine):
    engine = scripted_engine(lambda p, limit, call: Completion((5,) * limit, "length"))
    responses = run(engine, [PROMPT, [1] * 95], ["en", "en"], budget=None, max_new_tokens=50)
    assert engine.calls[0][1] == [50, 5]
    assert len(engine.calls) == 1
    assert not any(r.forced_close for r in responses)


def test_a_prompt_that_fills_the_context_is_refused(scripted_engine):
    engine = scripted_engine(lambda p, limit, call: Completion((), "stop"))
    with pytest.raises(ValueError, match="fills the context"):
        run(engine, [[1] * 100], ["en"], budget=None, max_new_tokens=5)


def test_split_of_a_forced_response_leaves_out_the_inserted_sentence():
    response = Response(first=Completion((5, 6), "length"), inserted_ids=EN_IDS, second=Completion((42, END), "stop"))
    parts = split_response(response, "think", (OPEN, CLOSE), [END])
    assert (parts.think, parts.thinking_ids, parts.answer_ids) == ("closed", [5, 6], [42])


def test_split_of_natural_responses():
    closed = Response(first=Completion((5, 6, CLOSE, 42, END), "stop"))
    parts = split_response(closed, "think", (OPEN, CLOSE), [END])
    assert (parts.think, parts.thinking_ids, parts.answer_ids) == ("closed", [5, 6], [42])
    unclosed = Response(first=Completion((5, 6, END), "stop"))
    parts = split_response(unclosed, "think", (OPEN, CLOSE), [END])
    assert (parts.think, parts.thinking_ids, parts.answer_ids) == ("unclosed", [5, 6], [])
    for mode in ("no_think", "raw"):
        parts = split_response(closed, mode, (OPEN, CLOSE), [END])
        assert (parts.think, parts.thinking_ids, parts.answer_ids) == ("none", [], [5, 6, CLOSE, 42])


def test_make_budget_encodes_both_sentences_with_one_close(tokenizer, special):
    budget = make_budget(tokenizer, 100, 50)
    assert budget.close_id == special["</think>"]
    assert tokenizer.decode(list(budget.stop_ids("ru"))) == STOP_THINKING_TEXT_RU
    assert tokenizer.decode(list(budget.stop_ids("en"))) == STOP_THINKING_TEXT
    assert tokenizer.decode(list(budget.stop_ids(None))) == STOP_THINKING_TEXT
    for ids in budget.stop_ids_by_text.values():
        assert ids.count(special["</think>"]) == 1
    assert "Учитывая" in STOP_THINKING_TEXT_RU and STOP_THINKING_TEXT_RU.endswith("</think>\n\n")
