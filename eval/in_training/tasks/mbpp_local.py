"""Prompt, few-shot examples and scoring of the in-training MBPP task (itr_mbpp.yaml).

The prompt and the three few-shot examples are those of the stock mbpp task of lm-eval 0.4.13
(lm_eval/tasks/mbpp/mbpp.yaml and utils.py). They are copied here instead of imported, because
importing lm_eval.tasks.mbpp.utils loads evaluate's code_eval metric and runs a test program
through it at import time. tests/train/test_train_eval.py checks that the copy still equals the files of
the pinned release; it reads them with ast and yaml and never imports them.

Two things differ from the stock task. Here, the test program of a problem runs the problem's
test_setup_code before its asserts. The stock task leaves that code out. In the test split only
task 367 has such code: it builds the trees that the asserts check, so without it the problem fails
whatever the model writes. In itr_mbpp.yaml, the generation budget is 512 tokens, because the stock
task sets none and lm-eval's vLLM backend then stops every answer at 256 tokens.

The generated programs run through evaluate's code_eval metric, as HumanEval's do
(humaneval_local.py), with the same limit of 3 seconds per program.
"""
import functools

# code_eval's default limit, which HumanEval runs with as well. It is passed explicitly, so the
# protocol does not change if the metric's default ever does.
TIMEOUT_SECONDS = 3.0


def doc_to_text(doc):
    """The stock prompt: the task, its first three asserts, and [BEGIN] on a line of its own."""
    tests = "\n".join(doc["test_list"][:3])
    return (
        f"You are an expert Python programmer, and here is your task: {doc['text']} "
        f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    )


def doc_to_target(doc):
    """A few-shot example's solution followed by [DONE], or the test program of a test problem.

    The stock template tells the two apart by whether the document has an is_fewshot key, and so
    does this function.
    """
    if "is_fewshot" in doc:
        return f"{doc['code']}\n[DONE]"
    return check_code(doc)


def check_code(doc):
    """The code that checks a solution: the problem's test_setup_code, if any, then its asserts."""
    parts = [doc.get("test_setup_code") or "", *doc["test_list"][:3]]
    return "\n".join(part for part in parts if part)


def build_predictions(resps, docs):
    """The program to run for each response, which is the generation itself, as in the stock task.

    lm-eval has already cut each generation at the stop string [DONE]. The responses of a document
    stay in one list, because code_eval takes a list of candidate programs per problem.
    """
    return [list(responses) for responses in resps]


@functools.lru_cache(maxsize=1)
def _code_eval():
    # The metric is loaded when the first program is scored, not when lm-eval imports this module,
    # so the import has no side effects. The stock module runs a test program at import instead,
    # to find out early whether code can run at all. Here the trainer's startup check covers most
    # of that: it imports evaluate and starts a multiprocessing manager, and every evaluation
    # process gets HF_ALLOW_CODE_EVAL=1.
    #
    # The startup check does not load the code_eval metric itself. If the offline cache lacks it,
    # MBPP's process fails here, after its generation, and not when its task loads. That wastes one
    # process per pause. No other benchmark is lost, because MBPP runs in processes of its own.
    import evaluate

    return evaluate.load("code_eval")


def pass_at_k(references, predictions, k=None):
    """pass@k of the predicted programs against the test programs, from evaluate's code_eval.

    It returns a dict such as {"pass@1": 1.0}, as the stock HumanEval scorer does, and lm-eval
    reports each key under the filter's name, here "pass@1,create_test".
    """
    if isinstance(k, int):
        k = [k]
    scores, _ = _code_eval().compute(
        references=references, predictions=predictions, k=k, timeout=TIMEOUT_SECONDS,
    )
    return scores


def list_fewshot_samples():
    # A verbatim copy of list_fewshot_samples in lm_eval/tasks/mbpp/utils.py of lm-eval 0.4.13:
    # MBPP task_ids 2, 3 and 4, from the dataset's prompt split.
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
