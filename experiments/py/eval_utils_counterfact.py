"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast, generate_mine, load_pipeline
from util.perplexity import perplexity

# if torch.cuda.device_count()==2:
#     device = 1
# elif torch.cuda.device_count()==1:
#     device = 0
# else:
#     device = "cpu"

def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    language: str,
    device, 
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: Related to measuring semantic consistency in generation prompts
    in our case this feature is disabled
    :param vec: Related to TF IDF for snips

    :return: Dictionary containing rewriting metrics
    """

    if language == "catalan":
        # First, unpack rewrite evaluation record.

        # Both subjects should be the same, but the gender may change
        subject_new = record["eval_target_new"]["requested_rewrite"]["subject"]
        subject_true = record["eval_target_true"]["requested_rewrite"]["subject"] 

        target_new = record["eval_target_new"]["requested_rewrite"]["target_new"]
        target_true = record["eval_target_true"]["requested_rewrite"]["target_true"]

        rewrite_prompts_new = [record["eval_target_new"]["requested_rewrite"]["prompt"].format(subject_new)]
        paraphrase_prompts_new = record["eval_target_new"]["paraphrase_prompts"]
        neighborhood_prompts_new = record["eval_target_new"]["neighborhood_prompts"]
        generation_prompts_new = record["eval_target_new"]["generation_prompts"]

        rewrite_prompts_true = [record["eval_target_true"]["requested_rewrite"]["prompt"].format(subject_true)]
        paraphrase_prompts_true = record["eval_target_true"]["paraphrase_prompts"]
        neighborhood_prompts_true = record["eval_target_true"]["neighborhood_prompts"]
        generation_prompts_true = record["eval_target_true"]["generation_prompts"]

        # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts_true,
            rewrite_prompts_new,
            paraphrase_prompts_true,
            paraphrase_prompts_new,
            neighborhood_prompts_true,
            neighborhood_prompts_new
        ]

        which_correct = [
            [0 for _ in range(len(rewrite_prompts_new))],
            [0 for _ in range(len(paraphrase_prompts_new))],
            [1 for _ in range(len(neighborhood_prompts_new))],
        ]

        probs, targets_correct = test_batch_prediction_catalan(
            model,
            tok,
            list(chain(*prob_prompts)),
            list(chain(*which_correct)),
            target_new["str"],
            target_true["str"],
            device,
        )

    elif language == "english":
        # First, unpack rewrite evaluation record.
        subject, target_new, target_true = (
            record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
        )
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]
        neighborhood_prompts = record["neighborhood_prompts"]
        generation_prompts = record["generation_prompts"]

        # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
        ]

        which_correct = [
            [0 for _ in range(len(rewrite_prompts))],
            [0 for _ in range(len(paraphrase_prompts))],
            [1 for _ in range(len(neighborhood_prompts))],
        ]

        # Flatten all the evaluated prefixes into one list.
        probs, targets_correct = test_batch_prediction_english(
            model,
            tok,
            list(chain(*prob_prompts)),
            list(chain(*which_correct)),
            target_new["str"],
            target_true["str"],
            device,
        )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, which_correct))).tolist() # Here I have changed prob_prompts by which_correct. It should not affect anything
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        if language == "catalan":
            essence_texts = [
                x["text"]
                for x in snips[rel_id][target_new["id"]]
                if x["name"] == record["eval_target_new"]["requested_rewrite"]["subject"]
            ]
        elif language == "english":
            essence_texts = [
                x["text"]
                for x in snips[rel_id][target_new["id"]]
                if x["name"] == record["requested_rewrite"]["subject"]
            ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts_new,
            consistency_texts,
            essence_texts,
            vec,
            device,
        )
        ret.update(gen_stats)

    return ret


def compute_rewrite_quality_counterfact_new(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    language: str,
    device, 
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: Related to measuring semantic consistency in generation prompts
    in our case this feature is disabled
    :param vec: Related to TF IDF for snips

    :return: Dictionary containing rewriting metrics
    """

    if language == "catalan":
        # First, unpack rewrite evaluation record.

        # Both subjects should be the same, but the gender may change
        subject_new = record["eval_target_new"]["requested_rewrite"]["subject"]
        subject_true = record["eval_target_true"]["requested_rewrite"]["subject"] 

        target_new = record["eval_target_new"]["requested_rewrite"]["target_new"]
        target_true = record["eval_target_true"]["requested_rewrite"]["target_true"]

        rewrite_prompts_new = [record["eval_target_new"]["requested_rewrite"]["prompt"].format(subject_new)]
        paraphrase_prompts_new = record["eval_target_new"]["paraphrase_prompts"]
        neighborhood_prompts_new = record["eval_target_new"]["neighborhood_prompts"]
        generation_prompts_new = record["eval_target_new"]["generation_prompts"]

        rewrite_prompts_true = [record["eval_target_true"]["requested_rewrite"]["prompt"].format(subject_true)]
        paraphrase_prompts_true = record["eval_target_true"]["paraphrase_prompts"]
        neighborhood_prompts_true = record["eval_target_true"]["neighborhood_prompts"]
        generation_prompts_true = record["eval_target_true"]["generation_prompts"]

        # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts_true,
            rewrite_prompts_new,
            paraphrase_prompts_true,
            paraphrase_prompts_new,
            neighborhood_prompts_true,
            neighborhood_prompts_new
        ]

        which_correct = [
            [1 for _ in range(len(rewrite_prompts_new))],
            [1 for _ in range(len(paraphrase_prompts_new))],
            [1 for _ in range(len(neighborhood_prompts_new))],
        ]

        probs, targets_correct = test_batch_prediction_catalan(
            model,
            tok,
            list(chain(*prob_prompts)),
            list(chain(*which_correct)),
            target_new["str"],
            target_true["str"],
            device,
        )

    elif language == "english":
        # First, unpack rewrite evaluation record.
        subject, target_new, target_true = (
            record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
        )
        rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
        paraphrase_prompts = record["paraphrase_prompts"]
        neighborhood_prompts = record["neighborhood_prompts"]
        generation_prompts = record["generation_prompts"]

        # Form a list of lists of prefixes to test.
        prob_prompts = [
            rewrite_prompts,
            paraphrase_prompts,
            neighborhood_prompts,
        ]

        which_correct = [
            [1 for _ in range(len(rewrite_prompts))],
            [1 for _ in range(len(paraphrase_prompts))],
            [1 for _ in range(len(neighborhood_prompts))],
        ]

        # Flatten all the evaluated prefixes into one list.
        probs, targets_correct = test_batch_prediction_english(
            model,
            tok,
            list(chain(*prob_prompts)),
            list(chain(*which_correct)),
            target_new["str"],
            target_true["str"],
            device,
        )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, which_correct))).tolist() # Here I have changed prob_prompts by which_correct. It should not affect anything
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        if language == "catalan":
            essence_texts = [
                x["text"]
                for x in snips[rel_id][target_new["id"]]
                if x["name"] == record["eval_target_new"]["requested_rewrite"]["subject"]
            ]
        elif language == "english":
            essence_texts = [
                x["text"]
                for x in snips[rel_id][target_new["id"]]
                if x["name"] == record["requested_rewrite"]["subject"]
            ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts_new,
            consistency_texts,
            essence_texts,
            vec,
            device,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction_english(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
    device,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct

def test_batch_prediction_catalan(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
    device,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]


    all_prompts = []
    targets = [target_new, target_true]
    for i, prefix in enumerate(prefixes):
        all_prompts.append(prefix + f" {targets[i%2]}")

    prompt_tok = tok(
        all_prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

                if logits[i, prefix_lens[i] + j - 1, :].argmax().item() != cur_tok:
                    correct = False
                    break
            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
    device
):

    pipeline_base = load_pipeline(model, tok, device)
    gen_texts = generate_mine(model, tok, prefixes, pipeline_base)

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
