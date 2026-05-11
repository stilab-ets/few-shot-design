import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from vllm_llm_logit_jinja import count_input_tokens, count_chat_style_tokens

# ============================================================
# CONFIG
# ============================================================

REPOS = [
    "facebook_react",
    "bitcoin_bitcoin",
    "opencv_opencv",
    "tensorflow_tensorflow",
    "microsoft_vscode",
]

MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", 32768),
    ("meta-llama/Llama-3.1-8B-Instruct", 131072),
    ("mistralai/Mistral-7B-Instruct-v0.3", 32768),
    ("google/gemma-3-12b-it", 32768),
]

EMBED_MODEL_NAME = "all-mpnet-base-v2"

FEW_SHOT_COUNTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]

OUTPUT_TOKEN_BUFFER = 1

BASE_OUTPUT_DIR = "../rqs/similarity_selection_RAG"

encoder = SentenceTransformer(EMBED_MODEL_NAME)

SYSTEM_PROMPT = (
    "Your job is to analyze and assign the most fitting labels to GitHub issues. "
    "Identify and assign the most accurate label for the issue using its title and body. "
    "Select the label exclusively from the options given in this list: "
    "\"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one "
    "from the list. Output the label as a single string format with no additional "
    "characters, text, or formatting."
)

# ============================================================
# HELPERS
# ============================================================

def load_data(repo):
    base = f"../data/{repo}"
    with open(f"{base}/{repo}_train_issues_normalized.json") as f:
        train = json.load(f)
    with open(f"{base}/{repo}_test_issues_normalized.json") as f:
        test = json.load(f)
    return train, test


def embed(repo, data, name):
    path = f"../rqs/cache/{repo}_{name}.npy"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return np.load(path)

    texts = [f"{x['title']} {x['body']}" for x in data]
    emb = encoder.encode(texts, convert_to_numpy=True)
    np.save(path, emb)
    return emb


def compute_tokens(model_name, few_shots, issue):
    user_prompt = f"Title: {issue['title']}\nBody: {issue['body']}"
    return (
        count_chat_style_tokens(model_name, SYSTEM_PROMPT, few_shots)
        + count_input_tokens(model_name, user_prompt)
        + OUTPUT_TOKEN_BUFFER
    )


def example_length(idx, train_data):
    return len(train_data[idx]["title"] + train_data[idx]["body"])

# ============================================================
# EXACT-k PROMPT BUILDER
# ============================================================

def build_prompt_exact_k(
    sorted_idx,
    k,
    issue,
    train_data,
    model_name,
    context_limit,
    test_idx,
    repo
):

    selected = [int(i) for i in sorted_idx[:k]]
    overflow = [int(i) for i in sorted_idx[k:]]

    overflow_ptr = 0

    while True:

        few_shots = [
            {
                "input": f"Title: {train_data[i]['title']}\nBody: {train_data[i]['body']}",
                "output": train_data[i]["labels"].strip().lower()
            }
            for i in selected
        ]

        total_tokens = compute_tokens(model_name, few_shots, issue)

        if total_tokens <= context_limit:
            return selected

        if overflow_ptr >= len(overflow):
            print(
                f"[WARNING] Infeasible: repo={repo}, model={model_name}, "
                f"test_idx={test_idx}, k={k}",
                flush=True
            )
            return None

        # remove longest example instead of last
        longest_idx = max(
            range(len(selected)),
            key=lambda i: example_length(selected[i], train_data)
        )

        selected.pop(longest_idx)
        selected.append(overflow[overflow_ptr])
        overflow_ptr += 1

# ============================================================
# MAIN LOOP
# ============================================================

for repo in REPOS:

    print(f"\n==============================")
    print(f"[REPO] {repo}")
    print(f"==============================")

    train_data, test_data = load_data(repo)

    train_emb = embed(repo, train_data, "train")
    test_emb = embed(repo, test_data, "test")

    for model_name, context_limit in MODELS:

        print(f"\n[MODEL] {model_name}")

        output_dir = os.path.join(
            BASE_OUTPUT_DIR,
            repo,
            EMBED_MODEL_NAME,
            model_name.replace("/", "_")
        )
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "selections.json")

        results = {
            "metadata": {
                "repo": repo,
                "model": model_name,
                "context_limit": context_limit,
                "embedding_model": EMBED_MODEL_NAME,
                "created_at": str(datetime.now())
            },
            "selections": {}
        }

        for k in FEW_SHOT_COUNTS:

            print(f"[INFO] k={k}")

            results["selections"][str(k)] = {}

            for test_idx, issue in enumerate(test_data):

                sims = cosine_similarity([test_emb[test_idx]], train_emb)[0]
                sorted_idx = np.argsort(sims)[::-1]

                selected = build_prompt_exact_k(
                    sorted_idx,
                    k,
                    issue,
                    train_data,
                    model_name,
                    context_limit,
                    test_idx,
                    repo
                )

                if selected is None:
                    results["selections"][str(k)][str(test_idx)] = {
                        "indices": None,
                        "status": "infeasible"
                    }
                else:
                    results["selections"][str(k)][str(test_idx)] = {
                        "indices": selected,
                        "status": "ok"
                    }

                if test_idx % 50 == 0:
                    print(f"k={k} | test={test_idx}", flush=True)

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved - {output_file}")
