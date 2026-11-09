import os
import json
import csv
import uuid
from datetime import datetime
from sklearn.metrics import f1_score

from vllm_llm_logit_jinja import predict, count_chat_style_tokens

# ============================================================
# CONFIG (SET PER RUN)
# ============================================================

REPO = "facebook_react"
#REPO = "bitcoin_bitcoin"
#REPO = "opencv_opencv"
#REPO = "tensorflow_tensorflow"
#REPO = "microsoft_vscode"

#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "google/gemma-3-12b-it"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

EMBED_MODEL_NAME = "all-mpnet-base-v2"
EXP_NUM = "diversity_selection"

FEW_SHOT_COUNTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]

SELECTION_FILE = os.path.join(
    f"../rqs/{EXP_NUM}", REPO, EMBED_MODEL_NAME, "selections.json"
)

LOG_DIR = f"../rqs/{EXP_NUM}/{REPO}/{EMBED_MODEL_NAME}/models/{MODEL_NAME}/logs"
LOG_FILE = os.path.join(LOG_DIR, "cluster_diversity_fixed_results.csv")
os.makedirs(LOG_DIR, exist_ok=True)

SYSTEM_PROMPT = "Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting."

VALID_LABELS = ["bug", "feature", "question"]

# ============================================================
# LOAD DATA + SELECTIONS
# ============================================================

def load_data(repo_name):
    base_path = f"../data/{repo_name}"
    with open(os.path.join(base_path, f"{repo_name}_train_issues_normalized.json"), encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(base_path, f"{repo_name}_test_issues_normalized.json"), encoding="utf-8") as f:
        test = json.load(f)
    return train, test


train_data, test_data = load_data(REPO)

with open(SELECTION_FILE, "r") as f:
    selections = json.load(f)["selections"]

# ============================================================
# LOGGING
# ============================================================

def log_predictions(filepath, individual, title, body, ground_truth, prediction):
    file_exists = os.path.isfile(filepath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writerow(["Timestamp", "Individual", "Title", "Body", "Ground_Truth", "Prediction"])
        writer.writerow([timestamp, individual, title, body, ground_truth, prediction])


def log_result(row):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Repo", "FewShot_Count", "Macro-F1", "Token_Count",
            ])
        writer.writerow(row)

# ============================================================
# BUILD FEW-SHOTS
# ============================================================

def build_few_shots(indices):
    return [
        {
            "input": f"Title: {train_data[i]['title']}\nBody: {train_data[i]['body']}",
            "output": train_data[i]["labels"].strip().lower(),
        }
        for i in indices
    ]

# ============================================================
# EVALUATION
# ============================================================

def evaluate_prompt(k, indices):

    few_shots = build_few_shots(indices)
    system_tokens = count_chat_style_tokens(MODEL_NAME, SYSTEM_PROMPT, few_shots)
    print(f"[k={k}] System prompt tokens ({MODEL_NAME}): {system_tokens}", flush=True)

    gen_dir = os.path.join(LOG_DIR, f"generations/cluster_{k}")
    os.makedirs(gen_dir, exist_ok=True)
    pred_file = os.path.join(gen_dir, f"predictions_{uuid.uuid4()}.csv")

    predictions, true_labels = [], []

    for test_idx, issue in enumerate(test_data):
        user_prompt = f"Title: {issue['title']}\nBody: {issue['body']}"

        try:
            predicted_label = predict(
                MODEL_NAME,
                user_prompt,
                system_prompt=SYSTEM_PROMPT,
                few_shots=few_shots,
                few_shots_style="chat",
            )
        except Exception as e:
            print(f"Inference error: {e}", flush=True)
            continue

        true_label = issue["labels"].strip().lower()
        predictions.append(predicted_label)
        true_labels.append(true_label)

        log_predictions(pred_file, indices, issue["title"], issue["body"], true_label, predicted_label)

        if test_idx % 50 == 0:
            print(f"k={k} | test={test_idx}", flush=True)

    if len(predictions) == 0:
        return None, system_tokens

    macro_f1 = f1_score(true_labels, predictions, labels=VALID_LABELS, average="macro", zero_division=0)
    print(f"[k={k}] Macro-F1: {macro_f1:.4f}", flush=True)
    return macro_f1, system_tokens

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for k in FEW_SHOT_COUNTS:

        entry = selections[str(k)]

        if entry["status"] == "infeasible":
            print(f"[WARNING] k={k} infeasible for this repo (all models), skipping", flush=True)
            continue

        print(f"\n[INFO] Evaluating k={k}", flush=True)
        macro_f1, token_count = evaluate_prompt(k, entry["indices"])

        if macro_f1 is None:
            print(f"[WARNING] No valid predictions for k={k}")
            continue

        log_result([datetime.now(), REPO, k, macro_f1, token_count])

    print("Evaluation completed.", flush=True)