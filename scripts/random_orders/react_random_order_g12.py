import os
import json
import random
import csv
from datetime import datetime
import ast
import math
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
import uuid
import pandas as pd
from vllm_llm_logit_jinja import predict, count_input_tokens

# =========================================================
# CONFIG
# =========================================================

REPO = "facebook_react"
# REPO = "bitcoin_bitcoin"
# REPO = "opencv_opencv"
# REPO = "tensorflow_tensorflow"
# REPO = "microsoft_vscode"

MODEL_NAME = "google/gemma-3-12b-it"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME = "google/gemma-3-1b-it"

MIN_K = 2
MAX_K = 32
SHUFFLES_PER_SET = 30
MAX_WORKERS = 20

VALID_LABELS = {"bug", "feature", "question"}
EXP_NUM = "random_order_strat"

ORDER_BASE_FILE = f"../rqs/{EXP_NUM}/order_base_sets.json"

LOG_DIR = f"../rqs/{EXP_NUM}/{REPO}/models/{MODEL_NAME}/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "order_permutation_results.csv")

print("Order permutation experiment started", flush=True)
print("Model:", MODEL_NAME, flush=True)
print("Log file:", LOG_FILE, flush=True)

# =========================================================
# LOAD DATA
# =========================================================

def load_data(repo_name):
    base_path = f"../data/{repo_name}"
    with open(os.path.join(base_path, f"{repo_name}_train_issues_normalized.json"), encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(base_path, f"{repo_name}_test_issues_normalized.json"), encoding="utf-8") as f:
        test_data = json.load(f)
    return train_data, test_data

train_data, test_data = load_data(REPO)

# =========================================================
# PROMPT + EVALUATION
# =========================================================

def generate_few_shots(selected_indices):

    few_shots = []

    for idx in selected_indices:

        issue = train_data[idx]

        few_shots.append({
            "input": f"Title: {issue['title']}\nBody: {issue['body']}",
            "output": issue["labels"].strip().lower()
        })

    return few_shots

def log_predictions(filepath,
                    individual,
                    title,
                    body,
                    ground_truth,
                    prediction):

    file_exists = os.path.isfile(filepath)

    # ISO 8601 UTC timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # IMPORTANT: newline="" prevents blank lines on Windows
    with open(filepath, "a", encoding="utf-8", newline="") as f:

        writer = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL  # auto-escape when needed
        )

        # Write header only if file is new
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Individual",
                "Title",
                "Body",
                "Ground_Truth",
                "Prediction"
            ])

        writer.writerow([
            timestamp,
            individual,
            title,
            body,
            ground_truth,
            prediction
        ])

def log_result(row):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Repo",
                "FewShot_Count",
                "Individual",
                "Accuracy",
                "Token_Count",
            ])
        writer.writerow(row)

def evaluate_prompt(system_prompt, test_data, gen_num, individual, few_shots):

    predictions = []
    true_labels = []

    print("evaluation of prompt now", flush=True)

    system_tokens = count_input_tokens(MODEL_NAME, system_prompt)

    # count few-shot tokens too
    for fs in few_shots:
        system_tokens += count_input_tokens(MODEL_NAME, fs["input"])
        system_tokens += count_input_tokens(MODEL_NAME, fs["output"])

    print(f"[{gen_num}] System prompt tokens: {system_tokens}", flush=True)

    gen_dir = os.path.join(LOG_DIR, f"generations/{gen_num}")
    os.makedirs(gen_dir, exist_ok=True)

    pred_file = os.path.join(
        gen_dir,
        f"predictions_{uuid.uuid4()}.csv"
    )

    for issue in test_data:

        user_prompt = f"Title: {issue['title']}\nBody: {issue['body']}"

        try:
            predicted_label = predict(
                MODEL_NAME,
                user_prompt,
                system_prompt=system_prompt,
                few_shots=few_shots,
                few_shots_style="chat"
            )
        except Exception as e:
            print(f"Inference error: {e}", flush=True)
            continue

        true_label = issue["labels"].strip().lower()

        predictions.append(predicted_label)
        true_labels.append(true_label)

        log_predictions(
            pred_file,
            individual,
            issue["title"],
            issue["body"],
            true_label,
            predicted_label
        )

    if len(predictions) == 0:
        return None, system_tokens

    accuracy = accuracy_score(true_labels, predictions)

    print(f"[{gen_num}] Accuracy: {accuracy:.4f}", flush=True)

    return accuracy, system_tokens
# =========================================================
# LOAD BASE SETS
# =========================================================

with open(ORDER_BASE_FILE, "r", encoding="utf-8") as f:
    base_data = json.load(f)

repo_bases = {
    int(k): v
    for k, v in base_data[REPO].items()
    if MIN_K <= int(k) <= MAX_K
}

print(f"Loaded {len(repo_bases)} base sets for ordering", flush=True)

# =========================================================
# LOG FILE HEADER
# =========================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Repo",
            "FewShot_Count",
            "Original_Individual",
            "Original_Accuracy",
            "Original_Prompt_Tokens",
            "Shuffled_Individual",
            "Shuffled_Accuracy",
            "Shuffled_Prompt_Tokens",
        ])

# =========================================================
# LOAD ORIGINAL RANDOM RESULTS
# =========================================================

RANDOM_RESULTS_FILE = f"../rqs/stratified_selection/{REPO}/models/{MODEL_NAME}/logs/fewshot_results.csv"

random_df = pd.read_csv(RANDOM_RESULTS_FILE)
random_df["Individual"] = random_df["Individual"].apply(
    lambda x: tuple(ast.literal_eval(x))
)

# =========================================================
# SHUFFLE EVALUATION
# =========================================================

def evaluate_shuffles(k, base_indices):
    SYSTEM_INSTRUCTIONS = "Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting."
    print(f"Starting k={k}", flush=True)

    rng = random.Random(42 + k)

    base_indices = list(dict.fromkeys(base_indices))

    original_row = random_df[
        (random_df["FewShot_Count"] == k) &
        (random_df["Individual"] == tuple(base_indices))
    ]

    if original_row.empty:
        print(f"[k={k}] Could not find original accuracy in random_selection results.", flush=True)
        return

    original_accuracy = float(original_row.iloc[0]["Accuracy"])

    system_prompt = SYSTEM_INSTRUCTIONS
    original_few_shots = generate_few_shots(base_indices)

    original_prompt_tokens = count_input_tokens(MODEL_NAME, system_prompt)
    for fs in original_few_shots:
        original_prompt_tokens += count_input_tokens(MODEL_NAME, fs["input"])
        original_prompt_tokens += count_input_tokens(MODEL_NAME, fs["output"])

    seen = {tuple(base_indices)}

    max_shuffles = min(
        SHUFFLES_PER_SET,
        math.factorial(len(base_indices)) - 1
    )

    if max_shuffles <= 0:
        return

    count = 0

    while count < max_shuffles:

        shuffled = base_indices[:]
        rng.shuffle(shuffled)

        key = tuple(shuffled)
        if key in seen:
            continue

        seen.add(key)
        count += 1

        few_shots = generate_few_shots(shuffled)

        shuffled_accuracy, shuffled_token_count = evaluate_prompt(
            SYSTEM_INSTRUCTIONS,
            test_data,
            gen_num=f"fs_{k}_shuffle_{count}",
            individual=shuffled,
            few_shots=few_shots
        )

        if shuffled_accuracy is None:
            continue

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                REPO,
                k,
                base_indices,
                original_accuracy,
                original_prompt_tokens,
                shuffled,
                shuffled_accuracy,
                shuffled_token_count,
            ])

    print(f"Finished k={k}", flush=True)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            k: executor.submit(evaluate_shuffles, k, base_indices)
            for k, base_indices in repo_bases.items()
        }

        for k, future in futures.items():
            try:
                future.result()
            except Exception as e:
                print(f"[k={k}] Thread crashed: {e}", flush=True)
                raise

    print("All ordering experiments completed.", flush=True)
