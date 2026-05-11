import os
import json
import csv
from datetime import datetime
import uuid
import itertools
import ast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
from vllm_llm_logit_jinja import predict, count_input_tokens

# =========================================================
# CONFIG
# =========================================================

REPO = "facebook_react"
#REPO = "bitcoin_bitcoin"
#REPO = "opencv_opencv"
#REPO = "tensorflow_tensorflow"
#REPO = "microsoft_vscode"

#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "google/gemma-3-12b-it"

MAX_WORKERS = 20

OUTPUT_EXP = "targeted_orders"

LOG_DIR = f"../rqs/{OUTPUT_EXP}/{REPO}/models/{MODEL_NAME}/logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "targeted_order_results.csv")

ORDER_BASE_FILE = f"../rqs/random_orders/order_base_sets.json"

ORIGINAL_RESULTS_FILE = (
    f"../rqs/stratified_selection/{REPO}/models/"
    f"{MODEL_NAME}/logs/fewshot_results.csv"
)

print("Targeted order strategy experiment started", flush=True)
print("Model:", MODEL_NAME, flush=True)
print("Log file:", LOG_FILE, flush=True)

# =========================================================
# LOAD DATA
# =========================================================

def load_data(repo):
    base = f"../data/{repo}"
    with open(os.path.join(base, f"{repo}_train_issues_normalized.json"), encoding="utf-8") as f:
        train = json.load(f)
    with open(os.path.join(base, f"{repo}_test_issues_normalized.json"), encoding="utf-8") as f:
        test = json.load(f)
    return train, test

train_data, test_data = load_data(REPO)

# =========================================================
# LOAD ORIGINAL STRATIFIED RESULTS
# =========================================================

if not os.path.exists(ORIGINAL_RESULTS_FILE):
    raise FileNotFoundError(
        f"Could not find original results file: {ORIGINAL_RESULTS_FILE}"
    )

original_df = pd.read_csv(ORIGINAL_RESULTS_FILE)

original_df["Individual"] = original_df["Individual"].apply(
    lambda x: tuple(ast.literal_eval(x))
)

# =========================================================
# PROMPT GENERATION
# =========================================================

def generate_few_shots(selected_indices):

    few_shots = []

    for idx in selected_indices:

        issue = train_data[idx]

        few_shots.append({
            "input": f"Title: {issue['title']}\nBody: {issue['body']}",
            "output": issue["labels"]
        })

    return few_shots

# =========================================================
# EVALUATE PROMPT
# =========================================================

def evaluate_prompt(test_data, gen_num, individual):

    predictions = []
    true_labels = []

    system_prompt = (
        "Your job is to analyze and assign the most fitting labels to GitHub issues. "
        "Identify and assign the most accurate label for the issue using its title and body. "
        "Select the label exclusively from the options given in this list: "
        "\"question\", \"feature\", \"bug\". Ensure that the label is an exact match "
        "to one from the list. Output the label as a single string format with no "
        "additional characters, text, or formatting."
    )

    # generate few shots from indices
    few_shots = generate_few_shots(individual)

    system_tokens = count_input_tokens(MODEL_NAME, system_prompt)

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
# LOG PREDICTIONS
# =========================================================

def log_predictions(filepath,
                    individual,
                    title,
                    body,
                    ground_truth,
                    prediction):

    file_exists = os.path.isfile(filepath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL
        )

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

# =========================================================
# ORDER STRATEGIES
# =========================================================

def label_of(i):
    return train_data[i]["labels"].lower()

def length_of(i):
    return len(train_data[i]["title"]) + len(train_data[i]["body"])

def grouped_dynamic(indices, order):
    buckets = {}
    for i in indices:
        buckets.setdefault(label_of(i), []).append(i)

    out = []
    for l in order:
        if l in buckets:
            out.extend(buckets[l])
    return out

def short_to_long(indices):
    return sorted(indices, key=length_of)

def long_to_short(indices):
    return sorted(indices, key=length_of, reverse=True)

# =========================================================
# LOAD BASE SETS
# =========================================================

with open(ORDER_BASE_FILE, "r", encoding="utf-8") as f:
    base_data = json.load(f)

repo_bases = {int(k): v for k, v in base_data[REPO].items()}

# =========================================================
# LOG HEADER
# =========================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Repo",
            "FewShot_Count",
            "Order_Strategy",
            "Original_Individual",
            "Original_Accuracy",
            "Original_Tokens",
            "New_Individual",
            "New_Accuracy",
            "New_Tokens",
        ])

# =========================================================
# MAIN
# =========================================================

def evaluate_k(k, base):

    base = list(dict.fromkeys(base))

    original_row = original_df[
        (original_df["FewShot_Count"] == k) &
        (original_df["Individual"] == tuple(base))
    ]

    if original_row.empty:
        print(f"[k={k}] Could not find original accuracy.", flush=True)
        return

    original_accuracy = float(original_row.iloc[0]["Accuracy"])
    original_tokens = int(original_row.iloc[0]["Token_Count"])

    labels_present = sorted(set(label_of(i) for i in base))

    strategies = {}

    if len(labels_present) >= 2:
        for perm in itertools.permutations(labels_present):
            name = "group_" + "_".join(perm)
            strategies[name] = lambda x, p=tuple(perm): grouped_dynamic(x, p)

    strategies["short_to_long"] = short_to_long
    strategies["long_to_short"] = long_to_short

    for name, fn in strategies.items():

        ordered = fn(base)

        # Safety check: ensure we didn't change the example set
        if set(ordered) != set(base):
            raise ValueError("Ordering strategy changed the example set!")

        new_acc, new_tokens = evaluate_prompt(
            test_data,
            gen_num=f"k_{k}_{name}",
            individual=ordered
        )

        if new_acc is None:
            continue

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                REPO,
                k,
                name,
                base,
                original_accuracy,
                original_tokens,
                ordered,
                new_acc,
                new_tokens,
            ])

if __name__ == "__main__":

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            k: executor.submit(evaluate_k, k, base)
            for k, base in repo_bases.items()
        }

        for future in futures.values():
            future.result()

    print("Targeted order strategy experiment completed.", flush=True)
