import os
import json
import random
import csv
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import accuracy_score
from vllm_llm_logit_jinja import predict, count_input_tokens, count_chat_style_tokens

# ============================================================
# CONFIG
# ============================================================

REPO = "facebook_react"
# REPO = "bitcoin_bitcoin"
# REPO = "opencv_opencv"
# REPO = "tensorflow_tensorflow"
#REPO = "microsoft_vscode"

MODEL_NAME = "google/gemma-3-12b-it"
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

FEW_SHOT_COUNTS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
NUM_TRIALS_PER_COUNT = 20
MAX_WORKERS = 20

BUFFER = 4
#MODEL_CONTEXT_LIMIT = 131072
MODEL_CONTEXT_LIMIT = 32768
OUTPUT_TOKEN_BUFFER = 1

EXP_NUM = "random_selection"

LOG_DIR = f"../rqs/{EXP_NUM}/{REPO}/models/{MODEL_NAME}/logs"
LOG_FILE = os.path.join(LOG_DIR, "fewshot_results.csv")
PROMPT_SET_FILE = f"../rqs/{EXP_NUM}/{REPO}/shared_prompt_sets.json"

os.makedirs(LOG_DIR, exist_ok=True)

VALID_LABELS = ["bug", "feature", "question"]

# ============================================================
# DATA LOADING
# ============================================================

def load_data(repo_name):
    base_path = f"../data/{repo_name}"
    with open(os.path.join(base_path, f"{repo_name}_train_issues_normalized.json"), encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(base_path, f"{repo_name}_test_issues_normalized.json"), encoding="utf-8") as f:
        test_data = json.load(f)
    return train_data, test_data


train_data, test_data = load_data(REPO)
POOL_SIZE = len(train_data)

# ============================================================
# BUILD LABEL POOLS
# ============================================================

LABEL_TO_INDICES = {label: [] for label in VALID_LABELS}

for idx, issue in enumerate(train_data):
    label = issue["labels"].strip().lower()
    if label in LABEL_TO_INDICES:
        LABEL_TO_INDICES[label].append(idx)

print("Class pool sizes:")
for label in VALID_LABELS:
    print(f"{label}: {len(LABEL_TO_INDICES[label])}")

TOTAL_TRAIN = sum(len(v) for v in LABEL_TO_INDICES.values())
LABEL_PROPORTIONS = {
    label: len(LABEL_TO_INDICES[label]) / TOTAL_TRAIN
    for label in VALID_LABELS
}

# ============================================================
# CONTEXT CHECK
# ============================================================

print("Computing max test-instance token count...", flush=True)

MAX_TEST_TOKENS = max(
    count_input_tokens(
        MODEL_NAME,
        f"Title: {issue['title']}\nBody: {issue['body']}"
    )
    for issue in test_data
)

print(f"Max test-instance tokens: {MAX_TEST_TOKENS}", flush=True)

# ============================================================
# PROMPT GENERATION
# ============================================================
def generate_prompt(selected_indices):
    examples = "\n\n".join(
        f"Title: {train_data[idx]['title']}\n"
        f"Body: {train_data[idx]['body']}\n"
        f"Label: {train_data[idx]['labels']}"
        for idx in selected_indices
    )

    prompt = (
        f"Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting. Here's some examples:\n\n"
        f"{examples}"
    )

    return prompt

def generate_few_shots(selected_indices, train_data):
    """
    Converts few-shot indices into chat-style examples.
    """
    few_shots = []

    for idx in selected_indices:
        issue = train_data[idx]

        few_shots.append({
            "input": f"Title: {issue['title']}\nBody: {issue['body']}",
            "output": issue["labels"].strip().lower()
        })

    return few_shots

def is_prompt_within_context(prompt):
    system_tokens = count_input_tokens(MODEL_NAME, prompt)
    total_tokens = system_tokens + MAX_TEST_TOKENS + OUTPUT_TOKEN_BUFFER + BUFFER
    return total_tokens <= MODEL_CONTEXT_LIMIT

# ============================================================
# LOGGING PREDICTIONS
# ============================================================

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

# ============================================================
# PROMPT SET GENERATION
# ============================================================

def generate_and_save_prompt_sets():
    random.seed(42)
    prompt_sets = {}

    for k in FEW_SHOT_COUNTS:
        seen = set()
        valid_sets = []

        # -------------------------
        # k == 1
        # -------------------------
        if k == 1:
            for label in VALID_LABELS:
                idx = random.choice(LABEL_TO_INDICES[label])
                valid_sets.append([idx])
                seen.add((idx,))

            while len(valid_sets) < NUM_TRIALS_PER_COUNT:
                label = random.choice(VALID_LABELS)
                idx = random.choice(LABEL_TO_INDICES[label])
                key = (idx,)
                if key in seen:
                    continue
                valid_sets.append([idx])
                seen.add(key)

        # -------------------------
        # k == 2
        # -------------------------
        elif k == 2:

            distinct_pairs = [
                ("bug", "feature"),
                ("bug", "question"),
                ("feature", "question"),
            ]

            for l1, l2 in distinct_pairs:
                idx1 = random.choice(LABEL_TO_INDICES[l1])
                idx2 = random.choice(LABEL_TO_INDICES[l2])
                pair = tuple(sorted([idx1, idx2]))
                if pair not in seen:
                    valid_sets.append(list(pair))
                    seen.add(pair)

            for label in VALID_LABELS:
                if len(LABEL_TO_INDICES[label]) >= 2:
                    pair = tuple(sorted(random.sample(LABEL_TO_INDICES[label], 2)))
                    if pair not in seen:
                        valid_sets.append(list(pair))
                        seen.add(pair)

            while len(valid_sets) < NUM_TRIALS_PER_COUNT:
                pair = tuple(sorted(random.sample(range(POOL_SIZE), 2)))
                if pair in seen:
                    continue
                valid_sets.append(list(pair))
                seen.add(pair)

        # -------------------------
        # k >= 3
        # -------------------------
        else:
            while len(valid_sets) < NUM_TRIALS_PER_COUNT:

                individual = []

                # Step 1: guarantee 1 per class
                base_counts = {label: 1 for label in VALID_LABELS}
                remaining = k - len(VALID_LABELS)

                # Step 2: proportional allocation on remaining
                raw_counts = {
                    label: remaining * LABEL_PROPORTIONS[label]
                    for label in VALID_LABELS
                }

                floor_counts = {
                    label: int(raw_counts[label])
                    for label in VALID_LABELS
                }

                allocated = sum(floor_counts.values())
                remainder = remaining - allocated

                fractional_parts = sorted(
                    VALID_LABELS,
                    key=lambda l: raw_counts[l] - floor_counts[l],
                    reverse=True
                )

                for i in range(remainder):
                    floor_counts[fractional_parts[i]] += 1

                # Final counts
                final_counts = {
                    label: base_counts[label] + floor_counts[label]
                    for label in VALID_LABELS
                }

                # Sample indices
                for label in VALID_LABELS:
                    count = final_counts[label]
                    if count > 0:
                        individual.extend(
                            random.sample(LABEL_TO_INDICES[label], count)
                        )

                individual = sorted(individual)
                key = tuple(individual)

                if key in seen:
                    continue

                prompt = generate_prompt(individual)
                if not is_prompt_within_context(prompt):
                    continue

                valid_sets.append(individual)
                seen.add(key)

        prompt_sets[str(k)] = valid_sets

    os.makedirs(os.path.dirname(PROMPT_SET_FILE), exist_ok=True)

    with open(PROMPT_SET_FILE, "w", encoding="utf-8") as f:
        json.dump(prompt_sets, f, indent=2)

    print("Saved coverage-constrained proportional prompt sets.")

if not os.path.exists(PROMPT_SET_FILE):
    generate_and_save_prompt_sets()

with open(PROMPT_SET_FILE, "r", encoding="utf-8") as f:
    SHARED_PROMPT_SETS = json.load(f)

# ============================================================
# EVALUATION
# ============================================================

def evaluate_prompt(system_prompt, test_data, gen_num, individual, few_shots):

    predictions = []
    true_labels = []

    # chat-style token count
    system_prompt_token_count = count_chat_style_tokens(
        MODEL_NAME,
        system_prompt,
        few_shots
    )

    print(
        f"[{gen_num}] System + few-shot tokens: {system_prompt_token_count}",
        flush=True
    )

    pred_dir = os.path.join(LOG_DIR, f"generations/gen_{gen_num}")
    os.makedirs(pred_dir, exist_ok=True)
    pred_file = os.path.join(pred_dir, f"predictions_{uuid.uuid4()}.csv")

    for issue in test_data:

        user_prompt = f"Title: {issue['title']}\nBody: {issue['body']}\n"

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

    accuracy = accuracy_score(true_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}", flush=True)

    return accuracy, system_prompt_token_count

def run_fewshot_trials(num_few_shots):

    individuals = SHARED_PROMPT_SETS[str(num_few_shots)]

    system_instructions = "Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting."

    for trial, individual in enumerate(individuals):

        # build chat-style few shots
        few_shots = generate_few_shots(individual, train_data)

        accuracy, token_count = evaluate_prompt(
            system_instructions,
            test_data,
            gen_num=f"fs_{num_few_shots}_trial_{trial}",
            individual=individual,
            few_shots=few_shots
        )

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                REPO,
                num_few_shots,
                individual,
                accuracy,
                token_count
            ])

# ============================================================
# MAIN
# ============================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Timestamp", "Repo", "FewShot_Count", "Individual", "Accuracy", "Token_Count"]
        )

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run_fewshot_trials, fs)
            for fs in FEW_SHOT_COUNTS
        ]
        for f in futures:
            f.result()
