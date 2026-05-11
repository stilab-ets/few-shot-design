import os
import json
import csv
from datetime import datetime, timezone
import uuid
from sklearn.metrics import accuracy_score
from vllm_llm_logit_jinja import predict, count_input_tokens


# ============================================================
# CONFIGURATION
# ============================================================

REPO = "facebook_react"
#REPO = "bitcoin_bitcoin"
#REPO = "opencv_opencv"
#REPO = "tensorflow_tensorflow"
#REPO = "microsoft_vscode"

#MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
#MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "google/gemma-3-1b-it"
MODEL_NAME = "google/gemma-3-12b-it"

EXP_NUM = "zero_shot"

RUNS = 1
VALID_LABELS = {"bug", "feature", "question"}
LOG_DIR = f"../rqs/{EXP_NUM}/{REPO}/{MODEL_NAME}/logs"
LOG_FILE = os.path.join(LOG_DIR, "zeroshot_results.csv")

os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_data(repo_name):
    base_path = f"../data/{repo_name}"
    with open(os.path.join(base_path, f"{repo_name}_test_issues_normalized.json"), encoding="utf-8") as f:
        test = json.load(f)
    return test


test_data = load_data(REPO)

def log_predictions(filepath,
                    individual,
                    title,
                    body,
                    ground_truth,
                    prediction, bug_logprobs, feature_logprobs, question_logprobs):

    file_exists = os.path.isfile(filepath)

    # ISO 8601 UTC timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

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

# ============================================================
# ZERO-SHOT PROMPT
# ============================================================

def generate_zero_shot_prompt():

    prompt = (
        "Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting.\n"
    )

    return prompt


# ============================================================
# EVALUATION
# ============================================================

def evaluate_zero_shot(prompt, test_data, run_id):

    predictions = []
    true_labels = []

    print(f"\n=== ZERO-SHOT RUN {run_id} ===", flush=True)

    system_tokens = count_input_tokens(MODEL_NAME, prompt)

    print(f"[ZERO-SHOT | Run {run_id}] System prompt tokens: {system_tokens}", flush=True)

    # Create generation directory
    gen_dir = os.path.join(LOG_DIR, f"generations/zero_shot/run_{run_id}")
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
                prompt,
                few_shots=None,
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
            None,
            issue["title"],
            issue["body"],
            true_label,
            predicted_label
        )

    if len(predictions) == 0:
        return None, system_tokens

    accuracy = accuracy_score(true_labels, predictions)

    print(f"[ZERO-SHOT] Accuracy: {accuracy:.4f}", flush=True)

    return accuracy, system_tokens


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for run_id in range(RUNS):
        prompt = generate_zero_shot_prompt()

        acc, token_count = evaluate_zero_shot(
            prompt,
            test_data,
            run_id
        )

        print(f"Run {run_id} - Accuracy: {acc}", flush=True)
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(
                f,
                quoting=csv.QUOTE_MINIMAL
            )
            # Write header only if file is new
            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "FewShot_Count",
                    "Individual",
                    "Accuracy",
                    "Token_Count",
                ])
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                0,                 # FewShot_Count = 0
                f"ZERO_SHOT_{run_id}",  # Individual (unique per run)
                acc,
                token_count,
            ])

        print(
            f"[DONE] Zero-shot run {run_id} | accuracy={acc:.4f}, tokens={token_count}",
            flush=True
        )
