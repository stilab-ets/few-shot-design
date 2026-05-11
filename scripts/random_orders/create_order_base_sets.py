import os
import json
import pandas as pd
import ast

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

REPOS = [
    "facebook_react",
    "bitcoin_bitcoin",
    "opencv_opencv",
    "tensorflow_tensorflow",
    "microsoft_vscode"
]

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3"
]

EXP_NUM = "random_selection"

OUTPUT_FILE = "../../rqs/random_orders/order_base_sets.json"

# -------------------------------------------------
# MAIN
# -------------------------------------------------

order_base_sets = {}

for repo in REPOS:

    print(f"Processing repo: {repo}")

    all_model_dfs = []

    # Load each model's results
    for model in MODELS:
        path = f"../../rqs/{EXP_NUM}/{repo}/models/{model}/logs/fewshot_results.csv"

        if not os.path.exists(path):
            raise RuntimeError(f"Missing results for {repo} - {model}")

        df = pd.read_csv(path)

        # Keep only necessary columns
        df = df[["FewShot_Count", "Individual", "Accuracy"]]

        # Convert Individual string to tuple for grouping
        df["Individual"] = df["Individual"].apply(
            lambda x: tuple(ast.literal_eval(x))
        )

        df["Model"] = model
        all_model_dfs.append(df)

    # Concatenate all models
    combined = pd.concat(all_model_dfs)

    # Compute mean accuracy across models per (k, individual)
    grouped = (
        combined
        .groupby(["FewShot_Count", "Individual"])["Accuracy"]
        .mean()
        .reset_index()
    )

    repo_bases = {}

    for k, group in grouped.groupby("FewShot_Count"):

        # Sort by mean accuracy
        group = group.sort_values("Accuracy")

        # Select median-performing individual
        median_index = len(group) // 2
        selected_row = group.iloc[median_index]

        selected_individual = list(selected_row["Individual"])

        repo_bases[str(k)] = selected_individual

        print(f"  k={k} - selected median base")

    order_base_sets[repo] = repo_bases

# Save JSON
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(order_base_sets, f, indent=2)

print("\nSaved order_base_sets.json successfully.")
