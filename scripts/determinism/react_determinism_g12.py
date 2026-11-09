import os
import json
import csv
import ast
import threading
from datetime import datetime
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from sklearn.metrics import f1_score

from vllm_llm_logit_jinja import (
    predict,
    count_input_tokens,
    count_chat_style_tokens,
)


# ============================================================
# CONFIG
# ============================================================

REPO = "facebook_react"
# REPO = "bitcoin_bitcoin"
# REPO = "opencv_opencv"
# REPO = "tensorflow_tensorflow"
# REPO = "microsoft_vscode"


MODEL_NAME = "google/gemma-3-12b-it"
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


FEW_SHOT_COUNTS = [
    1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32
]

NUM_REPEATS = 20
MAX_WORKERS = 15


BUFFER = 4

# MODEL_CONTEXT_LIMIT = 131072
MODEL_CONTEXT_LIMIT = 32768

OUTPUT_TOKEN_BUFFER = 1


# ------------------------------------------------------------
# Experiment paths
# ------------------------------------------------------------

# Original experiment from which the selected prompts came.
SOURCE_EXP_NUM = "stratified_selection"

# New experiment name.
EXP_NUM = "determinism"

SELECTED_PROMPTS_FILE = (
    f"../rqs/{SOURCE_EXP_NUM}/median_prompts.csv"
)


# ------------------------------------------------------------
# Determinism experiment output
# ------------------------------------------------------------

LOG_DIR = (
    f"../rqs/{EXP_NUM}/{REPO}/"
    f"models/{MODEL_NAME}/logs"
)

LOG_FILE = os.path.join(
    LOG_DIR,
    "fewshot_results_macro_f1.csv",
)


# ------------------------------------------------------------
# Original shared prompt-set file
#
# This is used to verify that:
#
# FewShot_Count + Trial
#
# correspond exactly to the Individual stored in the CSV.
# ------------------------------------------------------------

PROMPT_SET_FILE = (
    f"../rqs/{SOURCE_EXP_NUM}/{REPO}/"
    f"shared_prompt_sets.json"
)


os.makedirs(
    LOG_DIR,
    exist_ok=True,
)


VALID_LABELS = [
    "bug",
    "feature",
    "question",
]


# Because different workers can finish at the same time,
# protect writes to the common summary CSV.
LOG_LOCK = threading.Lock()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(repo_name):

    base_path = f"../data/{repo_name}"

    with open(
        os.path.join(
            base_path,
            f"{repo_name}_train_issues_normalized.json",
        ),
        encoding="utf-8",
    ) as f:

        train_data = json.load(f)


    with open(
        os.path.join(
            base_path,
            f"{repo_name}_test_issues_normalized.json",
        ),
        encoding="utf-8",
    ) as f:

        test_data = json.load(f)


    return train_data, test_data


train_data, test_data = load_data(REPO)

POOL_SIZE = len(train_data)


# ============================================================
# BUILD LABEL POOLS
# ============================================================

# Retained from the original script so that the repository
# label distribution is still printed when the experiment
# starts.

LABEL_TO_INDICES = {
    label: []
    for label in VALID_LABELS
}


for idx, issue in enumerate(train_data):

    label = (
        issue["labels"]
        .strip()
        .lower()
    )

    if label in LABEL_TO_INDICES:
        LABEL_TO_INDICES[label].append(idx)


print("Class pool sizes:")

for label in VALID_LABELS:

    print(
        f"{label}: "
        f"{len(LABEL_TO_INDICES[label])}"
    )


# ============================================================
# CONTEXT INFORMATION
# ============================================================

# We retain the maximum test-instance token count calculation
# from the original setup.
#
# Prompt construction itself is now exclusively chat-style:
#
#   system instructions
#   +
#   structured few_shots
#   +
#   test issue
#
# No string-based few-shot prompt is generated.

print(
    "Computing max test-instance token count...",
    flush=True,
)


MAX_TEST_TOKENS = max(

    count_input_tokens(
        MODEL_NAME,
        (
            f"Title: {issue['title']}\n"
            f"Body: {issue['body']}"
        ),
    )

    for issue in test_data
)


print(
    f"Max test-instance tokens: "
    f"{MAX_TEST_TOKENS}",
    flush=True,
)


# ============================================================
# CHAT-STYLE FEW-SHOT GENERATION
# ============================================================

def generate_few_shots(
    selected_indices,
    train_data,
):
    """
    Converts the selected training-example indices into
    chat-style few-shot examples.

    Example:

    [
        {
            "input": "Title: ...\\nBody: ...",
            "output": "bug"
        },
        {
            "input": "Title: ...\\nBody: ...",
            "output": "question"
        }
    ]

    These are passed directly to predict() using:

        few_shots=few_shots
        few_shots_style="chat"
    """

    few_shots = []


    for idx in selected_indices:

        issue = train_data[idx]


        few_shots.append({

            "input": (
                f"Title: {issue['title']}\n"
                f"Body: {issue['body']}"
            ),

            "output": (
                issue["labels"]
                .strip()
                .lower()
            ),

        })


    return few_shots


# ============================================================
# LOGGING PREDICTIONS
# ============================================================

def log_predictions(
    filepath,
    individual,
    title,
    body,
    ground_truth,
    prediction,
):

    file_exists = os.path.isfile(
        filepath
    )


    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    with open(
        filepath,
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            quoting=csv.QUOTE_MINIMAL,
        )


        if not file_exists:

            writer.writerow([
                "Timestamp",
                "Individual",
                "Title",
                "Body",
                "Ground_Truth",
                "Prediction",
            ])


        writer.writerow([
            timestamp,
            individual,
            title,
            body,
            ground_truth,
            prediction,
        ])


# ============================================================
# LOAD ORIGINAL SHARED PROMPT SETS
# ============================================================

# IMPORTANT:
#
# The original shared_prompt_sets.json must already exist.
#
# We DO NOT regenerate prompt sets here because the purpose
# of this experiment is to reevaluate the exact individuals
# that were used in the original random-selection experiment.
#
# Regenerating them could make Trial=N correspond to a
# different Individual.

if not os.path.exists(
    PROMPT_SET_FILE
):

    raise FileNotFoundError(

        "Original shared prompt-set file was not found:\n"
        f"{PROMPT_SET_FILE}\n\n"

        "The determinism experiment requires the exact "
        "shared_prompt_sets.json used in the original "
        "random-selection experiment."
    )


with open(
    PROMPT_SET_FILE,
    "r",
    encoding="utf-8",
) as f:

    SHARED_PROMPT_SETS = json.load(f)


print(
    "\nLoaded original shared prompt sets from:"
)

print(
    PROMPT_SET_FILE
)


# ============================================================
# SELECTED-PROMPT CSV HELPERS
# ============================================================

def parse_individual(value):
    """
    Convert the Individual column from the CSV into a
    Python list of integer indices.

    Examples:

        "[72]"
            -> [72]

        "[120, 40]"
            -> [120, 40]

        "[22, 69, 158, 177, 245, 264, 324, 347]"
            -> [22, 69, 158, 177, 245, 264, 324, 347]
    """

    if isinstance(
        value,
        list,
    ):

        return [
            int(x)
            for x in value
        ]


    if pd.isna(value):

        raise ValueError(
            "Individual is missing."
        )


    value = str(value).strip()


    try:

        parsed = ast.literal_eval(
            value
        )


    except (
        ValueError,
        SyntaxError,
    ) as e:

        raise ValueError(

            "Could not parse Individual value: "
            f"{value}"

        ) from e


    # In case a k=1 individual was accidentally stored
    # as a scalar rather than "[72]".
    if isinstance(
        parsed,
        (int, float),
    ):

        parsed = [parsed]


    if not isinstance(
        parsed,
        (list, tuple),
    ):

        raise ValueError(

            "Individual must be stored as a list/vector. "
            f"Received: {value}"

        )


    return [
        int(x)
        for x in parsed
    ]


def split_model_name(
    model_name,
):
    """
    Split:

        meta-llama/Llama-3.1-8B-Instruct

    into:

        provider = meta-llama
        model    = Llama-3.1-8B-Instruct
    """

    if "/" in model_name:

        provider, model = (
            model_name.split(
                "/",
                1,
            )
        )

    else:

        provider = ""
        model = model_name


    return provider, model


# ============================================================
# LOAD AND VERIFY SELECTED PROMPTS
# ============================================================

def load_selected_prompts():

    if not os.path.exists(
        SELECTED_PROMPTS_FILE
    ):

        raise FileNotFoundError(

            "Selected-prompt CSV was not found:\n"
            f"{SELECTED_PROMPTS_FILE}"

        )


    print(
        "\nLoading selected prompts from:"
    )

    print(
        SELECTED_PROMPTS_FILE
    )


    df = pd.read_csv(
        SELECTED_PROMPTS_FILE
    )


    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "Repo",
        "Model",
        "FewShot_Count",
        "Trial",
        "Individual",
        "Macro_F1",
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            "Selected-prompt CSV is missing "
            "required columns:\n"
            f"{missing_columns}"

        )


    # --------------------------------------------------------
    # Determine the configured model/provider.
    #
    # Example:
    #
    # MODEL_NAME =
    # meta-llama/Llama-3.1-8B-Instruct
    #
    # becomes:
    #
    # provider = meta-llama
    # model    = Llama-3.1-8B-Instruct
    # --------------------------------------------------------

    model_provider, model = (
        split_model_name(
            MODEL_NAME
        )
    )


    # --------------------------------------------------------
    # Repository must match exactly.
    # --------------------------------------------------------

    repo_mask = (

        df["Repo"]
        .astype(str)

        == REPO

    )


    # --------------------------------------------------------
    # Model must also match exactly.
    #
    # If Model_Provider exists in the CSV, use both fields.
    #
    # This prevents a median prompt selected for Gemma, for
    # example, from accidentally being evaluated with Llama.
    # --------------------------------------------------------

    if "Model_Provider" in df.columns:

        model_mask = (

            (
                df["Model_Provider"]
                .astype(str)

                == model_provider
            )

            &

            (
                df["Model"]
                .astype(str)

                == model
            )
        )


    else:

        # Fallback in case the CSV contains only Model.
        #
        # Accept either:
        #
        # Llama-3.1-8B-Instruct
        #
        # or:
        #
        # meta-llama/Llama-3.1-8B-Instruct

        model_values = (
            df["Model"]
            .astype(str)
        )


        model_mask = (

            (model_values == model)

            |

            (model_values == MODEL_NAME)

        )


    selected = df[
        repo_mask
        & model_mask
    ].copy()


    # --------------------------------------------------------
    # Ensure this repo/model actually exists in the CSV.
    # --------------------------------------------------------

    if selected.empty:

        display_columns = [

            column

            for column in [
                "Repo",
                "Model_Provider",
                "Model",
            ]

            if column in df.columns

        ]


        available = (

            df[display_columns]

            .drop_duplicates()

            .to_string(
                index=False
            )

        )


        raise ValueError(

            "\nNo selected prompts were found for the "
            "configured repository/model.\n\n"

            f"Configured repository:\n"
            f"    {REPO}\n\n"

            f"Configured model:\n"
            f"    {MODEL_NAME}\n\n"

            "Repository/model combinations available "
            "in the CSV:\n"

            f"{available}"

        )


    # --------------------------------------------------------
    # Normalize column types
    # --------------------------------------------------------

    selected[
        "FewShot_Count"
    ] = pd.to_numeric(

        selected[
            "FewShot_Count"
        ],

        errors="raise",

    ).astype(int)


    selected[
        "Trial"
    ] = pd.to_numeric(

        selected[
            "Trial"
        ],

        errors="raise",

    ).astype(int)


    selected[
        "Macro_F1"
    ] = pd.to_numeric(

        selected[
            "Macro_F1"
        ],

        errors="raise",

    )


    selected[
        "Individual"
    ] = selected[
        "Individual"
    ].apply(
        parse_individual
    )


    # --------------------------------------------------------
    # There should be exactly one selected median prompt for
    # each k for a given repository/model.
    # --------------------------------------------------------

    duplicated_k = selected[
        "FewShot_Count"
    ].duplicated(
        keep=False
    )


    if duplicated_k.any():

        duplicates = selected.loc[

            duplicated_k,

            [
                "FewShot_Count",
                "Trial",
                "Individual",
                "Macro_F1",
            ],

        ]


        raise ValueError(

            "More than one selected prompt was found "
            "for the same k:\n\n"

            f"{duplicates.to_string(index=False)}"

        )


    # --------------------------------------------------------
    # VERIFY AGAINST ORIGINAL shared_prompt_sets.json
    #
    # Trial is zero-based.
    #
    # Therefore:
    #
    # Trial = 0
    #     -> first element in JSON
    #
    # Trial = 12
    #     -> 13th element in JSON
    #
    # Example:
    #
    # k = 8
    # Trial = 12
    #
    # maps to:
    #
    # SHARED_PROMPT_SETS["8"][12]
    #
    # This must EXACTLY equal the Individual stored in the
    # median-prompts CSV.
    #
    # The order must also match exactly because ordering is
    # part of the prompt.
    # --------------------------------------------------------

    verified_rows = []


    for _, row in selected.iterrows():

        k = int(
            row["FewShot_Count"]
        )

        trial = int(
            row["Trial"]
        )

        csv_individual = [

            int(x)

            for x in row[
                "Individual"
            ]

        ]


        # -------------------------
        # Validate k
        # -------------------------

        if k not in FEW_SHOT_COUNTS:

            raise ValueError(

                f"Unexpected k={k} "
                "in selected-prompt CSV."

            )


        k_key = str(k)


        if k_key not in SHARED_PROMPT_SETS:

            raise KeyError(

                "k was not found in "
                "shared_prompt_sets.json:\n"

                f"k = {k}"

            )


        # -------------------------
        # Validate Trial
        # -------------------------

        number_of_trials = len(

            SHARED_PROMPT_SETS[
                k_key
            ]

        )


        if (
            trial < 0
            or trial >= number_of_trials
        ):

            raise IndexError(

                "Trial index is outside the available "
                "prompt-set range.\n\n"

                f"k: {k}\n"
                f"Trial: {trial}\n"

                f"Available trials: "
                f"0 to {number_of_trials - 1}"

            )


        # -------------------------
        # Retrieve exact original
        # individual using k+Trial.
        # -------------------------

        json_individual = [

            int(x)

            for x in SHARED_PROMPT_SETS[
                k_key
            ][trial]

        ]


        # -------------------------
        # Critical validation
        # -------------------------

        if (
            json_individual
            != csv_individual
        ):

            raise ValueError(

                "\nPROMPT MISMATCH DETECTED.\n\n"

                "The CSV Individual does not match "
                "the original prompt stored at the "
                "specified k and Trial.\n\n"

                f"Repository:\n"
                f"    {REPO}\n\n"

                f"Model:\n"
                f"    {MODEL_NAME}\n\n"

                f"k:\n"
                f"    {k}\n\n"

                f"Trial:\n"
                f"    {trial}\n\n"

                f"CSV Individual:\n"
                f"    {csv_individual}\n\n"

                f"JSON Individual:\n"
                f"    {json_individual}\n\n"

                "Evaluation has been stopped to prevent "
                "running the wrong prompt."

            )


        # ----------------------------------------------------
        # Also verify that the vector length equals k.
        # ----------------------------------------------------

        if len(
            json_individual
        ) != k:

            raise ValueError(

                "\nINVALID INDIVIDUAL LENGTH.\n\n"

                f"Repository: {REPO}\n"
                f"Model: {MODEL_NAME}\n"
                f"k: {k}\n"
                f"Trial: {trial}\n"

                f"Individual: "
                f"{json_individual}\n"

                f"Expected length: {k}\n"

                f"Actual length: "
                f"{len(json_individual)}"

            )


        verified_rows.append({

            "Repo":
                REPO,

            "FewShot_Count":
                k,

            "Trial":
                trial,

            # Use the JSON version after verification.
            "Individual":
                json_individual,

            "Original_Macro_F1":
                float(
                    row["Macro_F1"]
                ),

        })


    verified_df = pd.DataFrame(
        verified_rows
    )


    verified_df = (

        verified_df

        .sort_values(
            "FewShot_Count"
        )

        .reset_index(
            drop=True
        )

    )


    # --------------------------------------------------------
    # Print verified selections before inference starts.
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "SELECTED PROMPTS VERIFIED"
    )

    print(
        "=========================================="
    )

    print(
        f"Repository: {REPO}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Number of prompts: "
        f"{len(verified_df)}"
    )

    print(
        f"Repeats per prompt: "
        f"{NUM_REPEATS}"
    )


    for _, row in verified_df.iterrows():

        print(

            f"k={row['FewShot_Count']} | "

            f"Trial={row['Trial']} | "

            f"Individual="
            f"{row['Individual']} | "

            f"Original Macro-F1="
            f"{row['Original_Macro_F1']:.6f}"

        )


    print(
        "==========================================\n"
    )


    return verified_df


# ============================================================
# EVALUATION
# ============================================================

def evaluate_prompt(
    system_prompt,
    test_data,
    gen_num,
    individual,
    few_shots,
):
    """
    Evaluate one selected prompt on the complete test set.

    Prompting is exclusively chat-style:

        system_prompt
        +
        few_shots
        +
        current test issue

    Only Macro-F1 is computed.
    """

    predictions = []
    true_labels = []


    # --------------------------------------------------------
    # Correct CHAT-STYLE token count.
    #
    # This counts the system instructions together with the
    # chat-style few-shot examples.
    # --------------------------------------------------------

    system_prompt_token_count = (
        count_chat_style_tokens(

            MODEL_NAME,
            system_prompt,
            few_shots,

        )
    )


    print(

        f"[{gen_num}] "
        f"System + few-shot tokens: "
        f"{system_prompt_token_count}",

        flush=True,

    )


    # --------------------------------------------------------
    # Context check using the actual chat-style token count.
    #
    # This replaces the previous non-chat/string-prompt
    # context check.
    # --------------------------------------------------------

    total_context_tokens = (

        system_prompt_token_count

        + MAX_TEST_TOKENS

        + OUTPUT_TOKEN_BUFFER

        + BUFFER

    )


    print(

        f"[{gen_num}] "
        f"Maximum estimated total context: "
        f"{total_context_tokens}",

        flush=True,

    )


    if (
        total_context_tokens
        > MODEL_CONTEXT_LIMIT
    ):

        raise ValueError(

            "\nCHAT-STYLE PROMPT EXCEEDS "
            "MODEL CONTEXT LIMIT.\n\n"

            f"Model: {MODEL_NAME}\n"

            f"Individual: {individual}\n"

            f"System + few-shot tokens: "
            f"{system_prompt_token_count}\n"

            f"Max test tokens: "
            f"{MAX_TEST_TOKENS}\n"

            f"Estimated total: "
            f"{total_context_tokens}\n"

            f"Context limit: "
            f"{MODEL_CONTEXT_LIMIT}"

        )


    # --------------------------------------------------------
    # Prediction file
    #
    # Each repeated evaluation receives its own directory.
    #
    # Example:
    #
    # generations/
    #   gen_fs_8_trial_12_repeat_0/
    #       predictions_<uuid>.csv
    #
    #   gen_fs_8_trial_12_repeat_1/
    #       predictions_<uuid>.csv
    #
    # etc.
    # --------------------------------------------------------

    pred_dir = os.path.join(

        LOG_DIR,

        f"generations/"
        f"gen_{gen_num}",

    )


    os.makedirs(
        pred_dir,
        exist_ok=True,
    )


    pred_file = os.path.join(

        pred_dir,

        f"predictions_"
        f"{uuid.uuid4()}.csv",

    )


    # --------------------------------------------------------
    # Evaluate all test issues
    # --------------------------------------------------------

    for issue in test_data:

        user_prompt = (

            f"Title: {issue['title']}\n"
            f"Body: {issue['body']}\n"

        )


        try:

            predicted_label = predict(

                MODEL_NAME,

                user_prompt,

                system_prompt=
                    system_prompt,

                few_shots=
                    few_shots,

                few_shots_style=
                    "chat",

            )


        except Exception as e:

            print(

                f"Inference error: {e}",

                flush=True,

            )

            continue


        true_label = (

            issue["labels"]

            .strip()

            .lower()

        )


        predictions.append(
            predicted_label
        )


        true_labels.append(
            true_label
        )


        correct = (
            predicted_label
            == true_label
        )


        print(

            f"{'✅' if correct else '❌'} "

            f"Predicted: "
            f"{predicted_label} | "

            f"GT: "
            f"{true_label}",

            flush=True,

        )


        log_predictions(

            pred_file,

            individual,

            issue["title"],

            issue["body"],

            true_label,

            predicted_label,

        )


    # --------------------------------------------------------
    # Make sure evaluation actually produced predictions.
    # --------------------------------------------------------

    if not predictions:

        raise RuntimeError(

            f"No predictions were produced for "
            f"{gen_num}."

        )


    if len(predictions) != len(test_data):

        print(

            f"WARNING: {gen_num} produced "
            f"{len(predictions)} predictions for "
            f"{len(test_data)} test instances.",

            flush=True,

        )


    # --------------------------------------------------------
    # Macro-F1 only
    # --------------------------------------------------------

    macro_f1 = f1_score(

        true_labels,

        predictions,

        labels=VALID_LABELS,

        average="macro",

        zero_division=0,

    )


    print(

        f"Macro-F1: "
        f"{macro_f1:.4f}",

        flush=True,

    )


    return (
        macro_f1,
        system_prompt_token_count,
    )


# ============================================================
# RUN REPEATED DETERMINISM EVALUATIONS
# ============================================================

def run_selected_prompt(
    prompt_info,
):

    num_few_shots = int(

        prompt_info[
            "FewShot_Count"
        ]

    )


    original_trial = int(

        prompt_info[
            "Trial"
        ]

    )


    individual = list(

        prompt_info[
            "Individual"
        ]

    )


    original_macro_f1 = float(

        prompt_info[
            "Original_Macro_F1"
        ]

    )


    # --------------------------------------------------------
    # Build the chat-style few-shot examples once.
    #
    # The exact same ordered Individual is reused for every
    # repetition.
    # --------------------------------------------------------

    few_shots = generate_few_shots(

        individual,

        train_data,

    )


    # --------------------------------------------------------
    # System instructions
    #
    # Few-shot examples are NOT concatenated into this text.
    #
    # They are supplied separately to predict() through the
    # chat-style few_shots parameter.
    # --------------------------------------------------------

    system_instructions = (

        "Your job is to analyze and assign the most fitting "
        "labels to GitHub issues. Identify and assign the "
        "most accurate label for the issue using its title "
        "and body. Select the label exclusively from the "
        "options given in this list: \"question\", "
        "\"feature\", \"bug\". Ensure that the label is an "
        "exact match to one from the list. Output the label "
        "as a single string format with no additional "
        "characters, text, or formatting."

    )


    print(
        "\n==========================================",
        flush=True,
    )

    print(
        "STARTING SELECTED PROMPT",
        flush=True,
    )

    print(
        f"Repo: {REPO}",
        flush=True,
    )

    print(
        f"Model: {MODEL_NAME}",
        flush=True,
    )

    print(
        f"k: {num_few_shots}",
        flush=True,
    )

    print(
        f"Original Trial: {original_trial}",
        flush=True,
    )

    print(
        f"Individual: {individual}",
        flush=True,
    )

    print(
        f"Original Macro-F1: "
        f"{original_macro_f1:.6f}",
        flush=True,
    )

    print(
        f"Repeats: {NUM_REPEATS}",
        flush=True,
    )

    print(
        "==========================================\n",
        flush=True,
    )


    # --------------------------------------------------------
    # Evaluate the EXACT SAME prompt NUM_REPEATS times.
    # --------------------------------------------------------

    for repeat in range(
        NUM_REPEATS
    ):

        repeat_index = repeat


        print(
            "\n------------------------------------------",
            flush=True,
        )


        print(

            f"Repo: {REPO} | "

            f"Model: {MODEL_NAME} | "

            f"k={num_few_shots} | "

            f"Trial={original_trial} | "

            f"Repeat="
            f"{repeat_index + 1}/{NUM_REPEATS}",

            flush=True,

        )


        print(
            "------------------------------------------",
            flush=True,
        )


        # ----------------------------------------------------
        # Unique generation identifier
        #
        # Example:
        #
        # fs_8_trial_12_repeat_0
        # ----------------------------------------------------

        gen_num = (

            f"fs_{num_few_shots}"

            f"_trial_{original_trial}"

            f"_repeat_{repeat_index}"

        )


        macro_f1, token_count = (
            evaluate_prompt(

                system_instructions,

                test_data,

                gen_num=
                    gen_num,

                individual=
                    individual,

                few_shots=
                    few_shots,

            )
        )


        # ----------------------------------------------------
        # Save summary result
        # ----------------------------------------------------

        with LOG_LOCK:

            with open(

                LOG_FILE,

                "a",

                newline="",

                encoding="utf-8",

            ) as f:

                writer = csv.writer(f)


                writer.writerow([

                    datetime.now(),

                    REPO,

                    MODEL_NAME,

                    num_few_shots,

                    original_trial,

                    repeat_index,

                    individual,

                    original_macro_f1,

                    macro_f1,

                    token_count,

                ])


        print(

            f"Completed: "

            f"k={num_few_shots}, "

            f"Trial={original_trial}, "

            f"Repeat={repeat_index}, "

            f"Macro-F1={macro_f1:.4f}",

            flush=True,

        )


# ============================================================
# MAIN
# ============================================================

if not os.path.exists(
    LOG_FILE
):

    with open(

        LOG_FILE,

        "w",

        newline="",

        encoding="utf-8",

    ) as f:

        writer = csv.writer(f)


        writer.writerow([

            "Timestamp",

            "Repo",

            "Model",

            "FewShot_Count",

            "Trial",

            "Repeat",

            "Individual",

            "Original_Macro_F1",

            "Macro_F1",

            "Token_Count",

        ])


if __name__ == "__main__":

    # --------------------------------------------------------
    # Load and verify the median prompts before any inference
    # is performed.
    # --------------------------------------------------------

    selected_prompts = (
        load_selected_prompts()
    )


    prompt_records = (

        selected_prompts

        .to_dict(
            orient="records"
        )

    )


    print(
        "\nStarting determinism experiment...",
        flush=True,
    )


    print(
        f"Output directory: "
        f"{LOG_DIR}",
        flush=True,
    )


    print(
        f"Summary log: "
        f"{LOG_FILE}",
        flush=True,
    )


    print(
        f"Total selected prompts: "
        f"{len(prompt_records)}",
        flush=True,
    )


    print(
        f"Evaluations per prompt: "
        f"{NUM_REPEATS}",
        flush=True,
    )


    print(

        f"Total repeated prompt evaluations: "
        f"{len(prompt_records) * NUM_REPEATS}",

        flush=True,

    )


    # --------------------------------------------------------
    # Parallelization
    #
    # One worker is assigned to one selected prompt/k.
    #
    # The repetitions of that prompt are performed
    # sequentially inside run_selected_prompt().
    #
    # Set MAX_WORKERS = 1 if you want every single repeated
    # evaluation to run sequentially across the entire
    # determinism experiment.
    # --------------------------------------------------------

    worker_count = min(

        MAX_WORKERS,

        len(prompt_records),

    )


    with ThreadPoolExecutor(
        max_workers=worker_count
    ) as executor:

        futures = [

            executor.submit(

                run_selected_prompt,

                prompt_info,

            )

            for prompt_info
            in prompt_records

        ]


        for future in futures:

            # Calling result() ensures that any exception from
            # a worker is surfaced instead of being silently
            # ignored.
            future.result()


    print(
        "\n==========================================",
        flush=True,
    )

    print(
        "DETERMINISM EXPERIMENT COMPLETED",
        flush=True,
    )

    print(
        f"Results saved to:\n"
        f"{LOG_FILE}",
        flush=True,
    )

    print(
        "==========================================",
        flush=True,
    )