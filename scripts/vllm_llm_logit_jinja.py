from openai import OpenAI
from transformers import AutoTokenizer
import os

# vLLM server URL
VLLM_SERVER_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

# Caches
_tokenizer_cache = {}
_logit_bias_cache = {}

def get_tokenizer(model_id):
    if model_id not in _tokenizer_cache:
        _tokenizer_cache[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _tokenizer_cache[model_id]

def get_logit_bias(model_name):
    if model_name in _logit_bias_cache:
        return _logit_bias_cache[model_name]

    tokenizer = get_tokenizer(model_name)
    allowed_words = ["bug", "feature", "question"]
    allowed_ids = [tokenizer.encode(word, add_special_tokens=False)[0] for word in allowed_words]

    logit_bias = {token_id: 100 for token_id in allowed_ids}
    _logit_bias_cache[model_name] = logit_bias
    return logit_bias

def predict(
    model_name, 
    user_prompt, 
    system_prompt=None, 
    few_shots=None,
    few_shots_style="chat"  # "concatenate" or "chat"
):
    """
    Make a prediction using OpenAI chat endpoint (vLLM server).
    
    Parameters:
    - model_name: string, model identifier
    - user_prompt: string, the query to classify
    - system_prompt: string, instructions (optional)
    - few_shots: list of dicts, each with {"input": ..., "output": ...}
    - few_shots_style: "concatenate" (few-shots inside system prompt) 
                       or "chat" (few-shots as user/assistant alternation)
    """
    client = OpenAI(base_url=VLLM_SERVER_URL, api_key="EMPTY")

    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({"role": "system", "content": "Your job is to analyze and assign the most fitting labels to GitHub issues. Identify and assign the most accurate label for the issue using its title and body. Select the label exclusively from the options given in this list: \"question\", \"feature\", \"bug\". Ensure that the label is an exact match to one from the list. Output the label as a single string format with no additional characters, text, or formatting."})

    if few_shots:
        if few_shots_style == "concatenate":
            # Append few-shots to system prompt
            few_shot_text = "\n".join(
                f"Input: {fs['input']}\nOutput: {fs['output']}" for fs in few_shots
            )
            # Update system message content
            messages[0]["content"] += "\n" + few_shot_text
        elif few_shots_style == "chat":
            # Add each example as user -> assistant message
            for fs in few_shots:
                messages.append({"role": "user", "content": fs["input"]})
                messages.append({"role": "assistant", "content": fs["output"]})
        else:
            raise ValueError("few_shots_style must be 'concatenate' or 'chat'")

    # Add the actual user query
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=1,
        logit_bias=get_logit_bias(model_name),
    )

    choice = completion.choices[0]

    prediction = choice.message.content.strip().lower()

    return prediction


def count_input_tokens(model_name, text):
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text, add_special_tokens=False))

def count_chat_style_tokens(model_name, system_prompt, few_shots=None):
    """
    Count tokens for a chat-style prompt (system + few-shots) the way chat-style models see it.
    few_shots: list of {"input": ..., "output": ...}
    """
    tokenizer = get_tokenizer(model_name)
    total_tokens = 0

    # System message
    total_tokens += len(tokenizer.encode(system_prompt, add_special_tokens=False)) + 2  # role/separator

    if few_shots:
        for fs in few_shots:
            # User message
            total_tokens += len(tokenizer.encode(fs["input"], add_special_tokens=False)) + 2
            # Assistant message
            total_tokens += len(tokenizer.encode(fs["output"], add_special_tokens=False)) + 2

    return total_tokens
