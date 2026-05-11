# Few-Shot Design Matters: Understanding the Impact on LLM Performance and Cost - A Case Study on GitHub Issue Classification: Replication Package

This repository provides the complete replication package for our study on the impact of Few-Shot design on Large Language Models (LLMs) performance and cost for **GitHub Issue Classification**. It contains the datasets, scripts, prompts and results required to reproduce our analysis.

---

## 📂 Repository Structure

### Data Collection
The datasets are available in the [[data](data/)] folder. They include json files for the GitHub issues fetched from the **Facebook React**, **Bitcoin**, **OpenCV**, **Tensorflow**, and **Microsoft VsCode** projects.

### Prompt
The **[SYSTEM_PROMPT](https://github.com/stilab-ets/few-shot-design/blob/main/SYSTEM_PROMPT.txt)** - File containing the system prompt instructions used for all inferences across all models and projects.

### Key Scripts
- **[vllm_llm_logit_jinja.py](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/vllm_llm_logit_jinja.py)** – Python script used for running the vLLM server, inference and compute token counts.
- **[run_react_g12.slurm](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/run_react_g12.slurm)** Example of a Slurm script used to run jobs on a slurm cluster. This script loads the Gemma-3-12B-it model on the **Facebook React** project.
- **[react_random_selection_g12.py](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/random_selection/react_random_selection_g12.py)** This script runs the random selection for Gemma-3-12B-it model on the **Facebook React** project.
- **[react_sim_sel_g12.py](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/similarity_selection/react_sim_sel_g12.py)** This script runs the similarity selection for Gemma-3-12B-it model on the **Facebook React** project.
- **[react_zs_g12.py](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/zero_shot/react_zs_g12.py)** This script runs the zero-shot for Gemma-3-12B-it model on the **Facebook React** project.
- **[react_random_order_g12.py](https://github.com/stilab-ets/few-shot-design/blob/main/scripts/random_orders/react_random_order_g12.py)** This script runs the random orders for Gemma-3-12B-it model on the **Facebook React** project.

### Inference results
Contains the results of the runs across all 4 LLMs and 5 GitHub projects for: **Zero-shot**, **Random Selection**, **Similarity Selection**, **Random orders**, and **Targeted orders**.

All results are organized under the [[rqs](rqs/)] directory.

### Analysis
Replication of statistical analyses can be found in the [[results](results/)] directory. The folder contains jupyter notebooks to reproduce the quantitative evaluation performed in the paper.

---

## 📊 Results and Plots
All processed results and plots are organized under the [[results](results/)] directory:

- **[RQ1](results/rq1/)** – Few-Shot number impact on performance and cost.
- **[RQ2](results/rq2/)** – Few-Shot example selection impact on performance and cost - Fixed few-shot number, comparison between similarity selection vs. random selection.
- **[RQ3](results/rq3/)** – Few-Shot example order impact on performance - Fixed few-shot examples set reordered randomly.

---

## LLMs
Models used for this study:
- **[google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it)**
- **[meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)**
- **[mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)**
- **[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)**

---
