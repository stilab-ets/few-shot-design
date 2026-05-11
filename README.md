# Few-Shot Design Matters: Understanding the Impact on LLM Performance and Cost - A Case Study on GitHub Issue Classification: Replication Package

This repository provides the complete replication package for our study on the impact of Few-Shot design on Large Language Models (LLMs) performance and cost for **GitHub Issue Classification**. It contains the datasets, scripts, prompt and results required to reproduce our analysis.

---

## 📂 Repository Structure

### Data Collection
The datasets are available in the [data](data/) folder. They include json files for the GitHub issues fetched from the **Facebook React**, **Bitcoin**, **OpenCV**, **Tensorflow**, and **Microsoft VsCode** projects.

### Prompt
The **[SYSTEM_PROMPT]()** - File containing the system prompt instructions used for all inferences across all models and projects.

### Key Scripts
- **[vllm_llm_logit_jinja.py]()** – Python script used for running the vLLM server, inference and compute token counts.
- **[run_react_g12.slurm]()** Exampl of a Slurm script used to run jobs on a slurm cluster. This script loads the Gemma-3-12B-it model on the **Facebook React** project.

### Inference results
Contains the results of the runs across all 4 LLMs and 5 GitHub projects for: **Zero-shot**, **Random Selection**, **Similarity Selection**, **Random orders**, and **Targeted orders**.

All results are organized under the [rqs](rqs/) directory.

### Analysis
Replication of statistical analyses can be found in the [results](results/) directory. The folder contains jupyter notebooks to reproduce the quantitative evaluation performed in the paper.

---

## 📊 Results and Plots
All processed results and plots are organized under the [Results](Results/) directory:

- **[RQ1](Results/RQ1/)** –
- **[RQ2](Results/RQ2/)** –
- **[RQ3](Results/RQ3/)** –

---