# ST5230 Project Experiment Spec

## Research Question
How sensitive are LLM uncertainty estimates to semantically equivalent prompt perturbations?

We study whether semantically equivalent prompt variants can change model predictions and uncertainty estimates on the same input.

## Models
- Llama-3 8B Instruct
- Llama-3 70B Instruct
- GPT-5

## Datasets
- SciQ
- TruthfulQA

Planned sample size:
- SciQ: 200–300 samples
- TruthfulQA: 200–300 samples

We will fix the random seed and save the sampled subset for reproducibility.

## Output Task Format
For each sample, the model must return a final answer in a structured format.

Planned output format:
{"answer": "A"}

For all experiments, we aim to use a fixed answer space whenever possible, so that uncertainty can be compared consistently across prompt variants.

## Prompt Variant Principle
Prompt variants must be semantically equivalent.

They may differ in:
- wording
- sentence order
- formatting emphasis

They must NOT differ in:
- task meaning
- answer space
- extra hints
- reasoning requirements


## UQ_ICL
### Dependencies
This code is written in Python. To use it you will need:
- Numpy - 1.16.2
- Scipy - 1.2.1
- pandas - 0.23.4
- Transformers - 4.35.0
- PyTorch 1.10.0+
- datasets - 2.15.0

### Usage
#### Data
The data can be downloaded from the file by datasets Python library.

#### Test Models
There are five datasets, you can test the results of different datasets with using the executable files (*cola.sh, ag_news.sh, financial.sh, ssh.sh, sentiment.sh*) provided.

Note that the parameter value ranges are hyper-parameters, and different range 
may result different performance in different dataset, be sure to tune hyper-parameters carefully. 
