# Classical Metrics

We implement several classical metrics, including BLEU, METEOR, EM, SimCSE, S.-BERT, etc. Please follow the guide below as reference to understand and use this evaluator.

## Usage

```shell
sh test_llm.sh path/to/folder
```

## Environment

```
torch                 1.11.0
torchaudio            0.11.0
torchvision           0.12.0

pycocoevalcap         1.2
transformers          4.41.1
simcse                0.4
```

# GPT metrics

To evaluate the relatively complex QA and captioning results more accurately, following the previous practice in Large Multimodal Models (LMMs), we also use GPT as an auxiliary tool for evaluation. The curated prompts and related APIs are provided. Please follow the guide below to use the evaluator.

## Usage

1. Process the predicted results of models to expected forms.
2. You can then start the GPT evaluation following the script shown below.
    ```
    python gpt_4_evaluation_{}.py
    python colloec_gpt_score_{}.py
    ```