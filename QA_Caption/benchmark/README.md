# MMScan QA/Captioning benchmark

## 🏠 About

This codebase preliminarily integrates the first 3 models shown below and will include the other two soon.

1. 3D-LLM
2. LL3DA
3. LEO
4. PointLLM
5. Chat3D-v2

## 📚 Basic Guide

This codebase is a beta version for understanding how we organize the MMScan's data and conduct the training and benchmark process. Next, we provide the guide for different aspects.

### Dataset

Please follow the `README.md` in `./data/` to organize the data.

### Evaluation Metric

Here we implement both classical metrics and the GPT metric in `./evaluation`. Please follow the `README.md` in `./evaluation` to evaluate the models' results.

### Models

#### 3D-LLM

1. Please follow the guide in `./llmzoo/3D-LLM/README.md` to install the 3D-LLM environment and download the pre-trained model.

2. We only implemented the zero-shot setting in the 3D-LLM model. To run the zero-shot 3D-LLM.

    ```bash
    conda activate lavis
    cd ./llmzoo/3D-LLM/3DLLM_BLIP2-base
    python inference_mmscan.py
    ```
#### LL3DA

1. Please follow the guide in `./llmzoo/LL3DA/README.md` to install the LL3DA environment and download the pre-trained model.

2. We implemented both the zero-shot setting and fine-tuning setting in the LL3DA model. 

    ```bash
    cd ./llmzoo/LL3DA
    bash scripts/tuning.mmscanqa.sh  # For tuning the model
    bash scripts/eval.mmscanqa.sh  # For inference with the model
    ```
#### LEO

1. Please follow the guide in `./llmzoo/LEO/README.md` to install the LEO environment and download the pre-trained model.

2. We implemented both the zero-shot setting and fine-tuning setting in the LL3DA model. 

    ```bash
    cd ./llmzoo/LEO
    bash scripts/train_tuning_mmscan.sh  # For tuning the model
    bash scripts/test_tuning_mmscan.sh  # For inference with the model
    ```