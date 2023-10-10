# LLM perspectives

This codebase is based on MMLU codebase. - link

## Installation

Setup the conda env
```
conda create -n llm_persp python=3.9
pip install -r requirements.txt 
pip install git+https://github.com/huggingface/transformers@c612628045821f909020f7eb6784c79700813eda
```


### Install llama if you want to run LLaMa models (this step is not needed to recreate experiments in the paper)
Initialize and fetch the llama submodule
```
git submodule update --init --recursive
```

``` pip install -r llama/requirements.txt```
``` pip install -e llama/```

Set up llama_dir in evaluate.py - the dir with checkpoints and encoder

Run LLaMa evaluation by:
```
torchrun --nproc_per_node <MP> evaluate.py -k 1 -d data -e <model>
```

Different models require different MP values:

| Model     | MP |
|-----------|----|
| llama_7B  | 1  |
| llama_13B | 2  |
| llama_30B | 4  |
| llama_65B | 8  |

Replace 7B with 13B, 30B, 65B per choice.

# Running experiments

Script run_dummy.sh shows an example of how to run a model.

Scripts run_neurips_[pvq,hof,big5].sh contain the commands used to run our experiments.

# Evaluation
Script neurips_evaluations.sh contrains the command to evaluate and plot the results from those experiments.
