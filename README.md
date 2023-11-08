# LLM perspectives

This codebase is based on MMLU codebase. - link

## Installation

Setup the conda env
```
conda create -n llm_persp python=3.9
pip install -r requirements.txt 

# install transformers
pip install git+https://github.com/huggingface/transformers.git
pip install -i https://test.pypi.org/simple/ bitsandbytes
conda install cudatoolkit -y
```

For openassistant create new env 
```
conda create --name llm_persp_oa --clone llm_persp
pip install git+https://github.com/huggingface/transformers@d04ec99bec8a0b432fc03ed60cea9a1a20ebaf3c
```

[//]: # (or)

[//]: # (```)

[//]: # (git clone https://github.com/huggingface/transformers.git)

[//]: # (cd transformers)

[//]: # (git checkout d04ec99bec8a0b432fc03ed60cea9a1a20ebaf3c)

[//]: # (pip install .)

[//]: # (```)




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

Scripts run_[neurips,iclr]_[pvq,hof,big5].sh contain the commands used to run our experiments.

# Evaluation

The bar_viz.py script is used for visualization evaluation and statistical analysis.
It can be used as such:
```
python visualization_scripts/bar_viz.py results_iclr/results_pvq_test_sim_conv_gpt-3.5-turbo-0301_perm_50_theme/*
```
Scripts [neurips,iclr]_evaluations.sh contain command to evaluate and plot the results from our experiments.


