# LLM perspectives

This codebase is based on MMLU codebase. - link

## Installation

Setup the conda env
```
conda create -n llm_persp python=3.9
conda activate llm_persp
pip install -r requirements.txt 

# install transformers
pip install git+https://github.com/huggingface/transformers.git
pip install -i https://test.pypi.org/simple/ bitsandbytes
conda install cudatoolkit -y
```


[//]: # (or)

[//]: # (```)

[//]: # (git clone https://github.com/huggingface/transformers.git)

[//]: # (cd transformers)

[//]: # (git checkout d04ec99bec8a0b432fc03ed60cea9a1a20ebaf3c)

[//]: # (pip install .)

[//]: # (```)


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


