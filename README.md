# LLM perspectives

This codebase is based on MMLU codebase. - link

## Installation

Setup the conda env
```
conda create -n llm_stability python=3.9
conda activate llm_persp
cd test/
pip install -r requirements.txt 

# install transformers
pip install git+https://github.com/huggingface/transformers.git
pip install -i https://test.pypi.org/simple/ bitsandbytes
conda install cudatoolkit -y
```


# Evaluating a model

The ``run_single.sh`` contains an example of how to evaluate a models.

It requires to set 7 parameters, which are by default set to:
```
1. Theme:grammar
2. Seed:1
3. N messages:3
4. LLM:dummy
5. Questionnaire:pvq
6. Population:tolkien_characters
7. Experiment name:test
```

You can modify those parameters inside the script (following the comments).


From the test directory, run
```
bash run_single.sh
```

This will evaluate a dummy model, which chooses random answers on the PVQ questionniare.

# Running all experiments


All the experiments in the paper are shown in ```run_campain*.sh``` scripts.

These are slurm scripts and enable parallel evaluation of different topics and seeds. These scripts require an argument, which defines the model.
The following command evaluates the Mistral-7B-Instruct-v0.2 model (```model_idx=7```):

```
sbatch run_campaign_sim_conv_pvq_seeds.sh <model_idx>
```

Here is a list of models and their indices (this correponds to the index in the model list in run_campain*.sh scripts) :

| Model | model_idx |
|-------|----------------|
|llama_2_7b| 0 |
|llama_2_13b| 1 |
|llama_2_7b_chat| 2 |
|llama_2_13b_chat| 3 |
|zephyr-7b-beta| 4 |
|Mistral-7B-v0.1| 5 |
|Mistral-7B-Instruct-v0.1| 6 |
|Mistral-7B-Instruct-v0.2| 7 |
|llama_2_70b| 8 |
|llama_2_70b_chat| 9 |
|Mixtral-8x7B-v0.1-4b| 10 |
|Mixtral-8x7B-Instruct-v0.1-4b| 11 |
|Mixtral-8x7B-v0.1| 12 |
|Mixtral-8x7B-Instruct-v0.1|13 |
|phi-2| 14 |
|phi-1| 15 |
|Qwen-72B| 17 |
|Qwen-14B| 18 |
|Qwen-7B| 19 |

Those scripts also require setting the population and the questionnaire. They can easily be changed following the scripts comments.
By default, they are set to fictional characters and PVQ:
```
## PVQ - tolkien characters
test_tag="pvq"
experiment_name="pvq_test"
data_dir="data_pvq"
population_type="tolkien_characters"
```


The scripts are used for various experiments as follows:

Experiments with simulated populations: ```run_campaign_sim_conv_pvq_seeds.sh``` 

Experiments with simulated populations and increasing conversation length:
```run_campaign_sim_conv_pvq_msgs.sh```

Experiments with no persona instructions: ```run_campaign_sim_conv_no_pop.sh```

Ablation study on the system message with LLaMa-2 models: ```run_campaign_sim_conv_pvq_NO_SYSTEM.sh```


## Non-slurm machine

The ```run_campain*.sh``` scripts can be run on a regular machine my manually setting the ```SLURM_ARRAY_TAK_ID''' variable as follows:

1. Check the slurm array size parameter

```
grep "$SBATCH --array=" run_campaign_sim_conv_pvq_seeds.sh
```

The expected output is:
```#SBATCH --array=0-29 # themes x n_seeds -> 6x5```
This means that slurm would run **30 parallel jobs** corresponding to 6 themes (5 + no theme) and 5 seeds.

2. Run the jobs manually

You can run the 30 evaluations sequentially on a regular machine as follows:
```
for i in {0..29}; do SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh <model_idx> ; done
```

or in parallel as follows:
```
for i in {0..29}; do SLURM_ARRAY_TASK_ID=$i bash run_campaign_sim_conv_pvq_seeds.sh <model_idx> & done
```


