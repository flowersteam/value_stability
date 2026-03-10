# Stick to Your Role!

This codebase is based on MMLU codebase. - https://github.com/hendrycks/test

This is the code to evaluate models on the Stick to Your Role! leaderboard, but also to recreate results from our related paper - https://arxiv.org/abs/2402.14846.

This README.md concerns the leaderboard, for the paper see thie [README](PLOSONE/README.md)



# Installation

## Conda environment
```
conda create -n llm_stability_311 python=3.11
conda activate llm_stability_311
cd test/
pip install -r requirements.txt 

pip install flash-attn 
```

## install R packages
```R```
And then in R shell
```
install.packages("lavaan")
install.packages("jsonlite")
```


## Setup environment variables

- To use OpenAI models with the OpenAI API, set the `OPENAI_API_KEY` env variable (e.g. in your `.bashrc`):
```commandline
export OPENAI_API_KEY="<your_key>"
```
- To use OpenAI model with the Azure API, set the variables for each model, for example:
```
export AZURE_OPENAI_ENDPOINT_gpt_35_turbo_0125="<your_endpoint>"
export AZURE_OPENAI_KEY_gpt_35_turbo_0125="<your_key>"
export AZURE_OPENAI_API_VERSION_gpt_35_turbo_0125="<your_version>"
```
For additional models (apart from gpt-3.5-turbo-0125/1106 and gpt-4o-0513), refer to `models/openaimodel.py` to make the additional changes.
- To use huggingface models, set the `HF_HOME` env variable to define your cache directory:

```commandline
export HF_HOME="$HOME/.cache/huggingface"
```

- To use huggingface gated models, set the `HF_TOKEN env variable`
```commandline
export HF_TOKEN="<your_token>"
```

- To use togetherAI models, install the library, and set the `TOGETHER_API_KEY`
```commandline
conda activate llm_stability
pip install together
export TOGETHER_API_KEY="<your_token>"
```



# Evaluating models and computing the stability


## Minimal example: one model in one context

You can run one evaluation with the following command

```
theme="joke"
python -u evaluate.py \
--model-config-path ./models/leaderboard_configs/dummy.json \
--engine "dummy" \
--experiment_name pvq_test \
--data_dir data/data_pvq \
--simulated-population-config tolkien_characters \
--simulated-conversation-theme $theme \
--simulated-conversation-n-messages 3 \
--permute-options-seed "testing_seed" \
--interlocutor-knows-persona \
--save_dir test_results/pvq_tolkien_dummy_$theme \
--pvq-version "pvq_auto" \
--verbose
```

This will evaluate a dummy (random) model simulating tolkien characters on PVQ.
It will save the results into: ```test_results/pvq_tolkien_dummy_chess_2024_04_29_19_54_19/results.json```

You can a different model in one of the following two ways:
- `--engine <model_name>` where `<model_name>` is the name of a config file in `./models/configs/*` (e.g. `--model phi-1`)
- `--model-config-path <path_to_config>`, where `<path_to_config>` is a config file path (e.g. `--model-config-path ./models/configs/phi-1.json`)


Now repeat the same command but with `theme="joke"` and with `theme="grammar"`.

You should have three results. You can now compute the stability of the model with the following command:
```
python ./visualization_scripts/data_analysis.py test_results/pvq_tolkien_dummy_*
```

This should give stabilities close to zero because we are testing a dummy model, which only selects a random answer:
```
------------------------
Aggregated metrics
------------------------
Rank-Order      Ipsative
0.0051          0.0074
```

You can see examples of other settings in the ``Leaderboard/run_scripts/run_single.sh`` script.
The purpose of this script if to increase the clarity of this tutorial.


Following the comments in the script, you can modify the relevant parameters.

From the test directory, you can run
```
bash Leaderboard/run_scripts/run_single_demo.sh
```

This will evaluate a dummy model, which chooses random answers on the PVQ questionniare.


## Evaluate and analyze a single model in all contexts

In the previous example we showed how to run one evaluation: administer a questionnaire to a simulated population with **one** model, **one** conversation topic.

### Evaluate

In practice, we want to run those evaluations as a campaign, i.e. evaluate on **many** topics.
We can do this with the ``Leaderboard/run_scripts/run_stability_leaderboard.sh`` script.

The model is defined as an argument to the script (e.g. ```dummy```).
This should correspond to a config file in `models/leaderboard_configs/` (e.g. `models/leaderboard_configs/dummy.json`).

We can use this script on a regular or a slurm-based machine:

- Regular machine:

    This will run the 9 evaluations sequentially:
    ```commandline
    for i in {0..9}; do SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh dummy ; done
    ```

    This will run the 25 evaluations in parallel:
    ```commandline
    for i in {0..9}; do SLURM_ARRAY_TASK_ID=$i bash Leaderboard/run_scripts/run_stability_leaderboard.sh dummy & done; wait
    ```

- Slurm-based machine:
    
    Make sure to modify your slurm config at the top of the ``run_stability_leaderboard.sh`` script.
    This will launch 9 parallel jobs
    ```commandline
    sbatch Leaderboard/run_scripts/run_stability_leaderboard.sh dummy 
    ```
    
    If you are using vllm make sure to set the arguments in `run_stability_leaderboard_vllm.sh`. And then you can use the following command to run the evaluations:
    ```commandline
    sbatch Leaderboard/run_scripts/run_stability_leaderboard_vllm.sh dummy_vllm 
    ```
  

After evaluating a dummy model by any of the above commands, you should have the following folders and jsons:
```commandline
Leaderboard/results/stability_leaderboard/dummy/chunk_0_2024_06_03_22_08_24/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_1_2024_06_03_22_08_24/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_2_2024_05_27_20_29_11/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_3_2024_05_27_20_29_11/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_4_2024_05_27_20_29_11/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_chess_0_2024_05_28_17_57_08/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_grammar_1_2024_05_29_18_57_05/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_no_conv_2024_05_27_20_29_11/results.json
Leaderboard/results/stability_leaderboard/dummy/chunk_svs_no_conv_2024_06_23_13_37_28/results.json
```

In other words the following command should return 9:
```
ls Leaderboard/results/stability_leaderboard/dummy/*/results.json | wc -l
```

### Analyze the results

We can compute the stability of one model with the `data_analysis.py` script by passing the directories with the results:
```commandline
python ./visualization_scripts/data_analysis.py Leaderboard/results/stability_leaderboard/dummy/*
```

As we are using a dummy model we should get close to zero stability:
```
------------------------
Aggregated metrics
------------------------
Rank-Order      Ipsative
0.0037         0.0050
```

The `data_analysis.py` script has a set of usefull parameters:
- `--no-ips` to compute only Rank-Order stability (this is much faster)
- `--plot-matrix` shows the correlations between pairs of contexts
- `--plot-matrix` shows the correlations between pairs of contexts
- `--plot-structure` shows the circular structure of values
- `--structure` only computes the correlation with the theoretical order
- `--cronbach-alpha` compute the cronbach alphas
- `--plot-ranks` visualizes the different simulated participant values
- `--results-json-savepath` save the results into a json
- `--assert-n-dirs <n>` asserts that there are n loaded jsons

## Evaluate and analyze multiple models in all contexts

### Evaluate multiple models

The previous sections showed how to evaluate one model. All models can be evaluated on a slurm-based server by running 
```
bash Leaderboard/run_scripts/run_all_models_leaderboard.sh
```
This script is very clear and intended to be used for reference.
It is simple or run it on a regular machine, with the changes described above.

### Analyze multiple models
Once all the models have been evaluated we can analyse all of their results by running
```
bash Leaderboard/data_analysis/analyse_all_models.sh
```
This script is also very clear and can be used for reference.
It This scripts should save the results for each model in `Leaderboard/data_analysis/analysis_results` (one `json` for each model).

### Rank the model (create the leaderboard)

The jsons in `Leaderboard/data_analysis/analysis_result` can be used to rank the models.

We can rank the models by running (first add the model to the list of models in the same script)
```
# To get the data for the leaderboard you can run
python Leaderboard/data_analysis/rank_models.py 

# To order models based on a single metrics you can use
python Leaderboard/data_analysis/rank_metric.py 
```

These scripts rank them models and also compute the leaderboards sensitivity and stability with [benchbench](https://socialfoundations.github.io/benchbench/).

# Adding a new model

Most models on the huggingface hub can be added by simply adding a new config file.
```commandline
touch ./models/leaderboard_configs/mymodelname.json
```

This assumes that the model can be used in the standard way as follows:
```
model = AutoModelForCausalLM.from_pretrained(self.model_id, **self.load_args)
tokenizer = AutoTokenizer.from_pretrained(self.model_id, **self.load_args)
prompt = "Hello

# for chat models
input_ids = self.tokenizer.apply_chat_template(
[{"role":"user", "content", prompt}], return_tensors="pt", add_generation_prompt=True).to(self.model.device
)

# for base models
input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device).input_ids

output_seq = model.generate(input_ids=input_ids, **generation_args, return_dict_in_generate=True, output_scores=True, stopping_criteria=stopping_criteria)
response = tokenizer.decode(output_seq.sequences[0][len(input_ids[0]):], skip_special_tokens=True)
```
You can refer to `models/hugginfacemodel.py` for more details.

Here is an example of the config file for the LLaMa-2-7b model:
```
{
  "model_class": "HuggingFaceModel",
  "model_id": "meta-llama/Llama-2-7b-hf",
  "system_message": true,
  "base_model_template": true,
  "load_args": {
    "torch_dtype": "torch.float16",
    "trust_remote_code": true,
    "device_map": "auto",
    "token": "HF_TOKEN"
  },
  "generation_args": {
    "max_new_tokens": 100,
    "do_sample": true,
    "top_p": 0.9,
    "top_k": 50,
    "temperature": 0.6,
    "repetition_penalty": 1.2,
    "num_beams": 1
  }
}
```
It should be filled as follows:
- `model_class` - should be `"HuggingFaceModel"` unless for you want to do define your own class (it should be in ``models/``)
- `model_id` is equivalent to the huggingface hub id or a path to a local model
- `base_model_template` - `true` if the model is a base model, `false` if the model is chat or instruct tuned (the tokenizer has the `apply_chat_template` function)
- `system_message` - `true` if the model has the system message input or if it's a base model
- `load_args` - arguments that will be passed to `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` in addition to the `model_id` 
- `generation_args` - arguments that will be passed to the `generate` function while simulating conversations

Minor points:
- HF_TOKEN is automatically parsed to the token encoded in the "HF_TOKEN" environment variable.
- The string "torch.float16" is parsed to torch.float16 value.

For additional details refer to `models/__init__.py` (`create_model` and `load_model_args` methods).


After correctly configuring the model config file, the new model can be passed as the `engine` argument to evaluate.py (name of the json file without the extension).
You should be able to evaluate it as any other model.
For example, using:
```commandline
sbatch run_campaign_seeds.sh mymodelname pvq_tolk
```

If a model requires a different transformers version, you can define the conda env to use in the `run_camapaign_seeds.sh` script on line `103`


