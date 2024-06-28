# LLM perspectives

This codebase is based on MMLU codebase. - link

## Installation

Setup the conda env
```
conda create -n llm_stability python=3.10
conda activate llm_stability
cd test/
pip install -r PLOSONE/requirements.txt 
```

For phi-1 and phi-2
```
conda create -n llm_stability python=3.10
conda activate llm_stability_437
cd test/
pip install -r PLOSONE/requirements.txt 
pip install transformers==4.37.0
```

## Setup environment variables
The rest of this guide will use the dummy model, which is a random baseline. For other models, you may wish to set various environment variables.

- To use OpenAI models, set the `OPENAI_API_KEY` env variable:
```commandline
export OPENAI_API_KEY="<your_key>"
```
- To use huggingface models, set the `HF_HOME` env variable to define your cache directory:

```commandline
export HF_HOME="$HOME/.cache/huggingface"
```

- To use huggingface gated models, set the `HF_TOKEN env variable`
```commandline
export HF_TOKEN="<your_token>"
```



## Evaluating and computing the stability

### Minimal example

You can run one evaluation with the following command

```
theme="chess"
python -u evaluate.py \
--engine "dummy" \
--experiment_name pvq_test \
--data_dir data/data_pvq \
--simulated-population-config personas/tolkien_characters/personas.json \
--simulated-conversation-theme $theme \
--simulated-conversation-n-messages 3 \
--permute-options-seed "testing_seed" \
--interlocutor-knows-persona \
--save_dir test_results/pvq_tolkien_dummy_$theme \
--pvq-version "pvq_auto" \
--verbose
```

This will evaluate a dummy (random) model simulating tolkien characters on PVQ.
It will save the results into: ```test_results/pvq_tolkien_dummy_chess_<timestamp>/results.json```

You can a different model in one of the following two ways:
- `--engine <model_name>` where `<model_name>` is the name of a config file in `./models/configs/*` (e.g. `--model phi-1`)
- `--model-config-path <path_to_config>`, where `<path_to_config>` is a config file path (e.g. `--model-config-path ./models/configs/phi-1.json`)


Now lets run the same command for "joke" conversation theme:

```
theme="chess"
python -u evaluate.py \
--engine "dummy" \
--experiment_name pvq_test \
--data_dir data/data_pvq \
--simulated-population-config personas/tolkien_characters/personas.json \
--simulated-conversation-theme $theme \
--simulated-conversation-n-messages 3 \
--permute-options-seed "testing_seed" \
--interlocutor-knows-persona \
--save_dir test_results/pvq_tolkien_dummy_$theme \
--pvq-version "pvq_auto" \
--verbose
```

And for "grammar":
```
theme="grammar"
python -u evaluate.py \
--engine "dummy" \
--experiment_name pvq_test \
--data_dir data/data_pvq \
--simulated-population-config personas/tolkien_characters/personas.json \
--simulated-conversation-theme $theme \
--simulated-conversation-n-messages 3 \
--permute-options-seed "testing_seed" \
--interlocutor-knows-persona \
--save_dir test_results/pvq_tolkien_dummy_$theme \
--pvq-version "pvq_auto" \
--verbose
```

Great now we have three results and we can compute the stability of the model with the following command:
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

The `data_analysis.py` script has some interesting flags.
 - `--no-ips` skips over ipsative stability computation (saves time)
 - `--plot-matrix` shows the pairwise comparison between pairs of contexts (themes)
- `--plot-ranks` visualizes the orders of participants in different contexts

You can see examples of other settings in the ``PLOSONE/run_scripts/demo_single.sh`` script.
The purpose of this script if to increase the clarity of this tutorial.

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

Following the comments in the script, you can modify those parameters.

From the test directory, you can run
```
bash PLOSONE/run_scripts/demo_single.sh
```

This will evaluate a dummy model, which chooses random answers on the PVQ questionniare.


## Campaign evaluations and stability computation


# Administer the questionnaire
In the previous example we showed how to run one evaluation: administer a questionnaire to a simulated population with **one** model, **one** conversation topic, and **one** seed.

In practice, we want to run those evaluations as a campaign (evaluate on **many** topics and **many** seeds).
We can do this with the ``PLOSONE/run_scripts/run_campaign_seeds.sh`` script.

This script accepts two arguments:
 - model: ```dummy``` (to see all available models run ``ls models/configs/``)
 - the experiment_type as defined by the following table

| experiment_type | task     | simulated population |
|-----------------|----------|----------------------|
| pvq_tolk        | PVQ      | tolkien characters   |
| pvq_fam         | PVQ      | real-world personas  |
| religion        | religion | real-world personas  |
| don             | donation | tolkien characters   |
| bag             | stealing | tolkien characters   |
| no_pop          | PVQ      | no population        |

We can use this script on a regular or a slurm-based machine. We want to run 25 evaluations: 5 seeds (answer permutations) x 5 conversation topics at once.

- Regular machine:

    This will run the 25 evaluations sequentially:
    ```commandline
    for i in {0..24}; do SLURM_ARRAY_TASK_ID=$i bash PLOSONE/run_scripts/run_campaign_seeds.sh dummy pvq_tolk ; done
    ```

    This will run the 25 evaluations in parallel:
    ```commandline
    for i in {0..24}; do SLURM_ARRAY_TASK_ID=$i bash PLOSONE/run_scripts/run_campaign_seeds.sh dummy pvq_tolk & done; wait
    ```

- Slurm-based machine:
    
    Make sure to modify your slurm config at the top of ``run_campaign_seeds.sh``.
    This will launch 25 parallel jobs
    ```commandline
    sbatch PLOSONE/run_scripts/run_campaign_seeds.sh dummy pvq_tolk
    ```

After evaluating a dummy model by any of the above commands, you should have the following folders structure:
```commandline
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0
results/stability_default_params_pvq_tolkien_characters/dummy/seed_2
results/stability_default_params_pvq_tolkien_characters/dummy/seed_4
results/stability_default_params_pvq_tolkien_characters/dummy/seed_6
results/stability_default_params_pvq_tolkien_characters/dummy/seed_8
```
Each should have 5 subdirectories with a results.json file, e.g:
```commandline
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/theme_chess_2024_04_29_20_33_03/results.json
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/theme_grammar_2024_04_29_20_23_19/results.json
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/theme_history_2024_04_29_20_33_06/results.json
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/theme_joke_2024_04_29_20_33_07/results.json
results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/theme_poem_2024_04_29_20_33_05/results.json
```

In other words the following command should return 25:

```
ls  results/stability_default_params_pvq_tolkien_characters/dummy/seed_*/*/results.json | wc -l
```

**Analyse the results - compute the stability**

We can compute the stability in one seed with the `data_analysis.py` script:
```commandline
python ./visualization_scripts/data_analysis.py --ips PLOSONE/results/stability_default_params_pvq_tolkien_characters/dummy/seed_0/*
```

As we are using a dummy model we should get close to zero stabilities:
```
------------------------
Aggregated metrics
------------------------
Rank-Order      Ipsative
-0.0030         -0.0088
```
TIP: you can add `--no-ips` argument to the `data_analysis.py` call to compute only Rank-Order stability (this is much faster).


We can also evaluate many seeds and models at once with the `campaign_data_analysis.py` script.
It takes the following arguments:
- `--fig-name` argument defines the experiment type to evaluate (looks in the correct results subdirectory), options are: `tolk_ro_t,fam_ro_t,religion_t,don_t,bag_t`
- `--assert-n-context 5` ensures that each seed has 5 topics
- `--all-models` evaluates all models in the `./models/configs` directory, if you do not set this argument you can manually define the models list on line `33`.


```commandline
python PLOSONE/data_analysis/campaign_data_analysis.py --fig-name tolk_ro_t --assert-n-context 5 --all-models
```

Towards the end of the output you should see a line as follows `random: -0.00029 +/- 0.005`, again the dummy model has near-zero stability.
The displayed figure should show one bar (it will not be easily visible as it is ~0.0).

**Other experiments**

Other experiments in the papers can be run with the following scripts:
`run_campaign_more_msgs.sh`, `run_campaign_more_themes.sh`, `run_campaign_no_pop_more_msgs`.
The scripts can be used in the same was as was shown above with `run_campaign_seeds.sh`.

# Adding a new model

Most models on the huggingface hub can be added by simply adding a new config file.
```commandline
touch ./models/configs/mymodelname.json
```

This assumes that the model can be used in the standard way as follows:
```
model = AutoModelForCausalLM.from_pretrained(self.model_id, **self.load_args)
tokenizer = AutoTokenizer.from_pretrained(self.model_id, **self.load_args)
prompt = "Hello"

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
- `model_id` is equivalent to the huggingface hub id
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

If a model requires a different transformers version, you can define the conda env to use in the `run_camapaign_seeds.sh` script on line `119`


