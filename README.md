# Measuring Massive Multitask Language Understanding This is the repository for [Measuring Massive Multitask Language Understanding](https://arxiv.org/pdf/2009.03300) by [Dan Hendrycks](https://people.eecs.berkeley.edu/~hendrycks/), [Collin Burns](http://collinpburns.com), [Steven Basart](https://stevenbas.art), [Andy Zou](https://andyzoujm.github.io/), Mantas Mazeika, [Dawn Song](https://people.eecs.berkeley.edu/~dawnsong/), and [Jacob Steinhardt](https://www.stat.berkeley.edu/~jsteinhardt/) (ICLR 2021). This repository contains OpenAI API evaluation code, and the test is available for download [**here**](https://people.eecs.berkeley.edu/~hendrycks/data.tar).


## Installation

Setup you conda env
```
conda create -n llm_persp python=3.9
pip install -r requirements.txt 
```


Install llama
``` pip install -r <llama_path>/requirements.txt```
``` pip install -e <llama_path>```

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

Run OpenAI Evaluation by:
```
python evaluate.py -k 1 -d data --e <model>
```
model values: "chat_gpt", "text-davinci-003", "text-davinci-002", "text-davinci-001", "curie", "babbage", "ada"

# Political compass
To evaluate on political compas use (from ./test)
```
python evaluate.py -k 0 -d data_political_compass --e <model>
```

And then (from ./test)
```
python evaluate_political_compass_csv.py -i ./test/results/political_compass/<results>.csv
```

# ToMi
```
python evaluate.py -k 3 -d data_tomi --e <model> -gqa
```

## Test Leaderboard

If you want to have your model added to the leaderboard, please reach out to us or submit a pull request.


Results of the test:
|                Model               | Authors |  Humanities |  Social Sciences  | STEM | Other | Average |
|------------------------------------|----------|:-------:|:-------:|:-------:|:-------:|:-------:|
| [Chinchilla](https://arxiv.org/abs/2203.15556) (70B, few-shot) | Hoffmann et al., 2022 | 63.6 | 79.3 | 54.9 | 73.9 | 67.5
| [Gopher](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) (280B, few-shot) | Rae et al., 2021 | 56.2 | 71.9 | 47.4 | 66.1 | 60.0
| [GPT-3](https://arxiv.org/abs/2005.14165) (175B, fine-tuned) | Brown et al., 2020 | 52.5 | 63.9 | 41.4 | 57.9 | 53.9
| [flan-T5-xl](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 46.3 | 57.7 | 39.0 | 55.1 | 49.3
| [UnifiedQA](https://arxiv.org/abs/2005.00700) | Khashabi et al., 2020 | 45.6 | 56.6 | 40.2 | 54.6 | 48.9
| [GPT-3](https://arxiv.org/abs/2005.14165) (175B, few-shot) | Brown et al., 2020 | 40.8 | 50.4 | 36.7 | 48.8 | 43.9
| [GPT-3](https://arxiv.org/abs/2005.14165) (6.7B, fine-tuned) | Brown et al., 2020 | 42.1 | 49.2 | 35.1 | 46.9 | 43.2
| [flan-T5-large](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 39.1 | 49.1 | 33.2 | 47.4 | 41.9
| [flan-T5-base](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 34.0 | 38.1 | 27.6 | 37.0 | 34.2
| [GPT-2](https://arxiv.org/abs/2005.14165) | Radford et al., 2019 | 32.8 | 33.3 | 30.2 | 33.1 | 32.4
| [flan-T5-small](https://arxiv.org/abs/2210.11416) | Chung et al., 2022 | 29.9 | 30.9 | 27.5 | 29.7 | 29.5
| Random Baseline           | N/A | 25.0 | 25.0 | 25.0 | 25.0 | 25.0 | 25.0


## Citation

If you find this useful in your research, please consider citing the test and also the [ETHICS](https://arxiv.org/abs/2008.02275) dataset it draws from:

    @article{hendryckstest2021,
      title={Measuring Massive Multitask Language Understanding},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }

    @article{hendrycks2021ethics,
      title={Aligning AI With Shared Human Values},
      author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
      journal={Proceedings of the International Conference on Learning Representations (ICLR)},
      year={2021}
    }
