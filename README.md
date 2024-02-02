## Combining Transformers with Natural Language Explanations

Official repo of "Combining Transformers with Natural Language Explanations" paper.

## Preliminaries

This project is based on [cinnamon](https://github.com/lt-nlp-lab-unibo/cinnamon), a lightweight library for facilitating rapid prototyping and fostering reproducible experiments.

More information about ``cinnamon`` are provided in the [official documentation page](https://lt-nlp-lab-unibo.github.io/cinnamon/).

## Setup

- Create a folder where to locate all repositories.
- Clone ``cinnamon-core`` package: `git clone https://github.com/lt-nlp-lab-unibo/cinnamon_core.git`
- Clone ``cinnamon-generic`` package: `git clone https://github.com/lt-nlp-lab-unibo/cinnamon_generic.git`
- Clone ``cinnamon-th`` package: `git clone https://github.com/lt-nlp-lab-unibo/cinnamon_th.git`
- Clone this repository: `git clone https://github.com/lt-nlp-lab-unibo/bert-natural-explanations.git`
- Create a docker image with the provided ``Dockerfile``: `docker build . -t nle`
- Run a docker container in interactive mode: ``sh run_container.sh``
- You can exit from the container w/o closing it via: ``Ctrl+Shift+P - Ctrl+Shift+Q``

### Wandb

If you want to enable wandb in docker, open ``Dockerfile`` and set ``WANDB_API_KEY`` field.

Then, after setup, change `configurations/callback.py` as follows:

```python

class WandDBConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='entity',
                   value='',            # <--- HERE
                   is_required=True,
                   type_hint=str,
                   description='Profile name on wandb for login.')
        config.add(name='project',
                   value='nsf',
                   is_required=True,
                   type_hint=str,
                   description='Project name on wandb.')
        config.add(name='disabled',
                   value=True,          # <-- set to False to enable it
                   type_hint=bool,
                   description='If True, the callback is disabled.')

        return config

```


## Registrations

- Run ``python runnable/setup_registry.py``
- A ``registrations`` folder should be created in project folder containing JSON files.
- Each entry, **string key** from now on, in a JSON file with ``name=pipeline`` is an experiment we can run.

Example:

    "name:pipeline--tags:['routine.baseline', 'routine.data_loader.category=A', 'routine.hf', 'routine.model.baseline', 'routine.model.hf', 'routine.model.hf_model_name=distilbert-base-uncased']--namespace:nle/tos"

The above string found in ``registrations/tos/valid.json`` is used to train the DistilBERT baseline on ToS-A.

## Running an experiment

Running an experiment requires a string key (see **Registrations**).

- Run ``python runnables/train_model.py -n pipeline -t *tags* -ns *namespace* --serialize True|False``

where

- *tags*: check the tags field in a string key.
- *namespace*: ``nle/tos`` for ToS and ``nle/ibm`` for IBM.

Examples:

    DistilBERT ToS-A: python runnables/train_model.py -n pipeline -t routine.baseline routine.data_loader.category=A routine.hf routine.model.baseline routine.model.hf routine.model.hf_model_name=distilbert-base-uncased -ns nle/tos --serialize True

    MemDistilBERT IBM-Topics-1 (WS): python runnables/train_model.py -n pipeline -t routine.data_loader.topics=1 routine.data_splitter.topics=1 routine.hf routine.kb routine.model.hf routine.model.hf_model_name=distilbert-base-uncased routine.model.kb_sampler.attention routine.model.memory routine.model.ss_coefficient=0.0 -ns nle/ibm --serialize True

## Results

Scripts with ``--serialize True`` will have their results stored in ``runs`` folder.

## Where is the unstructured KB located?

- [**ToS**] ``datasets/ToS/``: there's a .txt file for each category.
- [**IBM**] ``datasets/IBM``: there's a .txt file for each topics group.

## Configurations

All configurations are in python and can be overridden/extended.

- [**General**] ``configurations`` folder.
- [**ToS**] ``tos/configurations`` folder.
- [**IBM**] ``ibm/configurations`` folder.

**Note: 1**: configurations should be self-explanatory. Feel free to change them.

**Note: 2**: Model configurations are in ``model.py``.

**Note: 3**: Changing/adding configurations changes the set of string keys generated! Check ``registrations`` folder.

## Cross-validation folds

The ``prebuilt_folds`` folder contains all the pre-computed folds for cross-validation.

## Contact

Federico Ruggeri: federico.ruggeri6@unibo.it

## Cite

You can cite our work as follows:

```
@misc{ruggeri2023combining,
      title={Combining Transformers with Natural Language Explanations}, 
      author={Federico Ruggeri and Marco Lippi and Paolo Torroni},
      year={2023},
      eprint={2110.00125},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Credits

Many thanks to all these people for their valuable feedback!

* Marco Lippi
* Paolo Torroni
* Andrea Galassi

Additional thanks to all reviewers for improving the manuscript.