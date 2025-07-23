# Introduction
This repository contains an official implementation of **Targeted Test Selection**.

This work introduces a strategy
that selects which tests are likely to fail after the implementation of
targeted software changes. The approach proposes the construction
and original preprocessing of numerous factors about tests, soft-
ware changes and their co-occurrences. We incrementally propose
increasingly advanced techniques for obtaining additional features
based on code analysis and project structure to improve the quality
of test selection. The obtained features are used to train a machine
learning model that predicts the probability of a given test falling
on a given code change.

# Dependencies
To install project dependencies, execute
```
pip install -r requirements.txt
```

If you are going to use csaxgb or cba model,
run

```
huggingface-cli login --token $YOUR_HF_TOKEN
```

For information about the token, visit [User Access Token](https://huggingface.co/docs/hub/security-tokens)

# Benchmarks
1. Download IOF/ROL `feature-engineered.csv`: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GIJ5DE and place it as `benchmarks/datasets/iofrol.csv`

2. Download GSDTSR `feature-engineered.csv`: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MJFKDN and place it as `benchmarks/datasets/gsdtsr.csv`

3. Follow the instruction in https://github.com/Amannor/redhat_final_proj/tree/main to
create a psr.csv and place it as `benchmarks/datasets/redh.csv`. We used the data saved in the Amannor's repository.
4. Run
```
python benchmarks.py
```

# Training Instructions
1. Load the data according to the structure in the `data_structure/train_data_structure/`. Read `data_structure/README.md` for more information. When calling `train.py`, specify the path to this data.

2. Run the following command in the root of the project. You can see all of the parameters by running `python train.py --help`.
```zsh
python train.py --data data --num_train_commits 5 --model xgb
```

# Inference Instructions
1. Load the data for inference according to the structure: `data_structure/predict_data_structure/`. Read `data_structure/INFO.md` for more information. When calling `predict.py`, specify the path to this data.

2. Run the following command in the root of the project. You can see all of the parameters by running `python predict.py --help`.

```zsh
python predict.py --input data_inference --model xgb --output output/output.json
```

3. The prediction results, sorted in descending order of failure probability, will be available in `output/output.json`.

# Files description
.
├── benchmarks.py - script for running benchmarks
├── train.py - script for training models
├── predict.py - script for inference
├── configs - configs for models
├── data_structure - data structure for training and inference
├── src
│   ├── benchmarks - source files for benchmarks
│   ├── models - source files for models
│   ├── preprocessing - source files for preprocessing data (cleaning data, creating dataset, etc.)
|   └── metrics - utils for calculation metrics 
└── predict.py - script for inference


# License
Targeted Test Selection is available under the Apache License 2.0. See the LICENSE file for more info.
