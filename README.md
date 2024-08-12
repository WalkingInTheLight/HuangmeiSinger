# A Dataset for Synthesizing Huangmei Opera Singing


## Environments
1. Create an environment of anaconda:

    ```sh
    conda create -n your_env_name python=3.8
    conda activate your_env_name 
    pip install -r requirements.txt
    ```

2. Download the pre-trained [vocoder](https://github.com/MoonInTheRiver/DiffSinger/releases/download/pretrain-model/0109_hifigan_bigpopcs_hop128.zip) and unzip it to the `checkpoints` folder

3. Download the [dataset](data/HuangmeiOpera_Dataset) and unzip it to the `data/raw` folder


## Preprocessing
Run the following command to preprocess the dataset:

```sh
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/binarize.py --config configs/hmxopera_config.yaml
```


## Training
Run the following command to train:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/hmxopera_config.yaml --exp_name your_exp_name --reset
```

## Inferencing
Run the following command to inference:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/hmxopera_config.yaml --exp_name your_exp_name --reset --infer
```
## Visualization
Run the following command to view the results of the training:

```sh
tensorboard --logdir_spec checkpoints/your_exp_name/lightning_logs
```

