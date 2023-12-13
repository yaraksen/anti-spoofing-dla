# ASR project
### Aksenov Yaroslav

## Installation guide

Download [model](https://disk.yandex.ru/d/KG1T9gH7GKDrxg) to the folder ```final_model```

Create folder ```data```, path to the dataset should be ```data/datasets/librispeech```

```shell
pip install -r ./requirements.txt
```

## Launching guide

#### Testing:
   ```shell
   python test.py \
      -c final_model/default_test_config.json \
      -r final_model/model_best.pth \
      -t test_data \
      -o test_result.json
   ```

#### Training step 1:
   ```shell
   python train.py \
      -c hw_asr/train460_config.json \
      -wk "YOUR_WANDB_API_KEY"
   ```

#### Training step 2:
   ```shell
   python train.py \
      -c hw_asr/train500_config.json \
      -wk "YOUR_WANDB_API_KEY" \
      -p "PATH TO THE LATEST CHECKPOINT FROM TRAIN 1"
   ```

#### Results for LibriSpeech test-clean, test-other are available for [pruned](pruned_res.json) and [original](not_pruned_res.json) language models.
