# Anti-Spoofing project
### Aksenov Yaroslav

## Installation guide

Create folder ```data```, path to the dataset should be like ```data/LA/ASVspoof2019_LA_train```

```shell
mkdir final_model
yadisk https://disk.yandex.ru/d/PSwWLrafk2m6xg final_model
pip install -r ./requirements.txt
```

## Launching guide

#### Testing:
   ```shell
   python test.py \
      -c src/train_config.json \
      -ap test_audios \
      -r final_model/checkpoint-epoch90.pth \
      -o test_out.txt
   ```

#### Training:
   ```shell
   python train.py \
      -c src/train_config.json \
      -wk "YOUR_WANDB_API_KEY"
   ```

#### Results for LJSpeech samples, official test audios, my hifigan homework samples and some samples from the internet are available in [file](test_out.txt).
