# plantPathology

## report
[plant pathology report](./report/projet.md)

## kaggle

- [1st kaggle](https://www.kaggle.com/guillaume74140/plant-pathology-2020-eda-models/edit?rvi=1)
- [2nd kaggle](https://www.kaggle.com/a03102030/plant-pathology-2020-resnet50)


## Authors
- BADIKIAN Guillaume
- CLEMENT Gauthier
- LOEW Benoît
- REVOUY Théo
- SANOU Abou

## Packages
- numpy
- tensorflow
- keras
- pandas
- efficientnet
- tqdm
- matplotlib
- scikit-learn
- plotly

### create virtual env

```shell script
python3 -m venv mon_venv
source mon_venv/bin/activate
```

### install package
```shell script
pip3 install my_package
```

### create your config
`config.yaml` into ./config/
```yaml
plantPathology:
  batch_size: 10 # size of 1 batch
  epoch: 20 # number of epoch
  gpu_devices: # which gpus to use
  - GPU:0
  - GPU:1
  - GPU:2
  - GPU:3
  history:
    test: ./result/history_test # file to save test history
    train: ./result/history_train # file to save train history
  image_path: ./data/images/ # folder which contains images
  model: resnet #model to use : resnet, denseNet or efficientNet
  n_test: 8 # numero of the test (auto-increment)
  step_per_epoch: 80 
  sub_path: ./data/sample_submission.csv 
  test_path: ./data/test.csv
  train_path: ./data/train.csv
  use: gpu #gpu or tpu

```


### run plant pathology
`myvenv` : folder of your venv : default /venv
```shell script
source run.sh myvenv
```
