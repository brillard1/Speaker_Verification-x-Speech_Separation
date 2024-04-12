# SV -x- SS
The given repository contains two implementations

1. Speaker Verification (sv)
2. Speaker Separation (ss)

### Speaker Verification
___
To finetune or evaluate the models, run the below command
```
python main.py {model_name} {mode}
```
- The `model_name` is among wave2vec2_xlsr, hubert_large or wavlm_large. The checkpoints for these can be downloaded from the [link](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification)
- The `mode` can be test or train (for finetuning) depending upon the requirements

### Speaker Separation
___
Download the test-clean parition from the LibriSpeech dataset. Execute the `generate_librimix.sh` file to download the same.

To prepare the custom LibriMix dataset, specify the dataset paths inside the `prepare.py` file and run the following command
```
python prepare.py
```

Now for the speaker separation task, run the below command
```
python main.py {mode} {--load}
```
- Here `mode` refers to train or eval, use train for fine-tuning and eval for test results
- Specify `--load` parameter if model weights are present in 'ckpt/best.pt'
