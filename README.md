# You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling

Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling attention mechanism based on Locality Sensitive Hash- ing (LSH), decreases the quadratic complexity of such models to linear. We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant). This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of LSH (to enable deployment on GPU architectures).

## Requirements

```
docker, nvidia-docker
```

## Start Docker Container

Under `YOSO` folder, run

```
docker run --ipc=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<Device IDs> -v "$PWD:/workspace" -it mlpen/transformers:4
```

For Nvidia's 30 series GPU, run

```
docker run --ipc=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<Device IDs> -v "$PWD:/workspace" -it mlpen/transformers:5
```

Then, the `YOSO` folder is mapped to `/workspace` in the container.

## BERT

### Datasets

To be updated

### Pre-training

To start pre-training of a specific configuration: create a folder `YOSO/BERT/models/<model>` (for example, `bert-small`) and write `YOSO/BERT/models/<model>/config.json` to specify model and training configuration, then under `YOSO/BERT` folder, run
```
python3 run_pretrain.py --model <model>
```
The command will create a `YOSO/BERT/models/<model>/model` folder holding all checkpoints and log file.

### Pre-training from Different Model's Checkpoint

Copy a checkpoint (one of `.model` or `.cp` file) from `YOSO/BERT/models/<diff_model>/model` folder to `YOSO/BERT/models/<model>` folder and add a key-value pair in `YOSO/BERT/models/<model>/config.json`: `"from_cp": "<checkpoint_file>"`. One example is shown in `bert-small-4096/config.json`. This procedure also works for extending the max sequence length of a model (For example, use `bert-small` pre-trained weights as initialization for `bert-small-4096`).

### GLUE Fine-tuning

Under `YOSO/BERT` folder, run
```
python3 run_glue.py --model <model> --batch_size <batch_size> --lr <learning_rate> --task <task> --checkpoint <checkpoint>
```
For example,
```
python3 run_glue.py --model bert-small --batch_size 32 --lr 3e-5 --task MRPC --checkpoint cp-0249.model
```
The command will create a log file in `YOSO/BERT/models/<model>/model`.

## Long Range Arena Benchmark

### Datasets

To be updated

### Run Evaluations

To start evaluation of a specific model on a task in LRA benchmark:

- Create a folder `YOSO/LRA/models/<model>` (for example, `softmax`)
- Write `YOSO/LRA/models/<model>/config.json` to specify model and training configuration

Under `YOSO/LRA` folder, run
```
python3 run_task.py --model <model> --task <task>
```
For example, run
```
python3 run_task.py --model softmax --task listops
```

The command will create a `YOSO/LRA/models/<model>/model` folder holding the best validation checkpoint and log file. After completion, the test set accuracy can be found in the last line of the log file.

## RoBERTa

### Datasets

The pre-training dataset consists of English Wikipedia and BookCorpus. All downloaded data files should be placed in the corresponding folder under `data-preprocessing`. The original format of English Wikipedia dump is preprocessed using [wikiextractor](https://github.com/attardi/wikiextractor), and the resulting files are placed in `data-preprocessing/wiki`. Then, run `data-preprocessing/<dataset>/preprocess.py` under each corresponding folder to generate data files of unified format. After preprocessing, run `data-preprocessing/preprocess_data_<length>.py` to generate pre-training data of specific sequence length.

For GLUE datasets, download the datasets and place them under `glue` folder

### Pre-training

To start pretraining of a specific configuration:

- Create a folder `YOSO/RoBERTa/models/<model>` (for example, `bert-small`)
- Write `YOSO/RoBERTa/models/<model>/config.json` to specify model and training configuration

Under `YOSO/RoBERTa` folder, run
```
python3 run_pretrain.py --model <model>
```
For example, run
```
python3 run_pretrain.py --model bert-small
```

The command will create a `YOSO/RoBERTa/models/<model>/model` folder holding all checkpoints and log file.

### GLUE Fine-tuning

To fine-tune model on GLUE tasks:

Under `YOSO/RoBERTa` folder, run
```
python3 run_glue.py --model <model> --batch_size <batch_size> --lr <learning_rate> --task <task> --checkpoint <checkpoint>
```
For example,
```
python3 run_glue.py --model bert-small --batch_size 32 --lr 3e-5 --task MRPC --checkpoint 249
```

The command will create a log file in `YOSO/RoBERTa/models/<model>/model`.

## Efficiency


## Citation
```
@article{zeng2021yoso,
  title={You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling},
  author={Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2021}
}
```
