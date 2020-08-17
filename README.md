# CDial-GPT
* This project provides a large-scale cleaned Chinese **conversation dataset** and a **Chinese GPT model** pre-trained on this dataset. Please refer to our [paper](https://arxiv.org/abs/2008.03946) for more details.

* Our code used for the pre-training is adapted from the [TransferTransfo](https://github.com/huggingface/transfer-learning-conv-ai) model based on the [Transformers](https://github.com/huggingface/transformers) library. The codes used for both pre-training and fine-tuning are provided in this repository.

* 本项目提供了一个大规模的经过系统清洗的中文对话数据集，并提供在此数据集上的对话预训练模型，更多信息可参考我们的[论文](https://arxiv.org/abs/2008.03946)。

* 本项目代码修改自[TransferTransfo](https://github.com/huggingface/transfer-learning-conv-ai)，使用了HuggingFace Pytorch版的[Transformers](https://github.com/huggingface/transformers)库, 可用于预训练与微调。

## Contents
* <a href="#Dataset">Dataset</a>
* <a href="#Pre-training">Pre-training</a>
* <a href="#Evaluation">Evaluation</a>

## <a name="#Dataset">Dataset</a>
We present a Large-scale Cleaned Chinese Conversation corpus (LCCC) containing:
[LCCC-base](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCC-base.zip) and
[LCCC-large](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCC-large.zip).
A rigorous data cleaning pipeline is designed to ensure the quality of the corpus.
This pipeline involves a set of rules and several classifier-based filters.
Noises such as offensive or sensitive words, special symbols, emojis, grammatically incorrect sentences, and incoherent conversations are filtered.

The statistic of our corpus is presented below.
Dialogues with only two utterances are regarded as "Single-turn", and dialogues with more than three utterances are regarded as "Multi-turn".
The vocabulary size is calculated in word-level, and [Jieba](https://github.com/fxsjy/jieba) is used to tokenize each utterance to words.

| [LCCC-base](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCC-base.zip) | Single-turn | Multi-turn  |
| :----------------------------------------------------------- | :--------- | :---------  |
| Sessions                                                     |  3,354,382 |  3,466,607  |
| Utterances                                                   |  6,708,554 | 13,365,268  |
| Characters                                                   | 68,559,727 | 163,690,614 |
| Vocabulary                                                   |   372,063  |   666,931   |
| Avg. words per utterance                                     |    6.79    |    8.32     |
| Avg. utterances per session                                  |      2     |    3.86     |

Note that LCCC-base is cleaned using more strict rules compared to LCCC-large. A [tokenized version of LCCC-base](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCC-base_splited.zip) is also provided.

| [LCCC-large](https://coai-dataset.oss-cn-beijing.aliyuncs.com/LCCC-large.zip) | Single-turn | Multi-turn  |
| :----------------------------------------------------------- | :---------  | :---------  |
| Sessions                                                     |  7,273,804  |  4,733,955  |
| Utterances                                                   | 14,547,608  | 18,341,167  |
| Characters                                                   | 162,301,556 | 217,776,649 |
| Vocabulary                                                   |   662,514   |   690,027   |
| Avg. words per utterance                                     |    7.45     |    8.14     |
| Avg. utterances per session                                  |      2      |    3.87     |

The raw dialogues for LCCC-base originate from a Weibo Corpus that we crawled from [Weibo](www.weibo.com), and the raw dialogues for LCCC-large is built by combining several conversation datasets in addition to the Weibo Corpus:

| Dataset                              | Sessions  | Sample                               |
| :---------------------------------- | :-------: | :---------------------------------- |
| Weibo Corpus                      | 79M | Q:火锅我在重庆成都吃了七八顿火锅 A: 哈哈哈哈！那我的嘴巴 可能要烂掉！ |
| [PTT Gossiping Corpus](https://github.com/zake7749/Gossiping-Chinese-Corpus) | 0.4M | Q:为什么乡民总是欺负国高中生呢QQ A:如果以为选好科系就会变成比尔盖兹那不如退学吧 |
| [Subtitle Corpus](https://github.com/skdjfla/dgk_lost_conv) | 2.74M | Q:京戏里头的人都是不自由的 A:他们让人拿笼子给套起来了了 |
| [Xiaohuangji Corpus](https://github.com/skdjfla/dgk_lost_conv) | 0.45M | Q:你谈过恋爱么 A:谈过，哎，别提了，伤心.. |
| [Tieba Corpus](https://github.com/codemayq/chinese_chatbot_corpus) | 2.32M | Q:前排，鲁迷们都起床了吧 A:标题说助攻，但是看了那球，真是活生生的讽刺了 |
| [Qingyun Corpus](https://github.com/codemayq/chinese_chatbot_corpus) | 0.1M | Q:看来你很爱钱 A:噢是吗？那么你也差不多了 |
| [Douban Conversation Corpus](https://github.com/MarkWuNLP/MultiTurnResponseSelection) | 0.5M | Q:看原版英文电影学纯正英语 A:大爱老友记反复看了好多次 了 Q:一样光盘都快被我看花了 A:那你现在的英语应该不错了 |
| [E-commerical Conversation Corpus](https://github.com/cooelf/DeepUtteranceAggregation) | 0.5M | Q:这个会不会聚划算 A:暂时没有哦 Q:后期会不会有 A:不一定哦亲多多关注我们哦 |
| [Chinese Chat Corpus](https://github.com/yangjianxin1/GPT2-chitchat) | 0.5M | Q: 我今天腿都废了，你们过节，我搬砖 A: 辛苦啊，圣诞节还去赚大钱了加油 Q: 毕竟是没男朋友的人，什么节都是一样的 |

## <a name="#Pre-training">Pre-training</a>
### Models  
We also present a series of Chinese GPT model that are first pre-trained on a Chinese novel dataset and then post-trained on our LCCC dataset.

Similar to [TransferTransfo](https://arxiv.org/abs/1901.08149), we concatenate all dialogue histories into one context sentence, and use this sentence to predict the response. The input of our model consists of word embedding, speaker embedding, and positional embedding of each word.

![Input representation](figures/inputs.png)

| Models        | Parameter Size | Pre-training Dataset   | Description                                       |
|---------------| ------ |--------------------------|-------------------------------------------------- |
| [GPT<sub>Novel</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/GPT_Novel.zip) | 95.5M | Chinese Novel            | A GPT model pre-trained on Chinese Novel dataset (1.3B words).(Our code does not support this model).  |
| [CDial-GPT<sub>LCCC-base</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/GPT_LCCC-base.zip)  | 95.5M | [LCCC-base](##datasets)  | A GPT model post-trained on LCCC-base dataset from GPT<sub>Novel</sub> |
| [CDial-GPT2<sub>LCCC-base</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/GPT2_LCCC-base.zip) | 95.5M | [LCCC-base](##datasets)  | A GPT2 model post-trained on LCCC-base dataset from GPT<sub>Novel</sub> |
| [CDial-GPT<sub>LCCC-large</sub>](https://coai-dataset.oss-cn-beijing.aliyuncs.com/GPT_LCCC-large.zip) | 95.5M | [LCCC-large](##datasets) | A GPT model post-trained on LCCC-large dataset from GPT<sub>Novel</sub> |

### Installation  
Install from the source codes:

    git clone https://github.com/lemon234071/GPT-Chinese.git
    cd GPT-Chinese
    pip install -r requirements.txt 
    
### Quick Start
Step 1: Prepare the data for fine-tuning (E.g., [STC dataset](https://arxiv.org/abs/1503.02364) or "data/toy_data.json" in our respository) and the pre-trianed model:
    
    wget https://coai-dataset.oss-cn-beijing.aliyuncs.com/STC-corpus.zip # Download the STC dataset and unzip into "data_path" dir (fine-tuning on STC)
    wget https://coai-dataset.oss-cn-beijing.aliyuncs.com/GPT_LCCC-large.zip # Download the GPT<sub>LCCC-large</sub> weights file and unzip into "model_checkpoint" dir
Note: If the computer's memory is insufficient, you can process the file into txt format, use the "train_path"
 to load data in a distributed manner (the function we adapted from Internet), and you need to leave "data_path" empty. 
  
Step 2: Train the model

    python train.py --pretrained --model_checkpoint ./models/ --data_path data/STC.json  # Single GPU training
    python -m torch.distributed.launch --nproc_per_node=8 train.py --pretrained --model_checkpoint ./models/ --data_path data/STC.json  # Training on 8 GPUs

Step 3: Inference mode

    python infer.py --model_checkpoint ./models/ --datapath data/STC_test.json --out_path STC_result.txt  # Do Inference on a corpus
    python interact.py --model_checkpoint ./models/  # Interact on the terminal

Training Arguments

| Arguments  | Type     | Default value  | Description |
| :---- | :---------- | :----- | :------- |
| model_checkpoint | str | "" | Path or URL of model files (Directory of pre-training model and config/vocab files) |
| pretrained  | bool | False | If False, then train the model from scratch |
| data_path | str | "" | Path of the dataset |
| dataset_cache | str | default="dataset_cache" | Path or url of the dataset cache |
| train_path | str | "" | Path of the training set for distributed dataset |
| valid_path | str | "" | Path of the validation set for distributed dataset |
| log_file | str | "" | Output logs to a file under this path |
| num_workers | int | 1 | Number of subprocesses for data loading |
| n_epochs | int | 70 | Number of training epochs |
| train_batch_size | int | 8 | Batch size for training |
| valid_batch_size | int | 8 | Batch size for validation |
| max_history | int | 15 | Number of previous exchanges to keep in history |
| scheduler | str | "noam" | Method of optimizer |
| n_emd | int | 768 | Number of n_emd in config file (for noam) |
| eval_before_start | bool | False | If true, start evaluation before training |
| warmup_steps | int | 5000 | Warm up steps |
| valid_steps | int | 0 | Perform validation every X steps, if is not 0 |
| gradient_accumulation_steps | int | 64 | Accumulate gradients on several steps |
| max_norm | float | 1.0 | Clipping gradient norm |
| device | str | "cuda" if torch.cuda.is_available() else "cpu" | Device (cuda or cpu) |
| fp16 | str | "" | Set to O0, O1, O2 or O3 for fp16 training (see apex documentation) |
| local_rank | int | -1 | Local rank for distributed training (-1: not distributed) |

## <a name="#Evaluation">Evaluation</a> 
Evaluation is performed on results generated by models fine-tuned on
[STC dataset](https://coai-dataset.oss-cn-beijing.aliyuncs.com/STC-corpus.zip).
All responses are generated using the [Nucleus Sampling](https://arxiv.org/abs/1904.09751) scheme with a threshold 0.9 and temperature 0.7.

#### Automatic Evaluation

| Models  | Model Size | PPL  | BLEU-2 | BLEU-4 | Dist-1 | Dist-2 | Greedy Matching | Embedding Average |
| :------ | :------: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| Attn-Seq2seq | 73M | 34.20 | 3.93 | 0.90 | 8.5 | 11.91 | 65.84 | 83.38 |
| Transformer | 113M | 22.10 | 6.72 | 3.14 | 8.8 | 13.97 | 66.06 | 83.55 |
| GPT2-chitchat | 88M | - | 2.28 | 0.54 | 10.3 | 16.25 | 61.54 | 78.94 |
| GPT<sub>Novel</sub> | 95.5M | 21.27 | 5.96 | 2.71 | 8.0 | 11.72 | 66.12 | 83.34 |
| GPT<sub>LCCC-base</sub> | 95.5M | 18.38 | 6.48 | 3.08 | 8.3 | 12.68 | 66.21 | 83.54 |
| GPT2<sub>LCCC-base</sub> | 95.5M | 22.76 | 5.69 | 2.50 | 7.7 | 10.87 | 66.24 | 83.46 |
| GPT<sub>LCCC-large</sub> | 95.5M | 18.23 | 6.63 | 3.20 | 8.3 | 12.71 | 66.25 | 83.63 |

#### Human Evaluation

We also employed 3 crowd-sourced annotators to evaluate 200 samples of each model about fluency, relevance, and diversity based on a 2/1/0 score schema. Refer to our [paper](https://arxiv.org/abs/2008.03946) for more details.

| Models  | +2      | +1      | +0      | Kappa   |
| :----- | :-----: | :-----: | :-----: | :-----: |
| Attn-Seq2Seq | 27.1% | 21.4% | 51.4% | 0.4544 |
| Transformer | 42.4% | 23.6% | 34.0% | 0.4429 |
| GPT2-chitchat | 24.3% | 19,1% | 56.6% | 0.3941 |
| GPT<sub>LCCC-base</sub> | 46.7% | 21.8% | 31.5% | 0.3954 |
| GPT<sub>LCCC-large</sub> | 48.6% | 24.5% | 27.0% | 0.4203 |

#### Toxic Generation Statistics

We randomly select 1000 generated samples, and make a statistic of toxic generation:

| Repetition | Dirty word | Other Bad Cases |
| :----------: | :-----: | :-------: |
| 23 | 9 | 2 |

## Case study of GPT<sub>LCCC-large</sub>
Single-turn generated examples from STC dataset (fine-tuned on STC):

| Post | Response |
| :----- | :---------- |
| 这 居 然 是 北 京 不 是 哈 尔 滨 。 | 哈 尔 滨 的 天 气 好 像 比 北 京 好 点 ， 我 这 里 就 是 小 雨 ， 白 天 都 是 大 风 |
| 运 营 商 校 园 广 告 大 P K ， 太 欢 乐 了 ！ 哈 哈 哈 。 | 我 以 为 是 马 云 的 广 告 。 |
| D e l s a S p o s a 糖 果 色 系 列 婚 纱 ， 粉 蓝 红 紫 ， 俏 皮 又 清 新 ， 你 喜 欢 吗 ？ | 最 后 一 件 太 美 了 ， 美 的 不 像 话 ！ |

Multi-turn generated examples using the interactive script:

| Roel | Response |
| :----- | :---------- |
| User | 别爱我没结果 |
| Bot | 没 结 果 的 爱 是 多 么 的 痛 。 |
| User | 但是爱过就可以了 |
| Bot | 有 时 候 ，爱 情 就 是 一 场 游 戏 。 |
| User | 输的总是我 |
| Bot | 我 爱 你 ， 就 像 我 爱 你 一 样 。 |
| User | 你可拉到吧 |
| Bot | 你 可 拉 到 吧 。 |
| User | 别学我说话 |
| Bot | 我 说 话 就 是 你 不 行 。 |
| User | 逼我动手是吧 |
| Bot | 不 ， 是 逼 你 动 口 是 吧 |
 
[//]: #  (Multi-turn self-playing dialogue with user prompt:)

[//]: #  (Commonsense question answering examples:)

## Authors
[Yida Wang](https://github.com/lemon234071), [Pei Ke](https://github.com/kepei1106), [Yinhe Zheng](https://github.com/silverriver), [Kaili Huang](https://github.com/KellyHuang63), Yong Jiang, Xiaoyan Zhu, Minlie Huang

## Disclaimer
The LCCC dataset and the pre-trained models aim to facilitate the research for conversation generation. The LCCC dataset provided in this repository originates from various sources. Although a rigorous cleaning process has been carried out, there is no guarantee that all the inappropriate contents have been completely filtered out. All the contents contained in this dataset does not represent the authors' opinion.
This repository contains only part of the modeling machinery needed to actually produce a dialogue model. The decoding script provided in this repository is only for the research purpose. We are not responsible for any contents generated using our model.

## Citation
Please kindly cite our [paper](https://arxiv.org/abs/2008.03946) if you use the datasets or models in your research:

    @inproceedings{wang2020chinese,
      title={A Large-Scale Chinese Short-Text Conversation Dataset},
      author={Wang, Yida and Ke, Pei and Zheng, Yinhe and Huang, Kaili and Jiang, Yong and Zhu, Xiaoyan and Huang, Minlie},
      booktitle={NLPCC},
      year={2020},
      url={https://arxiv.org/abs/2008.03946}
    }
