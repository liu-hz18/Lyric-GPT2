# GPT2 歌词生成

## Preparation
```
pip install -r requirements.txt
```

## Dataset Preprocess
删去包含“录音”、“合声”、“演唱”等含歌词作者信息的数据。只保留只含歌词的数据。清洗后的数据在`./data/lyric_train_clean.json`。
```
python preprocess.py
```

## Train a GPT-2 from scratch
使用`--no_pretrain`参数。训练需要约11GB显存
```
python main.py --expname gpt-bare --no_pretrain --lr 1e-3 --epoch 20
```

## Finetune a pre-trained GPT-2
训练需要约11GB显存
```
python main.py --expname gpt-finetune --model uer/gpt2-chinese-cluecorpussmall --batchsize 16 --gradient_accumulation_steps 2 --eval_batchsize 32
```

## Finetune a pre-trained GPT-2-distil
训练需要约8GB显存
```
python main.py --expname gpt-distil-finetune --model uer/gpt2-distil-chinese-cluecorpussmall --batchsize 16 --gradient_accumulation_steps 2 --eval_batchsize 32
```

## Test on a checkpoint
使用 Test Dataset 计算PPL和MAUVE分数, 需要约8GB显存。
```
python main.py --test_only --topk 10 --expname gpt-test --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 32 --mauve_batchsize 32
```

## Inference on a checkpoint
基于给定的前缀文件`--examples <prefix_file>`续写歌词。需要约8GB显存。
```
python main.py --inference_only --examples ./examples.txt --topk 10 --expname gpt-inference --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 32 --mauve_batchsize 32
```
