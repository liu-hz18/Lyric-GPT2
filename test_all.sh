python main.py --test_only --topk 2 --num_beams 2 --no_repeat_ngram_size 1 --expname test-topk1 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
python main.py --test_only --topk 10 --expname test-topk2 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
python main.py --test_only --topp 0.1 --expname test-topp1 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
python main.py --test_only --topp 0.5 --expname test-topp2 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
python main.py --test_only --rep_penalty 0.1 --expname test-rep1 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
python main.py --test_only --rep_penalty 10.0 --expname test-rep2 --model uer/gpt2-chinese-cluecorpussmall --checkpoint ./log/tunegpt-finetune/10.pth --eval_batchsize 128 --mauve_batchsize 128
