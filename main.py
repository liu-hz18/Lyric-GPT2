import os
import gc
import sys
import json
import jsonlines
import random
import itertools
import logging
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import mauve
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, GPT2LMHeadModel, GPT2Config, TextGenerationPipeline,
    get_linear_schedule_with_warmup,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)



def parse_args():
    parser = argparse.ArgumentParser(description="GPT Lyrics Model")
    parser.add_argument("--seed", type=int, default=23333333)
    # datasets
    parser.add_argument("--train", type=str, default="./data/lyric_train_clean.json")
    parser.add_argument("--valid", type=str, default="./data/lyric_valid.json")
    parser.add_argument("--test",  type=str, default="./data/lyric_test.json" )
    # checkpoints and logs
    parser.add_argument("--basedir", type=str, default="./log")
    parser.add_argument("--expname", type=str, default="tunegpt2")
    parser.add_argument("--save_freq", type=int, default=1)
    # model
    parser.add_argument("--no_pretrain", action="store_true")
    # candidates: ["uer/gpt2-chinese-cluecorpussmall", "uer/gpt2-distil-chinese-cluecorpussmall"]
    parser.add_argument("--model", type=str, default="uer/gpt2-chinese-cluecorpussmall", 
        choices=["uer/gpt2-chinese-cluecorpussmall", "uer/gpt2-distil-chinese-cluecorpussmall"])
    # training options
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=6000)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_batchsize", type=int, default=32)
    parser.add_argument("--mauve_batchsize", type=int, default=4)
    parser.add_argument("--maxlen", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--validation_after", type=int, default=0)
    parser.add_argument("--mauve_model", type=str, default="gpt2-medium",
        choices=["gpt2-medium", "uer/gpt2-chinese-lyric", "uer/gpt2-chinese-cluecorpussmall"])
    # testing options
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--savefile", type=str, default="generated_texts.txt")
    parser.add_argument("--plotfile0", type=str, default="divergence_curve.png")
    parser.add_argument("--plotfile1", type=str, default="quantized_version.png")
    # inference options
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--examples", type=str, default="./examples.txt")
    parser.add_argument("--inference_savefile", type=str, default="./inference_texts.txt")
    # generation options
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=40)
    parser.add_argument("--topp", type=float, default=0.9)   
    parser.add_argument("--rep_penalty", type=float, default=1.5)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2)
    parser.add_argument("--max_generate_idx", type=int, default=13349)
    args = parser.parse_args()
    return args


def init_logger(logdir):
    logger = logging.getLogger("default")
    cmd_handler = logging.StreamHandler(sys.stdout)
    cmd_handler.setLevel(logging.DEBUG)
    cmd_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    log_handler = logging.FileHandler(os.path.join(logdir, "train.log"), mode="w+", encoding="utf-8")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter(r"[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)s] %(message)s"))
    logger.addHandler(cmd_handler)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    return logger


def post_process(text: str):
    text = text.replace(" ", "")
    text = "".join([k for k, g in itertools.groupby(text)]).rstrip("，")
    return "，".join(text.split("，")[:-1])


def post_process_strict(text: str):
    return text.replace(" ", "")


class LyricDataset(Dataset):
    """
    retain_leading: whether only remain the leading few characters of each sequence
    """
    def __init__(self, file, retain_leading=False, min_leading_length=20):
        super(LyricDataset, self).__init__()
        self.raw_texts = []
        self.retain_leading = retain_leading
        self.min_leading_length = min_leading_length
        self.load_data(file)

    def load_data(self, file):
        with jsonlines.open(file, mode="r") as f:
            for line in f:
                if self.retain_leading:
                    texts = line["text"].strip().split("。")
                    if len(texts[0]) < self.min_leading_length:
                        text = texts[0] + "。" + texts[1].split("，")[0] + "，"
                    else:
                        subtexts = texts[0].split("，")
                        text = subtexts[0] + "，"
                        i = 1
                        while len(text) < self.min_leading_length:
                            text += subtexts[i] + "，"
                            i += 1
                else:
                    text = line["text"].strip()
                self.raw_texts.append("[CLS]" + text)

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        return self.raw_texts[idx]

    def examples(self, n=10):
        idxs = np.random.choice(list(range(len(self.raw_texts))), size=n, replace=False)
        return [self.raw_texts[idx] for idx in idxs]

    def all(self):
        return self.raw_texts


def train_epoch(args, model: GPT2LMHeadModel, tokenizer: BertTokenizer, optimizer, scheduler, dataloader: DataLoader):
    total_train_loss_value = 0.0
    train_loss_value = 0.0
    accumulation_count = 0
    tqdm_vars = {
        "lr": np.nan,
        "loss": np.nan,
        "norm": np.nan,
    }
    tbar = tqdm(enumerate(dataloader, start=1), desc="train", total=len(dataloader), postfix=tqdm_vars)
    gc.collect()
    torch.cuda.empty_cache()
    model.train()
    optimizer.zero_grad()
    for i, seqs in tbar:
        # for tokenizer options, see 
        # https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertTokenizer
        # https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizer
        batch_tokenized = tokenizer.batch_encode_plus(
            seqs,
            add_special_tokens=False,
            truncation=True,
            max_length=args.maxlen,
            padding="longest",
            return_offsets_mapping=False,
            return_token_type_ids=False,
            return_attention_mask=True,
        )
        # copy tensors to gpu
        input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
        # logger.info(f"texts: {seqs}\ninput_ids: {input_ids}\nattention_mask: {attention_mask}")
        # logger.info(f"input_ids shape: {input_ids.shape}")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        train_loss_value += outputs.loss.item()
        accumulation_count += 1
        total_train_loss_value += outputs.loss.item()
        loss = outputs.loss / args.gradient_accumulation_steps
        loss.backward()
        if accumulation_count % args.gradient_accumulation_steps == 0:
            norm = clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm, norm_type=2).item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tqdm_vars["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
            tqdm_vars["loss"] = train_loss_value / args.gradient_accumulation_steps
            tqdm_vars["norm"] = norm
            tbar.set_postfix(tqdm_vars)
            accumulation_count = 0
            train_loss_value = 0.0
    return total_train_loss_value / len(dataloader)


def valid_epoch(args, model: GPT2LMHeadModel, tokenizer: BertTokenizer, dataloader):
    valid_loss_value = 0.0
    valid_ppl_value = 0.0
    valid_mauve_value = 0.0
    generated_texts = []
    human_texts = []
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[VALID] validating...")
    model.eval()
    with torch.no_grad():
        for i, seqs in tqdm(enumerate(dataloader, start=1), desc="valid", total=len(dataloader)):
            batch_tokenized = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=False,
                truncation=True,
                max_length=args.maxlen,
                padding="longest",
                return_offsets_mapping=False,
                return_token_type_ids=False,
                return_attention_mask=True,
            )
            # copy tensors to gpu
            input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
            attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
            # forwarding
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            valid_loss_value += outputs.loss.item()
            pred_tokens = torch.argmax(outputs.logits, dim=-1)
            batch_texts = tokenizer.batch_decode(
                sequences=pred_tokens,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            _seqs = [seq.lstrip("[CLS]") for seq in seqs]
            batch_texts = [post_process_strict(text)[:len(seq)] for text, seq in zip(batch_texts, _seqs)]
            generated_texts.extend(batch_texts)
            human_texts.extend(_seqs)
            # logger.info(f'gt: {_seqs}')
            # logger.info(f'gen: {batch_texts}')
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[VALID] computing mauve...")
    mauve_outputs = mauve.compute_mauve(
        p_text=human_texts, q_text=generated_texts, device_id=0, 
        max_text_length=args.maxlen, verbose=False, batch_size=args.mauve_batchsize,
        featurize_model_name=args.mauve_model,
    )
    valid_mauve_value = mauve_outputs.mauve
    valid_loss_value = valid_loss_value / len(dataloader)
    valid_ppl_value = np.exp(valid_loss_value)
    return valid_loss_value, valid_ppl_value, valid_mauve_value


def test_epoch(args, model: GPT2LMHeadModel, tokenizer: BertTokenizer, dataloader, leading_dataset):
    test_loss_value = 0.0
    test_ppl_value = 0.0
    human_texts = []
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[TEST] testing...")
    model.eval()
    with torch.no_grad():
        for i, seqs in tqdm(enumerate(dataloader, start=1), desc="test ppl", total=len(dataloader)):
            batch_tokenized = tokenizer.batch_encode_plus(
                seqs,
                add_special_tokens=False,
                truncation=True,
                max_length=args.maxlen,
                padding="longest",
                return_offsets_mapping=False,
                return_token_type_ids=False,
                return_attention_mask=True,
            )
            # copy tensors to gpu
            input_ids = torch.tensor(batch_tokenized["input_ids"], device=device)
            attention_mask = torch.tensor(batch_tokenized["attention_mask"], device=device)
            # forwarding
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            test_loss_value += outputs.loss.item()
            human_texts.extend([seq.lstrip("[CLS]") for seq in seqs])
    test_loss_value = test_loss_value / len(dataloader)
    test_ppl_value = np.exp(test_loss_value)
    logger.info(f"[TEST] loss={test_loss_value}, ppl={test_ppl_value}")
    # generate texts
    text_generator = TextGenerationPipeline(model, tokenizer, device=0, batch_size=2 * args.eval_batchsize // args.num_beams)
    text_generator.model.config.pad_token_id = text_generator.model.config.eos_token_id
    gen_texts = []
    gc.collect()
    torch.cuda.empty_cache()
    with open(args.savefile, "w+", encoding="utf-8") as f:
        with torch.no_grad():
            logger.info(f"Generating sequences on [TEST](leading) dataset...")
            # for more parameters, see 
            # https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/text_generation#transformers.GenerationMixin.generate
            # https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline
            outputs = text_generator(
                leading_dataset.all(),
                return_tensors=False, 
                return_text=True,
                clean_up_tokenization_spaces=True,
                do_sample=True, # Whether or not to use sampling; use greedy decoding otherwise.
                max_length=args.maxlen,
                num_beams=args.num_beams,  # Number of beams for beam search. 1 means no beam search.
                temperature=args.temperature, #  The value used to module the next token probabilities.
                repetition_penalty=args.rep_penalty, # The parameter for repetition penalty. 1.0 means no penalty. (default 1.0)
                length_penalty=args.length_penalty, # Exponential penalty to the length that is used with beam-based generation. 
                # It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
                
                top_k=args.topk, # The number of highest probability vocabulary tokens to keep for top-k-filtering.(default 50)
                top_p=args.topp, # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
            )
            for output in outputs:
                idx = 0
                for i, indice in enumerate(output[0]["generated_token_ids"][1:args.maxlen+1]):
                    if indice > args.max_generate_idx:
                        idx = i
                        break
                text = tokenizer.decode(
                    output[0]["generated_token_ids"][1:idx],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                text = post_process(text)
                f.write(text + "\n")
                gen_texts.append(text)
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("[TEST] computing mauve...")
    mauve_outputs = mauve.compute_mauve(
        p_text=human_texts, q_text=gen_texts, device_id=0,
        max_text_length=args.maxlen, verbose=False, batch_size=args.mauve_batchsize,
        featurize_model_name=args.mauve_model,
    )
    test_mauve_value = mauve_outputs.mauve
    # Plot the divergence curve
    plt.figure()
    plt.plot(mauve_outputs.divergence_curve[:, 0], mauve_outputs.divergence_curve[:, 1])
    plt.savefig(args.plotfile0)
    # visualize quantized versions of P and Q
    plt.figure()
    idxs = np.argsort(mauve_outputs.p_hist)[:: -1]
    sample_p = np.random.multinomial(n=1000, pvals=mauve_outputs.p_hist[idxs])
    sample_q = np.random.multinomial(n=1000, pvals=mauve_outputs.q_hist[idxs])
    x = np.arange(mauve_outputs.p_hist.shape[0])
    plt.bar(x, sample_p , color="blue", alpha =0.3, label="P(human)")
    plt.bar(x, sample_q , color="red", alpha =0.3, label="Q(model)")
    plt.legend()
    plt.savefig(args.plotfile1)
    return test_loss_value, test_ppl_value, test_mauve_value


def generate(args, model: GPT2LMHeadModel, tokenizer: BertTokenizer):
    texts = []
    with open(args.examples, "r", encoding="utf-8") as f:
        for line in f.readlines():
            texts.append("[CLS]" + line.strip())
    gc.collect()
    torch.cuda.empty_cache()
    # generate texts
    text_generator = TextGenerationPipeline(model, tokenizer, device=0, batch_size=2 * args.eval_batchsize // args.num_beams)
    text_generator.model.config.pad_token_id = text_generator.model.config.eos_token_id
    model.eval()
    logger.info("Generating")
    with open(args.inference_savefile, "w+", encoding="utf-8") as f:
        with torch.no_grad():
            logger.info(f"Generating sequences on [{args.examples}] dataset...")
            outputs = text_generator(
                texts,
                return_tensors=False, 
                return_text=True, 
                clean_up_tokenization_spaces=True,
                do_sample=True, # Whether or not to use sampling; use greedy decoding otherwise.
                max_length=args.maxlen,
                num_beams=args.num_beams,  # Number of beams for beam search. 1 means no beam search.
                temperature=args.temperature, #  The value used to module the next token probabilities.
                repetition_penalty=args.rep_penalty, # The parameter for repetition penalty. 1.0 means no penalty. (default 1.0)
                length_penalty=args.length_penalty, # Exponential penalty to the length that is used with beam-based generation. 
                # It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.
                
                top_k=args.topk, # The number of highest probability vocabulary tokens to keep for top-k-filtering.(default 50)
                top_p=args.topp, # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                early_stopping=True,
            )
            for output in outputs:
                idx = 0
                for i, indice in enumerate(output[0]["generated_token_ids"][1:args.maxlen+1]):
                    if indice > args.max_generate_idx:
                        idx = i
                        break
                text = tokenizer.decode(
                    output[0]["generated_token_ids"][1:idx],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                text = post_process(text)
                f.write(text + "\n")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CACHE_DIR = "./models"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    # make log dirs
    os.makedirs(args.basedir, exist_ok=True)
    logdir = os.path.join(args.basedir, args.expname)
    os.makedirs(logdir, exist_ok=True)
    # build logger
    logger = init_logger(logdir)
    # change save files' path to dir
    args.savefile = os.path.join(logdir, args.savefile)
    args.plotfile0 = os.path.join(logdir, args.plotfile0)
    args.plotfile1 = os.path.join(logdir, args.plotfile1)
    args.inference_savefile = os.path.join(logdir, args.inference_savefile)
    # save configs
    with open(os.path.join(logdir, "config.json"), "w+", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Building datasets")
    # build dataset
    train_dataset = LyricDataset(args.train)
    valid_dataset = LyricDataset(args.valid)
    test_dataset  = LyricDataset(args.test)
    test_leading_dataset = LyricDataset(args.test, retain_leading=True, min_leading_length=20)
    # logging some samples
    logger.info(f"Training samples: {train_dataset.examples(3)}")
    logger.info(f"Validating samples: {valid_dataset.examples(3)}")
    logger.info(f"Testing(full) samples: {test_dataset.examples(3)}")
    logger.info(f"Testing(leading) samples: {test_leading_dataset.examples(3)}")
    # build dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batchsize, shuffle=False, num_workers=args.num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=args.eval_batchsize, shuffle=False, num_workers=args.num_workers)
    # build model
    logger.info("Building model")
    tokenizer = BertTokenizer.from_pretrained(args.model, unk_token="[UNK]", cache_dir=MODEL_CACHE_DIR)
    if args.no_pretrain:
        config = GPT2Config.from_pretrained(args.model)
        model = GPT2LMHeadModel(config).to(device)
        assert args.lr > 1e-4, "[WARN] use bare model without pretrained params, max lr should be larger than 1e-4."
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model, cache_dir=MODEL_CACHE_DIR).to(device)
    logger.info(f"[MODEL] prefix: {model.config.prefix}")
    logger.info(f"[MODEL] pad_id: {model.config.pad_token_id}")
    logger.info(f"[MODEL] sep_id: {model.config.sep_token_id}")
    logger.info(f"[MODEL] eos_id: {model.config.eos_token_id}")
    logger.info(f"[MODEL] bos_id: {model.config.bos_token_id}")
    logger.info(f"[MODEL] decoder_start_token_id: {model.config.decoder_start_token_id}")
    logger.info(f"[MODEL] configs: {model.config}")
    # build optimizer
    logger.info("Building optimizer")
    num_training_steps = args.epoch * len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)
    
    # training & validating
    if not args.test_only and not args.inference_only:
        begin_epoch = 0
        if args.checkpoint is not None:
            logger.info(f"Load params from checkpoint: {args.checkpoint}")
            params = torch.load(args.checkpoint)
            model.load_state_dict(params["model"], strict=True)
            optimizer.load_state_dict(params["optimizer"])
            scheduler.load_state_dict(params["scheduler"])
            begin_epoch = params["epoch"]
        logger.info("Start training and validating")
        for i in range(begin_epoch+1, args.epoch+1):
            logger.info(f"[EPOCH] {i}")
            train_loss = train_epoch(args, model, tokenizer, optimizer, scheduler, train_dataloader)
            logger.info(f"[TRAIN] loss={train_loss}")
            # save checkpoints
            if i % args.save_freq == 0:
                torch.save({
                    "epoch": i,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, os.path.join(logdir, f"{i}.pth"))
            # validating
            if i > args.validation_after:
                valid_loss, valid_ppl, valid_mauve = valid_epoch(args, model, tokenizer, valid_dataloader)
                logger.info(f"[VALID] loss={valid_loss}, ppl={valid_ppl}, mauve={valid_mauve}")
    else:
        assert args.checkpoint is not None, "args.checkpoint should not be `None`"
        logger.info(f"Load params from checkpoint: {args.checkpoint}")
        params = torch.load(args.checkpoint)
        model.load_state_dict(params["model"], strict=True)

    # generating
    if not args.test_only:
        assert args.examples is not None, "args.examples should not be `None`"
        assert args.inference_savefile is not None, "args.inference_savefile should not be `None`"
        generate(args, model, tokenizer)

    # testing
    if not args.inference_only:
        test_loss, test_ppl, test_mauve = test_epoch(args, model, tokenizer, test_dataloader, test_leading_dataset)
        logger.info(f"[TEST] loss={test_loss}, ppl={test_ppl}, mauve={test_mauve}")
