import jsonlines

texts = []

with jsonlines.open("./data/lyric_train.json", mode="r") as f:
    for line in f:
        text = line["text"].strip()
        if "录音" in text:
            continue
        if "合声" in text:
            continue
        if "演唱" in text:
            continue
        if "词曲" in text:
            continue
        if "原唱" in text:
            continue
        if "吉他" in text:
            continue
        text.replace("\"", "")
        text.replace("\'", "")
        texts.append({"text": text})

with jsonlines.open("./data/lyric_train_clean.json", mode="w") as f:
    f.write_all(texts)
