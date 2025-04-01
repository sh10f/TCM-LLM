import json
import tqdm
import os

import pandas as pd

dataset_path = "../dataset"

train = pd.read_csv("../dataset/CMExam/data/train.csv")
test = pd.read_csv("../dataset/CMExam/data/test_with_annotations.csv")
val = pd.read_csv("../dataset/CMExam/data/val.csv")

with open("../dataset/CMExam/CMExam.json", "w", encoding="utf-8") as f:
    merged_data = []
    for i, sample in enumerate(tqdm.tqdm(test.itertuples(), desc="test merge")):
        data = {"question": sample.Question,
                "option": sample.Options,
                "answer": sample.Answer,
                "analysis": sample.Explanation}
        merged_data.append(data)

    for i, sample in enumerate(tqdm.tqdm(train.itertuples(), desc="train merge")):
        data = {"question": sample.Question,
                "option": sample.Options,
                "answer": sample.Answer,
                "analysis": sample.Explanation}
        merged_data.append(data)

    for i, sample in enumerate(tqdm.tqdm(val.itertuples(), desc="val merge")):
        data = {"question": sample.Question,
                "option": sample.Options,
                "answer": sample.Answer,
                "analysis": sample.Explanation}
        merged_data.append(data)

    json.dump(merged_data, f, ensure_ascii=False, indent=4)





