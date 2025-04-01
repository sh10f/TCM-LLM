import json
import tqdm
import os

dataset_path = "../dataset"


# test dataset
# answer and question merge
with open("../dataset/CMB/CMB-test/CMB-test-choice-question-merge.json", "r", encoding="utf-8") as f:
    question = json.load(f)
    print("test question: ", len(question))

with open("../dataset/CMB/CMB-test/CMB-test-choice-answer.json", "r", encoding="utf-8") as f:
    answer = json.load(f)
    print("test answer: ", len(answer))

for i, q in enumerate(tqdm.tqdm(question, desc="Test QA Merging")):
    q["answer"] = answer[i]["answer"]

with open("../dataset/CMB/CMB-test/test_merged.json", "w", encoding="utf-8") as f:
    json.dump(question, f, indent=4, ensure_ascii=False)


# train test val merge
with open("../dataset/CMB/CMB-test/test_merged.json", "r", encoding="utf-8") as f:
    test = json.load(f)
    print("test : ", len(test))

with open("../dataset/CMB/CMB-train/CMB-train-merge.json", "r", encoding="utf-8") as f:
    train = json.load(f)
    print("train : ", len(train))

with open("../dataset/CMB/CMB-val/CMB-val-merge.json", "r", encoding="utf-8") as f:
    val = json.load(f)
    print("val : ", len(val))

with open("../dataset/CMB/CMB.json", "w", encoding="utf-8") as f:
    merged_data = []
    for i, sample in enumerate(tqdm.tqdm(test, desc="test merge")):
        data = {"question": sample["question"],
                "option": str(sample["option"]),
                "answer": sample["answer"],
                "analysis": sample.get("explanation", "")}
        merged_data.append(data)

    for i, sample in enumerate(tqdm.tqdm(train, desc="train merge")):
        data = {"question": sample["question"],
                "option": str(sample["option"]),
                "answer": sample["answer"],
                "analysis": sample.get("explanation", "")}
        merged_data.append(data)

    for i, sample in enumerate(tqdm.tqdm(val, desc="val merge")):
        data = {"question": sample["question"],
                "option": str(sample["option"]),
                "answer": sample["answer"],
                "analysis": sample.get("explanation", "")}
        merged_data.append(data)

    json.dump(merged_data, f, ensure_ascii=False, indent=4)





