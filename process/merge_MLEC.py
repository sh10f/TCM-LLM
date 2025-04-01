import json
import tqdm
import os

dataset_path = "../dataset"


# train test val merge
path_list = ["TCM_train", "TCM_test", "TCM_dev", "CWM_train", "CWM_test", "CWM_dev"]


merged_data = []
for path in path_list:
    with open("../dataset/MLEC-QA/" + path + ".json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        print(f"{path} : ", len(dataset))

        for i, sample in enumerate(tqdm.tqdm(dataset, desc=f"{path} merge")):
            data = {"question": sample["qtext"],
                    "option": str(sample["options"]),
                    "answer": sample["answer"],
                    "analysis": sample.get("explanation", "")}
            merged_data.append(data)

with open("../dataset/MLEC-QA/MLEC-QA.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)





