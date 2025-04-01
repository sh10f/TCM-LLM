import json
import tqdm
import os

import pandas as pd

dataset_path = "../dataset"

choice_herb = pd.read_csv("../dataset/TCMChat/single/choice_herb_500.csv")
choice_formula = pd.read_csv("../dataset/TCMChat/single/choice_formula_500.csv")


with open("../dataset/TCMChat/TCMChat.json", "w", encoding="utf-8") as f:
    merged_data = []
    for i, sample in enumerate(tqdm.tqdm(choice_herb.itertuples(), desc="choice_herb merge")):
        options = ("A. " + sample.A + " \nB. " + sample.B + " \nC. " + sample.C + " \nD. " + sample.D + " \nE. " + sample.E)
        data = {"question": sample.question,
                "option": options,
                "answer": sample.answer,
                "analysis": sample.parse}
        merged_data.append(data)

    for i, sample in enumerate(tqdm.tqdm(choice_formula.itertuples(), desc="choice_formula merge")):
        options = ("A. " + sample.A + " \nB. " + sample.B + " \nC. " + sample.C + " \nD. " + sample.D + " \nE. " + sample.E)
        data = {"question": sample.question,
                "option": options,
                "answer": sample.answer,
                "analysis": sample.parse}
        merged_data.append(data)


    with open("../dataset/TCMChat/medical_case/medical_case.json", "r", encoding="utf-8") as file:
        medical_data = json.load(file)

    for i, sample in enumerate(tqdm.tqdm(medical_data, desc="medical_case merge")):
        data = {"question": sample["input"],
                "option": "",
                "answer": sample["syndrome"],
                "analysis": sample["output"]}
        merged_data.append(data)

    json.dump(merged_data, f, ensure_ascii=False, indent=4)





