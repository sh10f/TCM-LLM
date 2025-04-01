import json

import tqdm
from pymilvus import MilvusClient
import numpy as np
from pymilvus.milvus_client import milvus_client

import tqdm
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks



client = MilvusClient("./milvus_demo.db")
collection_name = "QA_demp"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    dimension=768  # The vectors we will use has 768 dimensions because of GTE-Embedding model
)




model_id = "./chatLLM/GTE-Embedding"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id,
                       sequence_length=512
                       ) # sequence_length 代表最大文本长度，默认值为128
print("========= Embedding Model initialized =======")

'''
{'text_embedding': array([[-9.9107623e-03,  1.3627578e-03, -2.1072682e-02, ...,
         2.6786461e-02,  3.5029035e-03, -1.5877936e-02],
       [ 1.9877627e-03,  2.2191243e-02, -2.7656069e-02, ...,
         2.2540951e-02,  2.1780970e-02, -3.0861111e-02],
       [ 3.8688166e-05,  1.3409532e-02, -2.9691193e-02, ...,
         2.9900728e-02,  2.1570563e-02, -2.0719109e-02],
       [ 1.4484422e-03,  8.5943500e-03, -1.6661938e-02, ...,
         2.0832840e-02,  2.3828523e-02, -1.1581291e-02]], dtype=float32), 'scores': []}
'''

with open("./dataset/CMB/merge.json", "r", encoding="UTF-8") as f:
    data = json.load(f)[:10000]
    print("data: ", len(data))


for i, item in enumerate(tqdm.tqdm(data, desc="Embedding and Inserting ....")):
    inputs = {"source_sentence": [item["question"]]}
    result = pipeline_se(input=inputs)["text_embedding"][0]

    item["vector"] = result.tolist()
    item["id"] = i
    client.insert(collection_name=collection_name, data=item)

question = "具有益气祛风，健脾利水的方剂是什么"
search_res = client.search(
    collection_name=collection_name,
    data=[
        pipeline_se({"source_sentence": ["具有益气祛风，健脾利水的方剂是什么"]})["text_embedding"][0].tolist()
    ],
    limit=3,  # Return top 3 results
    search_params={"metric_type": "COSINE", "params": {}},  # Inner product distance
    output_fields=["question", "option", "answer", "analysis"],  # Return the text field
)

print(question, "\n")
retrieved_lines_with_distances = [
    (res["entity"]["question"], res["distance"]) for res in search_res[0]
]

print(json.dumps(retrieved_lines_with_distances, ensure_ascii=False, indent=4))








