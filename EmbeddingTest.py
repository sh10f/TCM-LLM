from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "./chatLLM/GTE-Embedding"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id,
                       sequence_length=512
                       ) # sequence_length 代表最大文本长度，默认值为128

# 当输入包含“soure_sentence”与“sentences_to_compare”时，会输出source_sentence中首个句子与sentences_to_compare中每个句子的向量表示，以及source_sentence中首个句子与sentences_to_compare中每个句子的相似度。
inputs = {
        "source_sentence": ["吃完海鲜可以喝牛奶吗?"],
        "sentences_to_compare": [
            "不可以，早晨喝牛奶不科学",
            "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
            "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
            "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷"
        ]
    }

result = pipeline_se(input=inputs)
print (result)
'''
{'text_embedding': array([[ 1.6415151e-04,  2.2334497e-02, -2.4202393e-02, ...,
         2.7710509e-02,  2.5980933e-02, -3.1285528e-02],
       [-9.9107623e-03,  1.3627578e-03, -2.1072682e-02, ...,
         2.6786461e-02,  3.5029035e-03, -1.5877936e-02],
       [ 1.9877627e-03,  2.2191243e-02, -2.7656069e-02, ...,
         2.2540951e-02,  2.1780970e-02, -3.0861111e-02],
       [ 3.8688166e-05,  1.3409532e-02, -2.9691193e-02, ...,
         2.9900728e-02,  2.1570563e-02, -2.0719109e-02],
       [ 1.4484422e-03,  8.5943500e-03, -1.6661938e-02, ...,
         2.0832840e-02,  2.3828523e-02, -1.1581291e-02]], dtype=float32), 'scores': [0.8859604597091675, 0.9830712080001831, 0.966042160987854, 0.891857922077179]}
'''


# 当输入仅含有soure_sentence时，会输出source_sentence中每个句子的向量表示。
inputs2 = {
        "source_sentence": [
            "不可以，早晨喝牛奶不科学",
            "吃了海鲜后是不能再喝牛奶的，因为牛奶中含得有维生素C，如果海鲜喝牛奶一起服用会对人体造成一定的伤害",
            "吃海鲜是不能同时喝牛奶吃水果，这个至少间隔6小时以上才可以。",
            "吃海鲜是不可以吃柠檬的因为其中的维生素C会和海鲜中的矿物质形成砷"
        ]
}
result = pipeline_se(input=inputs2)
print (result)
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