* TCMBench
  * 知识理解(选择题)： 选择正确候选答案，并给出解析
  * 语义推理(判断)： 给定 前提和假设，判断是否 **蕴含**
* 两榜
  * 不微调模型
    * **Prompt Engineering**
    * 示例选择 --- **In-Context Learning**
    * **RAG**
    * 大小模型协同 --- **可微调小模型**
  * PEFT





# Day 1 -- 3.26

* Windows 本地安装 **Qwen2.5-1.5B-Instruct**
* 设置 ***角色配置 - 问题 - 问题说明 - 分析要求 - 回答要求*** 的提示词模板
  * Qwen2.5-1.5B-Instruct -- **A1/2 Test Accuracy: 0.66**
  * DeepSeek-R1-Distill-Llama-8B  -- **A1/2 前10题 Accuracy：0.3**
  * Qwen2.5-7B-Instruct -- **A1/2 Test Accuracy: 0.66**
  * 

{   "share_content": "患者，女，60岁，有反复尿路感染病史5年，3天前因劳累而复发，发症：小便淋沥不已，尿感湿痛，时作时止，伴腰膝酸软，神疲乏力，舌淡，脉细弱。",      "question": [        {          "sub_question": "1)．其辨证是（  ）。\nA．热淋\nB．血淋\nC．气淋\nD．膏淋\nE．劳淋\n",          "answer": [            ""          ],          "analysis": ""        },        {          "sub_question": "2)．其治法是（  ）。\nA．补气益血\nB．补脾益肾\nC．补益肝肾\nD．健脾补肺\nE．补肺益肾\n",          "answer": [            ""          ],          "analysis": ""        },        {          "sub_question": "3)．治疗应首选的方剂是（  ）。\nA．无比山药丸\nB．八珍汤\nC．河车大造丸\nD．金匮肾气丸\nE．附子理中丸\n",          "answer": [                 ],          "analysis": ""        }      ],      "index": 0 }



# Day 2 -- 3.27

* **Qwen2.5-1.5B-Instruct** 作为 *peer_model* 协作
  * ***Google Searpapi***
  * peerModel 提取、整理网页信息中对关于题目的信息
  * **A1/2 Test Accuracy: 0.75**
* Qwen2.5-1.5B-Instruct Prompt
  * **A1/2 Test Accuracy: 0.78**





# Day3 -- 3.30

* 4个开源数据集

  * CMB-Exam(24 **NAACL**): https://github.com/FreedomIntelligence/CMB

    * ```java
          {
              "exam_type": "专业知识考试",
              "exam_class": "中医学与中药学",
              "exam_subject": "中医学",
              "question": "肝主疏泄，主要表现在",
              "answer": "BCDE",
              "explanation": "肝主疏泄的表现：肝的疏泄功能反映了肝为刚脏....",
              "question_type": "多项选择题",
              "option": {
                  "A": "通调水道",
                  "B": "调畅气机",
                  "C": "助脾运化",
                  "D": "条达情志",
                  "E": "调节生殖功能"
              }
          }
      ```

      

  * TCMchat(24 **Pharmacological Research**):  https://github.com/ZJUFanLab/TCMChat

  * CMExam(23 **NeurlPS**): https://github.com/williamliujl/CMExam

  * MLEC-QA(21 **EMNLP**): https://github.com/Judenpech/MLEC-QA
