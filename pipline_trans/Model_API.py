# --*-- conding:utf-8 --*--
# @Time : 2024/1/11 17:21
# @Author : YWJ
# @Email : 52215901025@stu.ecnu.edu.cn
# @File : Model_API.py
# @Software : PyCharm
# @Description :  各个模型的API接口
import os

# import openai
# from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from pymilvus import MilvusClient
import numpy as np

import tqdm
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks







from searchTest import chinese_crawler


os.environ["OPENAI_API_KEY"] = 'sk-20a8638044224dc89470bce7b13b57e6d75d90909c99ec82c234fdc46e1887b5'


class API():
    def __init__(self, api_key_list: str, model_name: str = "gpt-3.5-turbo", peer_name: str = "Qwen2.5-1.5B-Instruct",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 cache_dir: str = "../chatLLM",
                 peer_dir: str = "../chatLLM"):
        self.api_key_list = api_key_list
        self.model_name = model_name  # 新的model, 支持1w+
        self.peer_name = peer_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # self.llm = LLM("/data/home-old/mguan/MyProject/ChatWK-main/Qwen-merge-lora/pt-chat-1-9-lora",tokenizer_mode='auto',
        #       trust_remote_code=True,
        #       enforce_eager=True,
        #       enable_prefix_caching=True)
        self.cache_dir = cache_dir
        self.peer_dir = peer_dir
        self.client = MilvusClient("../db/milvus_demo.db")
        self.collection_name = "QA_demp"
        # self.client.load_collection(self.collection_name)
        print("Collection initialized")

        model_id = "../chatLLM/GTE-Embedding"
        self.pipeline_se = pipeline(Tasks.sentence_embedding,
                               model=model_id,
                               sequence_length=512
                               )  # sequence_length 代表最大文本长度，默认值为128
        print("========= Embedding Model initialized =======")

        # self.sampling_params = SamplingParams(temperature=0.85, top_p=0.8, max_tokens=512)
        print("+++++++++++++++++++++")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=self.cache_dir)
        print("Tokenizer initialized")

        # self.llm = LLM("Qwen/Qwen2.5-1.5B-Instruct", tokenizer_mode='auto',
        #       trust_remote_code=True,
        #       enforce_eager=True,
        #       enable_prefix_caching=True)

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=self.cache_dir
        )
        print("LLM init finished.")

        # self.peer_llm = AutoModelForCausalLM.from_pretrained(
        #     self.peer_name,
        #     torch_dtype="auto",
        #     device_map="auto",
        #     cache_dir=self.peer_dir
        # )
        self.peer_llm = self.llm
        self.peer_tokenizer = self.tokenizer
        # self.peer_tokenizer = AutoTokenizer.from_pretrained(self.peer_name, trust_remote_code=True,
        #                                                     cache_dir=self.peer_dir)
        print("peer LLM and Tokenizer init finished.")

    # GPT系列

    def get_turbo_Webinfo(self, problem):
        # problem = "治疗肺痈的要药是（  ）。\nA．野菊花\nB．金银花\nC．蒲公英\nD．鱼腥草\nE．败酱草"
        question = [problem]
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[
                self.pipeline_se({"source_sentence": question})["text_embedding"][0].tolist()
            ],
            limit=3,  # Return top 3 results
            search_params={"metric_type": "COSINE", "params": {}},  # Inner product distance
            output_fields=["question", "option", "answer", "analysis"],  # Return the text field
        )

        print(problem, "\n")
        retrieved_lines_with_distances = [
            {"question": res["entity"]["question"],
             "option": res["entity"]["option"],
             "answer": res["entity"]["answer"],
             "analysis": res["entity"]["analysis"],
             "distance": res["distance"]} for res in search_res[0]
        ]

        Webdata = ""
        for res in retrieved_lines_with_distances:
            Webdata += str(res) + "\n"
        print("Webdata is: ", Webdata)

        # Webdata = str(retrieved_lines_with_distances)

        prompt = f"请你提取出对解决问题有用的信息。医疗问题如下\n：{problem}\n 相关文本集合如下: {Webdata}\n"

        systemPrompt = f"""
        ### 你是一个专注中医领域的智能筛选系统，具备以下核心能力：从相关文本集合中识别关键医学要素；过滤重复/低质/无关内容。
        ### 当给定「医疗问题」和「相关文本集合」时：
        # 1.信息存在多重冗余（重复方剂/相似病例/矛盾结论）
        # 3.与问题无关的文本信息
        ### 请你提取出对解题有帮助的信息，若没有，则返回你对题目中一些关键词的想法，但是不要给出你的答案，
        # 针对文本集合中有帮助的信息，应包括问题，答案在option中对应的内容，不是全部选项，和分析。应该简洁。
        ### 处理后以字符串的形式返回信息提取的结果，结果以‘[相关信息]’开头。
        ### 医疗问题和文本数据如下：
        """
        messages = [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt}
        ]
        text = self.peer_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.peer_tokenizer([text], return_tensors="pt").to(self.peer_llm.device)

        generated_ids = self.peer_llm.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.peer_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print("---peer LLN result: -----\n", response)
        return response

    def send_request_NLI(self, prompt, premise, hypothesis):
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}

        webInfo = " "
        messages = [zero_shot_prompt_message]
        # messages = []
        question = f"输入：\n示例如下:\n{webInfo}\n问题的前提和假设如下：\npremise：{premise}\nhypothesis:{hypothesis}\n输出：\n"
        message = {"role": "user", "content": question}
        messages.append(message)
        print("user mesages:", message["content"])
        output = {}
        model_output = ""
        try:
            if self.model_name != "":
                model_output = self.qwen15_14b_chat_api(messages)
        except Exception as e:
            print('Exception:', e)
        return model_output

    def send_request_turbo(self, prompt, question):
        """
        """

        zero_shot_prompt_message = {'role': 'system', 'content': prompt}

        webInfo = self.get_turbo_Webinfo(question)
        messages = [zero_shot_prompt_message]
        # messages = []
        question = f"输入：\n相关信息如下:\n{webInfo}\n问题是：\n{question}输出：\n"
        message = {"role": "user", "content": question}
        messages.append(message)
        print("user mesages:", message["content"])
        output = {}
        model_output = ""
        try:
            if self.model_name != "":
                model_output = self.qwen15_14b_chat_api(messages)
        except Exception as e:
            print('Exception:', e)
        return model_output

    # 多轮会话
    def send_request_chat(self, prompt, share_content, questions, question_type="A3+A4"):
        """
        reference_lists: 里面的列表元素是每个问题对应的参考信息列表
        title_lists
        """
        if question_type == "A3+A4":
            question_chose = "请根据相关的知识和案例的内容，选出唯一一个正确的选项\n"
        else:
            question_chose = "请根据相关的知识，选出在共享答案中唯一一个正确的选项\n"
        zero_shot_prompt_message = {'role': 'system', 'content': prompt}
        messages = [zero_shot_prompt_message]
        # messages = []
        i = 0
        if question_type == "A3+A4":
            messages.append({
                'role': 'user',
                'content': f"案例是：{share_content}"
            })
        else:
            messages.append({
                'role': 'user',
                'content': f"{share_content}"
            })
            # share_content_prompt = f"{share_content}"
        model_output_list = []
        while i < len(questions):
            sub_question = questions[i]['sub_question']

            if question_type == "A3+A4":
                webInfo = self.get_turbo_Webinfo(share_content+sub_question)
            else:
                webInfo = self.get_turbo_Webinfo(sub_question + share_content)

            question = f"输入：\n相关信息如下:\n{webInfo}\n问题是：\n问题{sub_question}{question_chose}输出：\n"
            message = {"role": "user", "content": question}
            messages.append(message)

            print("question message: ", message["content"])
            model_output = ""
            try:
                if self.model_name != "":
                    model_output = self.qwen15_14b_chat_api(messages)
            except Exception as e:
                print('Exception:', e)
            messages.append({
                "role": "assistant",
                "content": model_output
            })
            print("output \n", model_output)
            model_output_list.append(model_output)
            i += 1
        # print(model_output)
        return model_output_list

    def qwen15_14b_chat_api(self, messages):
        model_output = ""
        cnt = 0
        while (cnt < 10):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            # outputs = self.llm.generate(text, self.sampling_params)
            t_outputs = self.llm.generate(**model_inputs, max_new_tokens=self.max_tokens)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, t_outputs)
            ]

            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # for output in outputs:
            #     # model_output = output.outputs[0].text.split("<|endoftext|>")[0]
            #     model_output = output.outputs[0].text
            model_output = outputs
            if model_output != "" and model_output != "答案：":
                break
            cnt += 1
            print('*' * 50, cnt, '*' * 50, f"Generated text: {model_output!r}")
        return model_output
