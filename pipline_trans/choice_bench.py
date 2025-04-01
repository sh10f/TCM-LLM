import os
import sys


parent_path = os.path.dirname(sys.path[0])
print(parent_path)
if parent_path not in sys.path:
    sys.path.append(parent_path)


# from LLAMAAPI import LlamaAPI
from Model_API import API
from bench_function import export_distribute_json, export_union_json
import json
import argparse

os.environ["OPENAI_API_KEY"] = 'sk-20a8638044224dc89470bce7b13b57e6d75d90909c99ec82c234fdc46e1887b5'
def parse_args():
    parser = argparse.ArgumentParser(description="parameter of TCM_EDU_LLM")
    parser.add_argument(
        "--data_path",
        type=str,
        default='data/A_evaluate/evaluate',
        help="测试数据",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='qwen',
        help="The LLM name.",
    )
    parser.add_argument(
        "--sys_prompt",
        type=str,
        default='A1-2_prompt.json',
        help="选择不同测试题类型的指令.",
    )
    parser.add_argument(
        "--start_num",
        type=int,
        default=0,
        help="保存文档的起始id",
    )
    args = parser.parse_args()
    return args

# 测试主函数, 一个问题+一个答案 --》 bench_function

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 仅暴露前两个GPU
    os.environ["OPENAI_API_KEY"] = 'sk-20a8638044224dc89470bce7b13b57e6d75d90909c99ec82c234fdc46e1887b5'

    args = parse_args()
    args.model_name = '../chatLLM/Qwen2.5-7B'
    args.sys_prompt = 'A1-2_prompt.json'
    # args.model_name = '../chatLLM/DeepSeek-R1-8B'  #'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    # args.model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
    args.peer_name = '../chatLLM/Qwen2.5-7B' #"../chatLLM/Qwen2.5-1.5B"
    args.data_path = "../dataset/B_test/"

    with open(f"{args.sys_prompt}", "r", encoding="utf-8") as f:
        data = json.load(f)['examples']
    print("data: ", data)
    f.close()
    for i in range(len(data)):
        directory = args.data_path
        model_name = args.model_name
        peer_name = args.peer_name
        if model_name != "":
            print("model_name", model_name)
            api = API("", model_name=model_name, peer_name=peer_name,
                      cache_dir=model_name, peer_dir=peer_name)
        data[i]['keyword'] = "test_A12"
        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print("keyword: ", keyword)
        print("question_type: ", question_type)
        export_distribute_json(
            api,
            model_name,
            directory,
            keyword,
            zero_shot_prompt_text,
            question_type,
            args,
            parallel_num=1,
        )

        export_union_json(
            directory,
            model_name,
            keyword,
            zero_shot_prompt_text,
            question_type
        )
    

# export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# export JRE_HOME=${JAVA_HOME}/jre
# export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
# export PATH=${JAVA_HOME}/bin:$PATH

