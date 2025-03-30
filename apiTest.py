from serpapi import GoogleSearch


from transformers import AutoModelForCausalLM, AutoTokenizer


problem = "治疗肺痈的要药是（  ）。\nA．野菊花\nB．金银花\nC．蒲公英\nD．鱼腥草\nE．败酱草"
search = GoogleSearch({
    "q": problem,
    "location": "China",
    "api_key": "5ea7887de7fb0913e7b3b9da7b36f37b28aaa293497c3efa5658feb3fb09ef0a"
  })
result = search.get_dict()
print(result)

Webdata = ""
index = 1
for i in result['organic_results']:
    print(i['title'])
    print(i['snippet'])
    print("\n")

    Webdata += "信息"+str(index) + ":\n"
    Webdata += "title：\n" + i['title'] + "\n"
    Webdata += "具体信息:\n" +  i['snippet'] + "\n\n"


model_name = "Qwen/Qwen2.5-1.5B-Instruct"

cache_dir = "./chatLLM"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir

)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)



prompt = f"请你提取出对解决问题有用的信息。医疗问题如下\n：{problem}\n 信息如下: {Webdata}\n"

systemPrompt = f"""
### 你是一个专注中医领域的智能筛选系统，具备以下核心能力：从多源异构文本中识别关键医学要素；过滤重复/低质/无关内容（广告、无效案例等）。
### 当给定「大模型待解决的医疗问题」和「相关文本集合」时：
# 1.信息存在多重冗余（重复方剂/相似病例/矛盾结论）
# 2.包含非结构化数据（古籍摘录/现代论文/患者自述）
# 3.混杂商业推广等干扰信息
### 请你提取出对解题有帮助的信息

### 处理后以字符串的形式返回信息提取的结果，结果以‘【有效信息】’开头。
### 医疗问题和文本数据如下：
"""
messages = [
    {"role": "system", "content": systemPrompt},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


