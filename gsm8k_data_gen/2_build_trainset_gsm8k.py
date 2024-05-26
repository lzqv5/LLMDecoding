from time import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os, json, argparse, re, random
from collections import defaultdict

prompt_for_reso = """Given the following math question, answer it through step-by-step reasoning.
---
Question: {question}
---

Answer: """



prompt_for_judge = """Judge the correctness of the answer in the following Q&A scenario:
###
Given the following math question, answer it through step-by-step reasoning.
---
Question: {question}
---

Answer: {response}
###

Judge: """



def read_jsonl(file_path):
    return [json.loads(jsonline) for jsonline in open(file_path, 'r')] 



def save_dataset(path, parsed_dataset):
    with open(path, "w") as fout:
        for example in tqdm(parsed_dataset):
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("file saved.")
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #* fields must be included: question, model_generate, judge
    #* from (question, model_generate, judge) build training data ...
    parser.add_argument("--data_path", default='data_path', type=str, help="data dir") #* e.g. /path/to/your/gsm8k/train.jsonl
    parser.add_argument("--save_dir", default='save_dir', type=str, help="save dir")
    parser.add_argument("--data_name", default='data_name', type=str, help="save name")
    args = parser.parse_args()
    
    raw_data = read_jsonl(args.data_path)

    print(f'{len(raw_data)} samples to be processed...')

    trainset_reso = [] #* format 1: (q, a)
    trainset_eval = [] #* format 2: (q, a, j)

    qids = set()
    
    # responses
    cnt1, cnt0 = 0, 0
    # for d1, d2, d3 in tqdm(zip(mistral_data, qwen_data, baichuan2_data)):
    for d in raw_data:
        question = d["question"]
        model_generate = d["model_generate"]
        judge = d["judge"]
        if judge == 1: # 此条数据中的模型响应是正确的
            cnt1 += 1
            judge_output = "[[Y]], the answer is correct."
            trainset_reso.append({
                'instruction': prompt_for_reso.format(question=question),
                'input': '',
                'output': model_generate,
            })
        else: 
            cnt0 += 1
            judge_output = "[[N]], the answer is incorrect."
        trainset_eval.append({
                'instruction': prompt_for_judge.format(question=question, response=model_generate),
                'input': '',
                'output': judge_output,
                'judge': judge,
            })
        
    
    print(f"{len(trainset_reso)} samples for reasoning, {len(trainset_eval)} samples for eval;\n # right reasoning {cnt1}, wrong reasoning {cnt0} 个")
   
    save_dataset(f"{args.save_dir}/{args.data_name}_reso.json", trainset_reso)
    save_dataset(f"{args.save_dir}/{args.data_name}_eval.json", trainset_eval)
    

