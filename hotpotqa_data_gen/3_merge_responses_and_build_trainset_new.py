from time import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os, json, argparse, re, random
from collections import defaultdict


prompt_for_reso = """Given the context and question, answer the question through step-by-step reasoning based on the context.
---
Context: {reference_docs}

Question: {question}
---

Answer: """



prompt_for_judge = """Judge the correctness of the answer in the following Q&A scenario:
###
Given the context and question, answer the question through step-by-step reasoning based on the context.
---
Context: {reference_docs}

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
            # fout中每行写入序列化后的json对象
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("file saved.")
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mistral_data_path",type=str, help="datasets generated from Mistral")
    parser.add_argument("--qwen_data_path",type=str, help="datasets generated from Qwen")
    parser.add_argument("--baichuan2_data_path",type=str, help="datasets generated from Baichuan2")
    parser.add_argument("--save_dir",type=str, help="")
    args = parser.parse_args()
    
    mistral_data = read_jsonl(args.mistral_data_path)
    qwen_data = read_jsonl(args.qwen_data_path)
    baichuan2_data = read_jsonl(args.baichuan2_data_path)

    print(len(mistral_data))
    print(len(qwen_data))
    print(len(baichuan2_data))

    trainset_reso = []
    trainset_eval = []

    qids = set()
    
    # responses
    cnt1, cnt0 = 0, 0
    for d1, d2, d3 in tqdm(zip(mistral_data, qwen_data, baichuan2_data)):
        qid = d1["qid"]
        qids.add(qid)
        context = d1["instruction"]
        question = d1["input"]
        answer = d1["output"]
        responses = d1["responses"] + d2["responses"] + d3["responses"]
        judges = d1["judges"] + d2["judges"] + d3["judges"]
        for r,j in zip(responses, judges):
            if j==0: #* wrong model responses, convert to eval-format
                instruction = prompt_for_judge.format(reference_docs=context, question=question, response=r)
                output = "[[N]], the answer is incorrect."
                cnt0 += 1
                trainset_eval.append({
                    'qid': qid,
                    'context': context,
                    'question': question,
                    'ans': answer,
                    'instruction': instruction,
                    'input': '',
                    'output': output,
                    'judge': j,
                    'response': r,
                })
            else: #* correct model responses, convert to eval- or reso-format
                if random.uniform(0,1)<=0.5: 
                    #* eval-format
                    cnt1 += 1
                    instruction = prompt_for_judge.format(reference_docs=context, question=question, response=r)
                    output = "[[Y]], the answer is correct."
                    trainset_eval.append({
                        'qid': qid,
                        'context': context,
                        'question': question,
                        'ans': answer,
                        'instruction': instruction,
                        'input': '',
                        'output': output,
                        'judge': j,
                        'response': r,
                    })
                else: #* reso-format
                    instruction = prompt_for_reso.format(reference_docs=context, question=question)
                    trainset_reso.append({
                        'qid': qid,
                        'context': context,
                        'question': question,
                        'ans': answer,
                        'instruction': instruction,
                        'input': '',
                        'output': r,
                        'judge': j,
                        'response': r,
                    })
    
    print(f"{len(trainset_reso)} samples for reasoning, {len(trainset_eval)} samples for eval;\n {cnt1} correct reasoning, {cnt0} incorrect reasoning.")
    
    random.seed(2024)
    random.shuffle(trainset_reso)
    random.shuffle(trainset_eval)
    num_sampled = 10000
    #* sample some questions for gathering the corresponding evals
    sampled_qids = set(random.sample(list(qids), num_sampled))
    num_responses = 1
    cnter1 = defaultdict(int)
    cnter0 = defaultdict(int)
    sampled_trainset_eval = []
    for d in tqdm(trainset_eval):
        qid = d['qid']
        if qid in sampled_qids:
            if d['judge']==1 and cnter1[qid]<num_responses: #* correct
                cnter1[qid] += 1
                sampled_trainset_eval.append(d)     
            elif d['judge']==0 and cnter0[qid]<num_responses: #* incorrect
                cnter0[qid] += 1
                sampled_trainset_eval.append(d)
    stats = 0
    threshold = 1
    for qid in sampled_qids:
        if cnter1[qid]>=threshold and cnter0[qid]>=threshold: stats += 1
    print(f"After sampling, {len(sampled_trainset_eval)} samples for eval, including {sum(cnter1.values())} correct reasonings and {sum(cnter0.values())} incorrect reasonings\n{stats} questions contains at least {threshold} correct and incorrect answers")
    
    sampled_trainset_reso = []
    visited_qid = set()
    cnt = 0
    for d in tqdm(trainset_reso):
        qid = d['qid']
        if qid not in sampled_qids and qid not in visited_qid:
            visited_qid.add(qid)
            sampled_trainset_reso.append(d)
            cnt += 1
        if cnt==len(sampled_trainset_eval): break
    
    print(f"After sampling,  {len(sampled_trainset_reso)} samples for reasoning")
        
    # print(f"Some examples for reso+eval:",random.sample(sampled_trainset_eval, 3))
    for d in random.sample(sampled_trainset_eval, 5):
        instruction = d["instruction"]
        output = d["output"]
        print(f"instruction: {instruction}\noutput: {output}", end="\n\n\n\n*******************\n\n")
    
    final_data = sampled_trainset_eval + sampled_trainset_reso
    random.shuffle(final_data)
    # save_dataset(args.save_dir + f"/hotpotqa_qaj_num_{num_responses}_train.json", trainset)
    save_dataset(args.save_dir + f"/hotpotqa_qaj_num_{num_responses}_trainSampled_new.json", final_data)






