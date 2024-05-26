import argparse
import csv
import glob
import json
import os
import random
import datetime
from vllm_generate import vllm_generate

def clean_numbers(numbers):
    cleaned = [re.sub(r'\.$', '', re.sub(r'[^\d\.]', '', num)) for num in numbers]
    return cleaned

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

#& This prompt is only for display purposes. 
#& We will randomly sample instances as 'demonstrations' mentioned in the paper during actual use.
def wrapGSMQuery(data):
    question = data['question']
    mathPrompt = f'''please help me solve the math problem step by step. You must give your final answer at the end of response like 'So, the final answer is $ your answer'. I will provide an example for you:
---
Question: Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?
Answer: Step1: To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds.\nStep2: Next, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\nStep3: And finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.\nSo, the final answer is $16
---

Next, please answer this math problem: 
Question:{question}
Answer:
'''
    return mathPrompt


def load_gsm8k_csv(data_path):
    dataset = []
    with open (data_path, "r") as f:
        dataset = json.load(f)

    return dataset
def save_gsm8k_csv(args, data):
    with open (f"{args.save_path}.json", "w") as f:
        json.dump(data, f, indent=2)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="the path of model used for generating response")
    parser.add_argument("--data_path", type=str, help="the path of dataset") #* e.g. /path/to/your/gsm8k/train.jsonl
    parser.add_argument("--model_name", type=str, default="gsm8k_gen_model", help="the name of model, output_dir = ./output/save_name")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--seed", type=int, default=123456)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()
    t = datetime.datetime.now()
    args.time_str = f"{t.month:02d}-{t.day:02d}_{t.hour:02d}:{t.minute:02d}"
    args.save_path = os.path.join(args.output_path, f"{args.model_name}__{args.time_str}")
    random.seed(args.seed)
    return args

import re

def fs_cothub_gsm8k_match_answer(answer, response):
    # CoT hub match answer for GSM8k, match last numeric value and compare with the expected answer
    # Patterns to find numeric values after specific markers in both answer and response
    pattern1 = '####.*?(\d*\.?\d+)'
    pattern2 = 'So, the final answer is \$\s*(\d+\.?\d*)'
    pa = 'The answer is: \s*(\d+\.?\d*)'
    # Extract numbers from the answer and response
    ans_numbers = re.findall(pattern1, answer)
    pred_numbers = re.findall(pattern2, response)
    if len(pred_numbers) == 0:
        pred_numbers = re.findall(pa, response)
    pred_numbers = clean_numbers(pred_numbers)
    print(pred_numbers)
    # Check if both answer and response contain numbers and compare the last found number
    if ans_numbers and pred_numbers:
        # Compare the last number found in both answer and response
        if ans_numbers[-1] == pred_numbers[-1]:
            return 1, pred_numbers[-1]  # Match found, return 1 and the matching number
        else:
            return 0, pred_numbers[-1]  # Numbers do not match, return 0 and the last number from the response

    return 0, "No match or missing numbers"  # No numbers found or other issues

if __name__ == "__main__":
    args = parse_args()
    dataset = load_gsm8k_csv(args.data_path)
    dataset = dataset
    n = len(dataset)  # the number of question
    output_data = []
    prompts = []
    for i, data in enumerate(dataset):
        prompt = wrapGSMQuery(data)
        prompts.append(prompt)
        output_data.append({
            "question": data['question'],
            "prompt": prompt,
            "answer":data['answer'],
            "model_generate": "",
            "judge":"",
        })

    #* generate by vllm
    generated_texts = vllm_generate(prompts, args, args.model_name)
    cnt = 0
    #* save questions and inference result
    for i in range(n):
        generated_text = generated_texts[i]
        output_data[i]["model_generate"] = generated_text
        judge, ans = fs_cothub_gsm8k_match_answer(output_data[i]['answer'], output_data[i]["model_generate"])
        output_data[i]["judge"] = judge
        pattern1 = '####.*?(\d*\.?\d+)'
        ans_numbers = re.findall(pattern1, output_data[i]["answer"])
        output_data[i]["origi_result"] = ans_numbers
        output_data[i]["model_result"] = ans
        if judge == 1:
            cnt = cnt + 1
    #* save the output...
    #* three key fields: 1. question, 2. model_generate, 3. judge
    print(f"the right answer count down to {cnt}\n")
    save_gsm8k_csv(args,output_data)
