from time import time
from vllm import LLM, SamplingParams
from tqdm import tqdm
import os, json, argparse, re, random


zero_shot_prompt = """You are professional at reading comprehension. 

The following is one or several reference documents for you to comprehend and memory:
---
{reference_docs}
---

The answer to the following question can be inferred through the above reference(s).
Question: {question}

Provide the answer to the given question through step-by-step reasoning. After your reasoning, output your final answer in the format of [[your answer]].
make sure the final answer enclosed by the double square brackets.

Let's think step by step. """

# QwenLM format
# https://github.com/QwenLM/Qwen/issues/257
# https://xbot123.com/645a461b922f176d7cfdbc2d/
# https://huggingface.co/Qwen/Qwen-7B-Chat/discussions/11
zero_shot_prompt_qwen = """<|im_start|>system
You are professional at reading comprehension.<|im_end|>
<|im_start|>user
The following is one or several reference documents for you to comprehend and memory:
---
{reference_docs}
---

The answer to the following question can be inferred through the above reference(s).
Question: {question}

Provide the answer to the given question through step-by-step reasoning. After your reasoning, output your final answer in the format of [[your answer]].
make sure the final answer enclosed by the double square brackets.

Let's think step by step.<|im_end|>
<|im_start|>assistant
"""

zero_shot_prompt_mistral = """[INST]You are professional at reading comprehension. 

The following is one or several reference documents for you to comprehend and memory:
---
{reference_docs}
---

The answer to the following question can be inferred through the above reference(s).
Question: {question}

Provide the answer to the given question through step-by-step reasoning. After your reasoning, output your final answer in the format of [[your answer]].
make sure the final answer enclosed by the double square brackets.

Let's think step by step. [/INST]"""


def read_jsonl(file_path):
    return [json.loads(jsonline) for jsonline in open(file_path, 'r')] 

def format_hotpotqa_prompt(reference, question, template, examples=None):
    if examples is None or len(examples) == 0:
        return template.format(reference_docs=reference, question=question)
    #* TO BE IMPLEMENTED...
    

def save_json(file_path, data): 
    with open(file_path, "w") as fout:
        for example in tqdm(data):
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("file saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, help="data path") #* processed hotpotqa dataset.
    parser.add_argument("--model_path",type=str, help="model path for inference")
    parser.add_argument("--save_name",type=str, help="save name")
    parser.add_argument("--save_path",type=str, help="save dir")
    parser.add_argument("--gpu_id",type=int, default=0, help="which gpu to use")
    parser.add_argument("--n",type=int, default=1, help="num of repetitions")
    parser.add_argument("--temperature",type=float, default=1.0, help="tempereture coefficient for generation")
    parser.add_argument("--is_mistral",type=int, default=0, help="whether use mistral prompt template")
    parser.add_argument("--is_qwen",type=int, default=0, help="whether use qwen prompt template (ChatML format)")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_id}'
    file_path = args.data_path
    #* receive processed data by data_processing.py
    dataset = read_jsonl(file_path)
    # dataset = dataset[:10]
    print(f"There are {len(dataset)} samples for inference")  


    random.seed(2024)
    prompts = []
    for d in dataset:
        question, reference_docs = d["input"], d["instruction"]
        if args.is_qwen:
            template = zero_shot_prompt_qwen
        elif args.is_mistral:
            template = zero_shot_prompt_mistral
        else: template =  zero_shot_prompt
        prompts.append(format_hotpotqa_prompt(reference_docs, question, template))
    # prompts = prompts[:10]
    print("SOME PROMPTS:\n"+"\n***\n".join(prompts[:3]))
    print(f"There are {len(prompts)} prompts for inference")  
    # infos about SamplingParams: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    n_repetitions = args.n
    temperature = args.temperature
    
    if args.is_qwen: sampling_params = SamplingParams(n=n_repetitions, temperature=temperature, max_tokens=768, stop=["<|endoftext|>"])
    else: sampling_params = SamplingParams(n=n_repetitions, temperature=temperature, max_tokens=768)
    llm = LLM(model=args.model_path, tensor_parallel_size=1, trust_remote_code=True)

    print("begin inference...")
    time_start = time()
    outputs = llm.generate(prompts, sampling_params)
    time_end = time()
    print(f"average speed for inference: {(len(prompts)*n_repetitions)/(time_end-time_start):>.4f} gen/s")

    for output, example in zip(outputs, dataset):
        if args.is_qwen:
            example[f"{args.save_name}"] = [x.text.replace("<|im_end|>", "") for x in output.outputs] # output.outputs[0].text
        elif args.is_chatglm3:
            example[f"{args.save_name}"] = [x.text.replace("<|assistant|>", "") for x in output.outputs] # output.outputs[0].text
        else: example[f"{args.save_name}"] = [x.text for x in output.outputs] # output.outputs[0].text

    #* save model responses
    save_json(args.save_path + f"/{args.save_name}_n_{n_repetitions}_tem_{temperature}.jsonl", dataset)