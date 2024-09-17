from time import time
from tqdm import tqdm
import os, json, argparse, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList
from logits_processor_utils import SelfEvaluationDecodingLogitsProcessorpe, generation_config
from logits_processor_utils import self_consistency, best_of_N


prompt_for_reso = """Given the following math question, answer it through step-by-step reasoning.
---
Question: {question}
---

Answer: """


def clean_numbers(numbers):
    cleaned = [re.sub(r'\.$', '', re.sub(r'[^\d\.]', '', num)) for num in numbers]
    return cleaned
def fs_cothub_gsm8k_match_answer(answer, response):
    #* answer = standard CoT reasoning path in GSM8K dataset.
    #* response = model response
    # CoT hub match answer for GSM8k, match last numeric value and compare with the expected answer

    # Patterns to find numeric values after specific markers in both answer and response
    pattern1 = '####.*?(\d*\.?\d+)'
    patterns2 = [
    'So, the final answer is \$\s*(\d+\.?\d*)\s*',
    'The answer is: \s*(\d+\.?\d*)',
    'Therefore, the final answer is \$\s*(\d+\.?\d*)',
    'Therefore, the final answer is \s*(\d+\.?\d*)',
    'So, the final answer is \s*(\d+\.?\d*)'
]
    # Extract numbers from the answer and response
    ans_numbers = re.findall(pattern1, answer)
    pred_numbers = []
    for pa in patterns2:
        pred_numbers = re.findall(pa, response)
        if pred_numbers:
            break
    
    if len(pred_numbers) == 0:
        pred_numbers = re.findall(pa, response)
    pred_numbers = clean_numbers(pred_numbers)
    # Check if both answer and response contain numbers and compare the last found number
    if ans_numbers and pred_numbers:
        # Compare the last number found in both answer and response
        if ans_numbers[-1] == pred_numbers[-1]:
            return 1, pred_numbers[-1]  # Match found, return 1 and the matching number
        else:
            return 0, pred_numbers[-1]  # Numbers do not match, return 0 and the last number from the response

    return 0, "No match or missing numbers"  # No numbers found or other issues


def read_jsonl(file_path):
    return [json.loads(jsonline) for jsonline in open(file_path, 'r')] 
    
def save_json(file_path, data): 
    with open(file_path, "w", encoding="utf-8") as fout:
        for example in tqdm(data):
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("file saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",type=str, help="file path of evaluation data")
    parser.add_argument("--model_path",type=str, help="directory path of model for generation")
    parser.add_argument("--judge_model_path",type=str, help="directory path of model for judgement") #* in SED, it is identical to --model_path
    parser.add_argument("--gpu_ids",type=int, default=0, help="GPU ids for inference")
    parser.add_argument("--eval_res_name",type=str, help="save name")
    parser.add_argument("--do_sed",type=int, default=0, help="whether to apply SED")
    parser.add_argument("--do_beam_search",type=int, default=0, help="whether to apply beam search")
    parser.add_argument("--num_beams",type=int, default=3, help="whether to apply beam search")
    parser.add_argument("--do_sample",type=int, default=0, help="whether to apply sampling")
    parser.add_argument("--do_self_consistency",type=int, default=0, help="whether to apply self-consistency")
    parser.add_argument("--do_best_of_n",type=int, default=0, help="whether to apply best-of-N")
    parser.add_argument("--branching_limit",type=int, default=1, help="the maximum number of times the model can speculate on different chaotic points in one sequence")
    parser.add_argument("--topk",type=int, default=2, help="number of token candidates for speculation")
    parser.add_argument("--ratio",type=float, default=0.5, help="ratio-based detection threshold")
    parser.add_argument('--alpha',type=float,default=0.8,help='fusion coeifficient')
    parser.add_argument('--speculation_type',type=str,default='greedy',help='which decoding strategy to use for speculation')
    parser.add_argument('--generation_type',type=str,default='greedy',help='which decoding strategy to use for generation')
    parser.add_argument('--entropy',type=float, help="entropy-based detection threshold")

    args = parser.parse_args()

    #^ 1. read data
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu_ids}'
    file_path = args.data_path
    eval_dataset = read_jsonl(file_path)
    
    print(f"There are {len(eval_dataset)} samples for evaluation")  
    
    template = prompt_for_reso 
    
    prompts = []
    answers = []
    labels = []
    p = r'####.*?(\d*\.?\d+)'
    for d in eval_dataset:
        question = d['question']
        answer = d['answer']
        ans_number = re.findall(p, answer)[-1]
        prompts.append(template.format(question=question))
        answers.append(answer)
        labels.append(ans_number)
                
    print("SOME PROMPTS:\n"+"\n***\n".join(prompts[:3]))
    
    #^ 2. infer
    outputs = []
    llm = AutoModelForCausalLM.from_pretrained(args.model_path)
    llm.to("cuda")
    # llm.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm_judge, tokenizer_judge = None, None
    if args.do_sed:
        llm_judge = AutoModelForCausalLM.from_pretrained(
            args.judge_model_path,
            torch_dtype = "auto",
            device_map = "auto",
        ) if args.judge_model_path != args.model_path else llm
        tokenizer_judge = AutoTokenizer.from_pretrained(args.judge_model_path) if args.judge_model_path != args.model_path else tokenizer

   
        if 'falcon' in args.model_path:
            doubleBracketTokenIdx = 60513
            correctTokenIdx = 68
        elif 'gemma' in args.model_path:
            doubleBracketTokenIdx = 41492
            correctTokenIdx = 235342
        else:
            doubleBracketTokenIdx = 20526
            correctTokenIdx = 29979
        logits_processor = LogitsProcessorList()
        logits_processor.append(SelfEvaluationDecodingLogitsProcessorpe(llm, tokenizer, args.topk, generation_config, ratio = args.ratio, 
                                                                doubleBracketTokenIdx=doubleBracketTokenIdx, correctTokenIdx=correctTokenIdx,
                                                        #    doubleBracketTokenIdx=60513, correctTokenIdx=68, #* falcon 
                                                        #    doubleBracketTokenIdx=20526, correctTokenIdx=29979, #* llama2 
                                                        #    doubleBracketTokenIdx=41492, correctTokenIdx=235342, #* gemma
                                                           branchingLimit=args.branching_limit,
                                                           judge_model=llm_judge, judge_tokenizer=tokenizer_judge,alpha=float(args.alpha),speculation_type=args.speculation_type,entropy=args.entropy
                                                           ))
    time_start = time()
    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        #* applying Self-Evaluation Decoding on the model
        if args.do_sed:
            logits_processor[0].init_attn_mask = inputs.attention_mask
            if args.generation_type=='greedy':
                logits_processor[0].cur_branching_num = [0]
                output = llm.generate(**inputs, max_new_tokens=768, logits_processor=logits_processor)[0] #* greedy for generation
            elif args.generation_type=='sample':
                logits_processor[0].cur_branching_num = [0]
                output = llm.generate(**inputs, 
                                    do_sample=True, 
                                    temperature = 0.7,
                                    top_p = 0.9,
                                    top_k = 10,
                                    max_new_tokens=768, 
                                    logits_processor=logits_processor)[0] #* sample for generation
            elif args.generation_type=='beam':
                logits_processor[0].cur_branching_num = [0]*args.num_beams
                output = llm.generate(**inputs, max_new_tokens=768, num_beams=args.num_beams, logits_processor=logits_processor)[0] #* beam for generation
            outputs.append(tokenizer.decode(output, skip_special_tokens=True))
        elif args.do_self_consistency:
            outputs.append(
                self_consistency(llm, tokenizer, prompt, num_return_sequences = args.branching_limit*args.topk)
            )
        elif args.do_best_of_n:
            outputs.append(
                best_of_N(llm, tokenizer, prompt, N=args.branching_limit*args.topk, # N=args.branching_limit*args.topk,
                  correctTokenIdx=correctTokenIdx, 
                  doubleBracketTokenIdx=doubleBracketTokenIdx)
            )
        elif args.do_beam_search: #* Without applying SED, the model uses beam search for decoding.
            outputs.append(tokenizer.batch_decode(llm.generate(**inputs, num_beams=args.num_beams, max_new_tokens=768))[0])
        elif args.do_sample: #* Without applying SED, the model uses sampling for decoding.
            outputs.append(tokenizer.batch_decode(llm.generate(**inputs, 
                                                               do_sample=True, 
                                                               temperature = 0.7,
                                                               top_p = 0.9,
                                                               max_new_tokens=768))[0])
        else: #* Without applying SED, the model uses greedy search for decoding.
            outputs.append(tokenizer.batch_decode(llm.generate(**inputs, max_new_tokens=768))[0])
    time_end = time()
    print(f"average speed for inference: {(len(prompts))/(time_end-time_start):>.4f} gen/s")
    # pattern = r'\[\[(.*?)\]\]' # extract the answer from the response
    
    #^ 3. evaluate
    right_cnt, total_cnt = 0, len(eval_dataset)
    eval_res = []
    for i, (output, example) in enumerate(zip(outputs, eval_dataset)):
        ans = example["answer"].strip().lower()
        response = output
        #* extract answer and compare them.
        example["judge"], extracted_ans = fs_cothub_gsm8k_match_answer(ans, response)
        if example["judge"]:
            right_cnt += 1
        eval_res.append({**example, "response": response, "extracted_ans": extracted_ans})
    
    #* Print accuracy
    print(f"Model info: {args.model_path}\thit_num{right_cnt}\tEM rate: {right_cnt/total_cnt:>6f}")

    #* save results
    save_json(f"{args.eval_res_name}.json", eval_res)
