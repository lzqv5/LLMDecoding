import json, argparse, re
from tqdm import tqdm

def read_jsonl(file_path):
    return [json.loads(jsonline) for jsonline in open(file_path, 'r')] 

def parse_dataset(dataset):
    parsed_dataset = []
    n = len(dataset)
    #* skip header
    for i in tqdm(range(1, n)):
        example = dataset[i]
        context = example['context']
        context = re.sub(r'\[TLE\]|\[SEP\]', '\n', context).strip()
        context_pieces = context.split("[PAR]")
        parsed_context = ""
        cnt = 0
        for ctxt in context_pieces:
            if ctxt.strip():
                cnt += 1
                parsed_context += f"Reference Document {cnt}:[{ctxt.strip()}]\n\n"
        parsed_dataset.extend([{
            # "instruction": context,
            "qid": qa['qid'],
            "instruction": parsed_context,
            "input": qa['question'],
            "output": qa['answers'][0],
        } for qa in example['qas']])
    return parsed_dataset

def save_parsed_dataset(path, parsed_dataset):
    with open(path, "w") as fout:
        for example in tqdm(parsed_dataset):
            # fout中每行写入序列化后的json对象
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")
        print("file saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #* https://huggingface.co/datasets/mrqa, HotpotQA subset.
    parser.add_argument("--data_path",type=str,help="data path of mrqa dataset")
    args = parser.parse_args()

    file_path = args.data_path
    dataset = read_jsonl(file_path)
    parsed_dataset = parse_dataset(dataset)
    print(f"There are {len(dataset)} samples in data path")
    print(f"After processing, there are {len(parsed_dataset)} samples.")
    print("saving...")
    save_parsed_dataset(file_path.replace(".jsonl", "_parsed.json"), parsed_dataset)
    print("done.")
    

    

