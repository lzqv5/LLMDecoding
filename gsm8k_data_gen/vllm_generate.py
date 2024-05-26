from vllm import LLM, SamplingParams

vllm_initialized = False

def vllm_initialize(args):
    global vllm_initialized
    global sampling_params
    global model
    model = LLM(
        model=args.model_path, 
        trust_remote_code=True,
        tensor_parallel_size = args.tensor_parallel_size
    )
    sampling_params = SamplingParams(temperature=0.9, top_p=0.5, top_k=50, max_tokens=args.max_length, stop=['<|eot_id|>'])
    sampling_params = SamplingParams(
        temperature=0.7,  
        top_p=0.9,
        top_k=50,
        max_tokens=2048,
        stop=['<|eot_id|>']
        )
    
    vllm_initialized = True

def vllm_generate(prompts, args, generation_name=""):
    global vllm_initialized
    global model
    global sampling_params
    if not vllm_initialized:
        vllm_initialize(args)
    generated_texts = []
    prompts_list = []
    prompts_per_step = 1000
    for i in range(0, len(prompts), prompts_per_step):
        prompts_list.append(prompts[i:i+prompts_per_step])

    for i, inputs in enumerate(prompts_list):
        print(f"generating {generation_name} [{i * prompts_per_step + 1}, {len(prompts)}] in {len(prompts)} prompts")
        outputs = model.generate(inputs, sampling_params)
        for i in range(len(outputs)):
            output = outputs[i]
            if len(output.prompt_token_ids) >= args.max_length - 32:
                generated_text = "<input length exceeded>"
            else:
                generated_text = output.outputs[0].text
            generated_texts.append(generated_text)
    return generated_texts
