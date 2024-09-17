import transformers, torch, re
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList, LOGITS_PROCESSOR_INPUTS_DOCSTRING
from transformers.utils.doc import add_start_docstrings

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import LogSoftmax, Softmax

from collections import defaultdict


prompt_for_judge_hotpotqa = """Judge the correctness of the answer in the following Q&A scenario:
###
{output}
###

Judge: """


generation_config = {
    "use_cache": True, 
    "max_new_tokens": 512,
    "output_scores": True,
    "return_dict_in_generate": True,
}
    
class SelfEvaluationDecodingLogitsProcessorpe(LogitsProcessor):
    r"""
    Args:
        model: model used to generate tokens
        tokenizer: tokenizer used to tokenize the input
        k: If the predicted probability distribution of the current token is dispersed, select k tokens for speculation
        generation_config: generation configuration
        ratio: ratio-based threshold for judging whether the distribution of tokens is dispersed
        entropy: entropy-based threshold for judging whether the distribution of tokens is dispersed
        doubleBracketTokenIdx: judge model's token id of ']],', which is used to location of token 'Y' and thereby obtain the probability of predicting token 'Y'
        correctTokenIdx: judge model's token id for 'Y'
        branchingLimit: the maximum number of times the model can speculate on different chaotic points in one sequence
        judge_model: model used to evaluate the correctness of speculations
        judge_tokenizer: tokenizer used to tokenize the input of the judge model
        force_words_ids: force the model to generate the specified token
        alpha: fusion coefficient
    """

    def __init__(self, model, tokenizer, k, generation_config, ratio=0.5, entropy=None, doubleBracketTokenIdx=5262, correctTokenIdx=29979, branchingLimit=5, judge_model=None, judge_tokenizer=None, force_words_ids = [],alpha=0.8,speculation_type='greedy'):
        self.model = model
        self.judge_model = judge_model if judge_model else model
        self.reso_tokenizer = tokenizer
        self.judge_tokenizer = judge_tokenizer if judge_tokenizer else tokenizer
        self.force_words_ids = force_words_ids
        self.reso_tokenizer.padding_side = 'left'
        self.judge_tokenizer.padding_side = 'left' # Otherwise, model could output unexpected tokens
        self.k = k
        self.init_attn_mask = None # shape = [batch_size, init_seq_len]
        self.generation_config = generation_config
        self.pattern = r'\[\[(.*?)\]\]' # used for extract self-evaluation probability
        self.doubleBracketTokenIdx = doubleBracketTokenIdx 
        self.correctTokenIdx = correctTokenIdx
        self.log_softmax = LogSoftmax(dim=-1)
        self.softmax = Softmax(dim=-1) # calculate token probabilities
        self.branchingLimit = branchingLimit
        self.cur_branching_num = 0
        self.ratio = ratio
        self.entropy = entropy
        self.alpha=alpha
        self.speculation_type=speculation_type
        

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        #& input_ids: the generated tokens before the current token， shape = [batch_size*num_beams, seq_len]
        #& scores: scores for each vocabulary token before SoftMax, i.e. the logits of each token after the lm_head, in the shape of [batch_size*num_beams, vocab_size]
        #& If you want to obtain the distribution probability of each token, you need to do softmax on scores, i.e. probs = softmax(scores)
        #& Key steps:
        #& 1. chaotic detection
        #& 2. If the distribution is not dispersed, return scores directly. 
        #& 3. Otherwise, speculate on the top k tokens and judge the quality of each token based on the speculations
        
        #* chaotic detection  
        mask_for_input_ids = self._check(scores) 
        if torch.any(mask_for_input_ids): #* Chaotic
            assert self.init_attn_mask is not None           
            for i,b in enumerate(mask_for_input_ids):
                if b.item() and self.cur_branching_num[i]<self.branchingLimit:
                    self.cur_branching_num[i] += 1       
                    input_ids_sub = input_ids[i:i+1]
                    scores_sub = scores[i:i+1]
                    #* concat top-k tokens to the existing context.
                    new_input_ids, top_k_indices, token_probs = self._expand_input_ids(input_ids_sub, scores_sub, self.k)  
                    new_seq_len = new_input_ids.shape[-1]
                    new_input_ids = new_input_ids.reshape(-1, new_seq_len) # shape = [reduced_batch_size * k, seq_len+1] 
                    new_input_attn_mask = torch.ones_like(new_input_ids, device=new_input_ids.device) # shape = [reduced_batch_size * k, seq_len+1] 
                    init_seq_len = self.init_attn_mask.shape[1]
                    new_input_attn_mask[:,:init_seq_len] = self.init_attn_mask.unsqueeze(1).expand(-1,self.k,-1).reshape(-1,init_seq_len) # shape = [reduced_batch_size * k, seq_len+1] 
                    inputs = {
                        "input_ids": new_input_ids,
                        "attention_mask": new_input_attn_mask,
                    }
                    
                    #* speculation
                    model_outputs = self.model.generate(**inputs, **self.generation_config) 
                    
                    #* evaluation
                    output_ids = model_outputs["sequences"]
                    output_texts = self.reso_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    judge_texts = [prompt_for_judge_hotpotqa.format(output=t) for t in output_texts]
                    judge_inputs = self.judge_tokenizer(judge_texts, return_tensors='pt', padding=True)

                    if self.speculation_type=='greedy':
                        num_beams = None
                        model_outputs = self.judge_model.generate(input_ids=judge_inputs.input_ids.to(self.judge_model.device), 
                                                                attention_mask=judge_inputs.attention_mask.to(self.judge_model.device),
                                                                **self.generation_config)

                    elif self.speculation_type=='beam':
                        num_beams = 3
                        model_outputs = self.judge_model.generate(input_ids=judge_inputs.input_ids.to(self.judge_model.device), 
                                                            do_sample = False,
                                                            num_beams = num_beams,
                                                            attention_mask=judge_inputs.attention_mask.to(self.judge_model.device),
                                                            **self.generation_config)
                        
                    #* obtain the self-evaluation probabilities p_{se}
                    output_ids, logits = model_outputs["sequences"], model_outputs["scores"]

                    if num_beams: #* special implementation for beam search
                        logits = self._extract_for_beam(logits, num_beams)
                    
                    #* cat 之后: shape = [reduced_batch_size=1 * k * (full_seq_len-input_seq_len), vocab_size]
                    logits = torch.cat(logits, dim=0) 
                    
                    batch_size, input_seq_len, full_seq_len = judge_inputs["input_ids"].shape[0], judge_inputs["input_ids"].shape[1], output_ids.shape[1]
                    
                    label_token_pos = self._find_last_token_pre_idx(output_ids) 
                    reordered_rowIdxs = [] 
                    for r, idx in label_token_pos:
                        r, idx = r.item(), idx.item()-input_seq_len
                        reordered_rowIdxs.append(idx*batch_size+r)
                        
                    log_probs = self.softmax(logits[reordered_rowIdxs])
                    log_probs = log_probs[:,self.correctTokenIdx]
                    log_probs = log_probs.reshape(-1, self.k)
                    print('log_probs_dist:', log_probs)
                    sub_scores = scores_sub 
                    #* all tokens' self-evaluation probabilities are approximately 0 
                    if torch.max(log_probs).item()<1e-3:
                       sub_scores[torch.arange(top_k_indices.shape[0]), top_k_indices.squeeze(0)] = -float("inf")
                    else:
                        #* fusion
                        alpha = self.alpha
                        #* propensity score.
                        log_probs = alpha*log_probs + (1-alpha)*token_probs
                        #* pick the maximium one
                        target_tokens = top_k_indices[torch.arange(log_probs.shape[0]), torch.argmax(log_probs, dim=1)] # [reduced_batch_size]
                        sub_scores[torch.arange(target_tokens.shape[0]), target_tokens] = torch.topk(sub_scores, k=1).values.squeeze(1) + 1
                    #* update scores
                    scores[i:i+1] = sub_scores
        return scores
    
    
    def _extract_for_beam(self, logits, beam_size):
        extracted_logits = []
        for logits_for_each_pos in logits:
            res = []
            for i,l in enumerate(logits_for_each_pos):
                if i % beam_size==0: 
                    res.append(l.view(1,-1))
            extracted_logits.append(torch.cat(res))
        return extracted_logits
    
    def _set_init_attn_mask(self, init_attn_mask):
        self.init_attn_mask = init_attn_mask
    
    def _is_bad(self, prob_dist):
        if self.entropy: #* entropy-based threshold
            eps = 1e-10
            entr = -torch.sum(prob_dist * torch.log2(prob_dist + eps))
            return entr > self.entropy
        #* ratio-based threshold
        p1, p2 = torch.topk(prob_dist,k=2).values
        return p2>p1*self.ratio or p1<0.2
    
    def _check(self, scores: torch.FloatTensor) -> bool:
        scores = self.softmax(scores)
        return torch.tensor([
            self._is_bad(prob_dist) for prob_dist in scores
        ], device=scores.device)
    
    def _expand_input_ids(self, input_ids, next_token_prob_distribution, k):
        token_probs, top_k_indices = torch.topk(self.softmax(next_token_prob_distribution), k=k, dim=1) # [batch_size, k]

        input_ids_expanded = input_ids.unsqueeze(1).repeat(1, k, 1)  # [batch_size, k, seq_len]
        top_k_indices_expanded = top_k_indices.unsqueeze(2) # [batch_size, k, 1]

        new_input_ids = torch.cat((input_ids_expanded, top_k_indices_expanded), dim=-1)

        return new_input_ids, top_k_indices, token_probs # shape = [batch_size, k, seq_len+1], [batch_size, k]

    def _find_last_token_pre_idx(self, output_ids):
        indices = torch.nonzero(output_ids == self.doubleBracketTokenIdx, as_tuple=False)
        unique_indices = [] 
        cnt = defaultdict(int)
        for i in range(indices.shape[0]-1, -1, -1):
            r, idx = indices[i]
            if cnt[r.item()] == 0:
                cnt[r.item()] += 1
                unique_indices.append([r.item(), idx.item()])
        for i in range(output_ids.shape[0]): # full_batch_size
            if cnt[i]==0: 
                unique_indices.append([i, output_ids.shape[1]-1])
        indices = torch.tensor(unique_indices, device=output_ids.device)
        indices[:, 1] -= 1
        return indices[torch.argsort(indices[:, 0], dim=0)]

prompt_for_reso = """Given the following math question, answer it through step-by-step reasoning.
---
Question: {question}
---

Answer: """

prompt_for_judge =  """Judge the correctness of the answer in the following Q&A scenario:
###
{output}
###

Judge: """

generation_config_for_judge = {
    "use_cache": True, 
    "max_new_tokens": 512,
    # "do_sample": False,
    # "num_beams": 1,
    "output_scores": True,
    "return_dict_in_generate": True,
}

softmax = torch.nn.Softmax(dim=-1)
            
pattern = r'\[\[(.*?)\]\]'
def extract(response):
    try: extracted_ans = re.findall(pattern, response)[-1].strip()
    except: extracted_ans = ""
    return extracted_ans

def find_last_token_pre_idx(output_ids, doubleBracketTokenIdx):
    indices = torch.nonzero(output_ids == doubleBracketTokenIdx, as_tuple=False)
    unique_indices = [] 
    cnt = defaultdict(int)
    for i in range(indices.shape[0]-1, -1, -1):
        r, idx = indices[i]
        if cnt[r.item()] == 0:
            cnt[r.item()] += 1
            unique_indices.append([r.item(), idx.item()])
    for i in range(output_ids.shape[0]): # full_batch_size
        if cnt[i]==0: 
            unique_indices.append([i, output_ids.shape[1]-1])
    indices = torch.tensor(unique_indices, device=output_ids.device)
    indices[:, 1] -= 1
    return indices[torch.argsort(indices[:, 0], dim=0)]

def self_consistency(model, tokenizer, prompt, num_return_sequences, temperature=0.7, top_p=0.9, top_k=20):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # inputs.input_ids.shape = [1, seq_len]
    # model_output.shape = [num_return_sequences, max_seq_len]
    model_output = model.generate(**inputs,
                                  do_sample=True,
                                  temperature = temperature,
                                  top_p = top_p,
                                  top_k = top_k,
                                  num_return_sequences = num_return_sequences,
                                  max_new_tokens=320,
                                )
    # len(tokenizer.batch_decode(model_output)) == num_return_sequences
    memo = defaultdict(list) 
    for response in tokenizer.batch_decode(model_output, skip_special_tokens=True): 
        memo[extract(response)].append(response.replace("<|endoftext|>",""))
    print(memo.keys())
    # majority voting
    pred_num = sorted([(k, len(r)) for k,r in memo.items()], key=lambda x:-x[1])[0][0]
    return memo[pred_num][0]    

def best_of_N(model, tokenizer, prompt, N, correctTokenIdx, doubleBracketTokenIdx, temperature=0.7, top_p=0.9, top_k=20):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen_outputs = model.generate(**inputs,
                                  do_sample=True,
                                  temperature = temperature,
                                  top_p = top_p,
                                  top_k = top_k,
                                  num_return_sequences = N,
                                  max_new_tokens=512,
                                )
    output_texts = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
    judge_texts = [prompt_for_judge.format(output=t) for t in output_texts] 
    judge_inputs = tokenizer(judge_texts, return_tensors='pt', padding=True)
    judge_outputs = model.generate(input_ids=judge_inputs.input_ids.to(model.device), 
                                   attention_mask=judge_inputs.attention_mask.to(model.device),
                                   **generation_config_for_judge)
    judge_outputs = model.generate(input_ids=judge_inputs.input_ids.to(model.device),attention_mask=judge_inputs.attention_mask.to(model.device),**generation_config_for_judge)
    output_ids, logits = judge_outputs["sequences"], judge_outputs["scores"]
    logits = torch.cat(logits, dim=0) 
    batch_size, input_seq_len, full_seq_len = judge_inputs["input_ids"].shape[0], judge_inputs["input_ids"].shape[1], output_ids.shape[1]
    label_token_pos = find_last_token_pre_idx(output_ids, doubleBracketTokenIdx)
    reordered_rowIdxs = [] 
    for r, idx in label_token_pos:
        r, idx = r.item(), idx.item()-input_seq_len
        reordered_rowIdxs.append(idx*batch_size+r)   
    log_probs = softmax(logits[reordered_rowIdxs]) # shape = [N, vocab_size]
    log_probs = log_probs[:,correctTokenIdx] # shape = [reduced_batch_size * k]
    print(log_probs)
    return output_texts[torch.argmax(log_probs).item()]
    

# if __name__ == "__main__":
#     print("loading...")
#     tokenizer = AutoTokenizer.from_pretrained("/data1/zqluo/plms/gpt2-124m")
#     model = AutoModelForCausalLM.from_pretrained("/data1/zqluo/plms/gpt2-124m")
#     inputs = tokenizer("Name a President of China: ", return_tensors="pt")
#     print("input_ids.shape:", inputs.input_ids.shape)
    
#     logits_processor = LogitsProcessorList()
#     logits_processor.append(SelfEvaluationDecodingLogitsProcessorpe(model, tokenizer, 2, generation_config, doubleBracketTokenIdx=11907, correctTokenIdx=56, branchingLimit=1))
    
#     print("inferring...")
#     logits_processor[0].init_attn_mask = inputs.attention_mask
#     logits_processor[0].cur_branching_num = 0
#     gen_out = model.generate(**inputs, min_new_tokens=10, logits_processor=logits_processor)
#     print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))