import torch
import re

def check_inclusion(val, lst):
    if(val in lst):
        return True
    for l in lst:
        if(val in l):
            return True

def get_tokens(transcript, executives, analysts, tokeniser, max_num_utterances=256):
    executives = ['operator'] + [e.lower() for e in executives]
    analysts = ['unidentified analyst'] + [a.lower() for a in analysts]
    
    utterance_tokens = torch.tensor(tokeniser.batch_encode_plus([re.sub(' +', ' ', " ".join(i[1])) for i in transcript], truncation=True, padding='longest', max_length=512)['input_ids'])
    role_tokens = torch.zeros(len(transcript))
    part_tokens = torch.zeros(len(transcript))
    
    for (i, (speaker, _, is_presentation)) in enumerate(transcript):
        role_tokens[i] = 1 if check_inclusion(speaker.lower(), executives) else 2 if check_inclusion(speaker.lower(), analysts) else 0
        part_tokens[i] = 1 if is_presentation else 0
        
    role_tokens = role_tokens.long()
    part_tokens = part_tokens.long()
    
    return utterance_tokens, role_tokens, part_tokens