import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils.tokenise_transcript import get_tokens
import torch

tqdm.pandas()

def main(args):
    data = pd.read_pickle(args.input_path).sample(3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokeniser = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    
    def embed_row(row):
        (utterance_tokens, role_tokens, part_tokens) = get_tokens(row['transcript'], row['company_participants'], row['other_participants'], tokeniser, max_num_utterances=args.max_transcript_length)
        (utterance_tokens, role_tokens, part_tokens) = utterance_tokens.to(device), role_tokens.to(device), part_tokens.to(device)
        
        utterance_emb = model(utterance_tokens).pooler_output
        
        row['utterance_emb'] = utterance_emb.detach().numpy()
        row['role_tokens'] = role_tokens.detach().numpy()
        row['part_tokens'] = part_tokens.detach().numpy()
        
        return row
        
    data = data.progress_apply(embed_row, axis=1)
    
    data.to_pickle(args.output_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--input_path', type=str, required=True)
    argparser.add_argument('--output_path', type=str, required=True)
    argparser.add_argument('--model_name', type=str, default='prosusai/finbert')
    argparser.add_argument('--max_transcript_length', type=int, default=256)
    
    args = argparser.parse_args()
    
    main(args)