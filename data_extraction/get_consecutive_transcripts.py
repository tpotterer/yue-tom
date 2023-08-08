import pandas as pd
import re
import argparse

quarter_regex = re.compile(f"Q[1-4]\s+20[0-2][0-9]")

def do_zip(row):
    return [(a,b) for (a,b) in zip(row['quarter'], row['index'])]

def get_exchange(val):
    if(":" in val):
        return val.split(":")[0]
    return None

def get_ticker(val):
    if(":" in val):
        return val.split(":")[1]
    return val

def quarter_to_q(quarter):
    if(len(quarter_regex.findall(quarter)) == 1):
        return f"{quarter.split()[1]}-{quarter.split()[0]}"
    
    # assume it is of the other format
    q = quarter.split("Q")[0][1:]
    year = quarter.split("Q")[1].strip()
    if(len(year) == 2):
        year = '20' + year
        
    return f"{year}-Q{q}"

def quarter_to_number(quarter_str):
    year, quarter = quarter_str.split("-")
    quarter_number = int(year) * 4 + int(quarter[1])
    return quarter_number

def find_longest_consecutive_quarters(quarters):
    quarters_numbers = [quarter_to_number(q) for q in quarters]
    quarters_numbers.sort()
    
    longest_sequence = []
    current_sequence = []
    for i in range(len(quarters_numbers)):
        if i == 0 or quarters_numbers[i] == quarters_numbers[i-1] + 1:
            current_sequence.append(quarters_numbers[i])
        else:
            current_sequence = [quarters_numbers[i]]
        
        if len(current_sequence) > len(longest_sequence):
            longest_sequence = current_sequence
    longest_quarters_sequence = [f"{q // 4 if conv(q) != 4 else (q//4)-1}-Q{conv(q)}" for q in longest_sequence]
    return longest_quarters_sequence

def conv(q):
    t = q%4
    if(t == 0):
        return 4
    return t

def get_longest_seq(row):
    return find_longest_consecutive_quarters([q for q,idx in row['q_idx']])


def filter_transcripts(data, n, output_path):
    print('Performing initial filtering...')
    data = data.dropna()
    data['index'] = data.index
    data['exchange'] = data['ticker_exchange'].apply(get_exchange)
    data['ticker'] = data['ticker_exchange'].apply(get_ticker)
    
    data = data[data['exchange'].isin(["NYSE", "NASDAQ"])]
    data['q_idx'] = data.apply(lambda n: (quarter_to_q(n['quarter']), n['index']), axis=1)
    data['q'] = data['q_idx'].apply(lambda n: n[0])
    
    data = data.drop_duplicates(subset=['q', 'ticker_exchange'])
    
    print('Grouping by ticker...')
    data_by_ticker = data.groupby(by=['ticker_exchange']).agg({'q_idx': list})

    print('Finding longest uninterrupted sequence...')
    data_by_ticker['uninterrupted'] = data_by_ticker.apply(get_longest_seq, axis=1)
    data_by_ticker['unint_length'] = data_by_ticker['uninterrupted'].apply(len)
    
    data_by_ticker = data_by_ticker[data_by_ticker['unint_length'] >= n]
    
    print('Calculating indexes...')
    data_by_ticker['selected_idxs'] = data_by_ticker.apply(lambda n: [idx for q,idx in n['q_idx'] if q in n['uninterrupted']], axis=1)
    all_indexes = [idx for idxs in data_by_ticker['selected_idxs'] for idx in idxs]
    
    # select data with indexes
    print(f"Found {len(all_indexes)} transcripts with uninterrupted sequences of length {n} or more. Saving them now.")
    data.loc[all_indexes].to_pickle(output_path)
    
    
def main(args):
    print("Loading data...")
    data = pd.read_pickle(args.input_path)
    filter_transcripts(data, args.n, args.output_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    
    
    