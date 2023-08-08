from tqdm import tqdm
import argparse
import pandas as pd

def main(args):
    print("Loading data...")
    transcripts_df = pd.read_pickle(args.transcripts_path)
    stock_data_df = pd.read_pickle(args.stock_data_path)
    
    print(f"{len(transcripts_df)} transcripts")
    transcripts_df = transcripts_df[transcripts_df['ticker_exchange'].str.split(':').apply(lambda n: n[-1]).isin(stock_data_df['Ticker'].to_list())]
    print(f"{len(transcripts_df)} transcripts after filtering")
        
    print(f'{len(stock_data_df)} stock data rows')
    stock_data_df = stock_data_df[stock_data_df['Ticker'].isin(transcripts_df['ticker_exchange'].str.split(':').apply(lambda n: n[-1]).to_list())]
    print(f'{len(stock_data_df)} stock data rows after filtering')
    
    # Create a dictionary with ticker as the key and data as the value
    print("Creating stock dictionary...")
    stock_dict = {}
    for _, row in tqdm(stock_data_df.iterrows()):
        ticker = row['Ticker']
        if ticker not in stock_dict:
            stock_dict[ticker] = []
        stock_dict[ticker].append(row)
        
    # Add the Y value to each row of the data dataframe
    print("Matching targets...")
    vol_after = []
    vol_before = []
    after_cutoff = 3
    before_cutoff = 90
    for _, row in tqdm(transcripts_df.iterrows()):
        try:
            ticker = row['ticker_exchange'].split(':')[-1]
            if ticker in stock_dict:
                stock_data_df = stock_dict[ticker]
                data_after = [d for d in stock_data_df if d['DlyCalDt'] > row['date']]
                data_before = [d for d in stock_data_df if d['DlyCalDt'] < row['date']]
                if len(data_after) > 2:
                    vol_after += [[d['volatility'] for d in data_after[:after_cutoff]]]
                    vol_before += [[d['volatility'] for d in data_before[-before_cutoff:]]]
                else:
                    vol_after += [None]
                    vol_before += [None]
            else:
                vol_after += [None]
                vol_before += [None]
        except Exception as e:
            vol_after += [None]
            vol_before += [None]

    # Add the Y column to the data dataframe
    transcripts_df['vol_after'] = vol_after
    transcripts_df['vol_before'] = vol_before
    
    transcripts_df = transcripts_df.dropna(subset=['vol_after', 'vol_before'])
    
    transcripts_df = transcripts_df[transcripts_df['vol_before'].apply(len) == before_cutoff]
    transcripts_df = transcripts_df[transcripts_df['vol_after'].apply(len) == after_cutoff]
    
    transcripts_df['Y'] = transcripts_df['vol_after'].apply(lambda x: x[-1])

    # Drop rows with missing Y values
    transcripts_df = transcripts_df.dropna(subset=['Y'])
    
    # Save the dataframe
    print("Saving dataframe...")
    transcripts_df.to_pickle(args.output_path)
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--transcripts_path', type=str, required=True)
    parser.add_argument('--stock_data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)