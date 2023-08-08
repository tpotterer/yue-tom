import argparse
import pandas as pd
import os
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from utils.extract_data_from_html import extract_data

def extract_data_for_row(args, row):
    result = extract_data(os.path.join(args.input_dir, row['file_name']))
    if(result):
        for key in result:
            row[key] = result[key]
    return row

def main(args):
    # read file names
    file_list = [i for i in os.listdir(args.input_dir) if i[0] != '.']
    print(f"Found {len(file_list)} files in {args.input_dir}")
    
    # build filename df
    files_df = pd.DataFrame(file_list, columns=['file_name'])
    
    # create temp dir
    os.mkdir(args.temp_dir)
    
    # split file names into chunks
    print("Extracting data from files...")
    step = 10000
    chunk_idxs = list(range(0, len(files_df), step))
    for i in range(0, len(files_df), step):
        if(f"chunk_{i}.pkl" not in os.listdir(args.temp_dir)):
            chunk = files_df[i:i+step]
            data = chunk.parallel_apply(lambda row: extract_data_for_row(args, row), axis=1)
        else:
            data = pd.read_pickle(os.path.join(args.temp_dir, f"chunk_{i}.pkl"))
            
        data = data[data['has_error'] == False]
        data.to_pickle(os.path.join(args.temp_dir, f"chunk_{i}.pkl"))
        print(f"Completed {i}/{len(files_df)}")
        
            
    print("Combining chunks...")
    data = pd.DataFrame()
    for file in os.listdir(args.temp_dir):
        data = pd.concat([data, pd.read_pickle(os.path.join(args.temp_dir, file))])

    # save data
    print(f"Saving data to {args.output_path}")
    data = data.drop(['has_error', 'error'], axis=1)
    data.to_pickle(args.output_path)
    
    print("Finished, you should delete the temp directory")
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--page_source_type', type=str, default='seekingalpha', choices=['seekingalpha'])
    parser.add_argument('--temp_dir', type=str, default='temp')
    
    args = parser.parse_args()
    main(args)