To convert the raw html into transcripts.

convert_raw_pages.py 
    --input_dir (location of seekingalpha pages)
    --output_path (desired location of pickled dataframe)

get_consecutive_transcripts.py
    --input_path (output of above)
    --output_path (desired location of pickled dataframe)
    --n (required minimum number of consecutive transcripts to be included, i.e n=10 includes runs of transcripts >= 10)

get_targets.py
    --transcripts_path (path of pickled dataframe)
    --stock_data_path (path of daily crsp dataframe)
    --output_path (desired location of pickled dataframe)

Result is a dataframe of consecutive transcripts populated with the volatility 90 days prior and 3 days following the EC date.