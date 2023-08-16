import argparse
import torch
import pandas as pd
import re
from datasets import Dataset, DatasetDict
from functools import lru_cache, partial
import numpy as np
from gluonts.time_feature import get_lags_for_frequency, time_features_from_frequency_str, TimeFeature
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig, PretrainedConfig
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from accelerate import Accelerator
import torch.optim as optim
from tqdm import tqdm

quarter_regex = re.compile(f"Q[1-4]\s+20[0-2][0-9]")

def quarter_to_q(quarter):
    if(len(quarter_regex.findall(quarter)) == 1):
        return int(quarter.split()[0][1])-1
    
    # assume it is of the other format
    q = quarter.split("Q")[0][1:]
    year = quarter.split("Q")[1].strip()
    if(len(year) == 2):
        year = '20' + year
        
    return int(q)-1

@lru_cache(10_000)
def convert_to_pandas_period(date, freq):
    return pd.Period(date, freq)

def transform_start_field(batch, freq):
    batch['start'] = [convert_to_pandas_period(date, freq) for date in batch['start']]
    return batch


def create_transformation(freq, config):
    remove_field_names = []
    if(config.num_static_real_features == 0):
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if(config.num_dynamic_real_features == 0):
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if(config.num_static_categorical_features == 0):
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)
    
    # a bit like sklearn pipeline    
    return Chain(
        # remove unspecified fields
        [RemoveFields(field_names=remove_field_names)]
        # convert to numpy array
        +([AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=int)] if config.num_static_categorical_features > 0 else [])
        +([AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1)] if config.num_static_real_features > 0 else [])
        +([AsNumpyArray(field=FieldName.TARGET, expected_ndim=1 if config.input_size == 1 else 2)])
        # handle nans 
        +[
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            )
        ]
        # add temporal features based on freq
        +[
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=[],
                pred_length=config.prediction_length,   
            )
        ]
        # add another temporal feature
        +[
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            )
        ]
        # vertically stack features
        +[
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                +([FieldName.FEAT_DYNAMIC_REAL] if config.num_dynamic_real_features > 0 else [])
            )
        ]
        # rename to match huggingface
        +[
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask"
                }
            )
        ]
    )
    
def create_instance_splitter(
    config,
    mode,
    train_sampler = None,
    validation_sampler = None,
):
    assert mode in ['train', 'validation', 'test']
    
    instance_sampler = {
        'train': train_sampler or ExpectedNumInstanceSampler(num_instances=1, min_future=config.prediction_length),
        'validation': validation_sampler or ValidationSplitSampler(min_future=config.prediction_length),
        'test': TestSplitSampler(),
    }[mode]
    
    return InstanceSplitter(
        target_field='values',
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=['time_features', 'observed_mask'],
    )
    

def create_train_dataloader(
    config,
    freq,
    data,
    batch_size,
    num_batches_per_epoch,
    shuffle_buffer_length = None,
    cache_data = True,
    **kwargs
):
    PREDICTION_INPUT_NAMES = [
        'past_time_features',
        'past_values',
        'past_observed_mask',
        'future_time_features',
    ]
    if(config.num_static_categorical_features > 0):
        PREDICTION_INPUT_NAMES.append('static_categorical_features')
    if(config.num_static_real_features > 0):
        PREDICTION_INPUT_NAMES.append('static_real_features')
        
    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + ['future_values', 'future_observed_mask']
    
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=True)
    if(cache_data):
        transformed_data = Cached(transformed_data)
        
    instance_splitter = create_instance_splitter(config, 'train')
    
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(stream, is_train=True)
    
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    
def create_test_dataloader(
    config,
    freq,
    data,
    batch_size
):
    PREDICTION_INPUT_NAMES = [
        'past_time_features',
        'past_values',
        'past_observed_mask',
        'future_time_features',
    ]
    
    if(config.num_static_categorical_features > 0):
        PREDICTION_INPUT_NAMES.append('static_categorical_features')
    if(config.num_static_real_features > 0):
        PREDICTION_INPUT_NAMES.append('static_real_features')
        
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)
    
    instance_sampler = create_instance_splitter(config, 'test')
    
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,  
    )

def main(args):
    data = pd.read_pickle(args.input_path).reset_index(drop=True)
    data['q'] = data['quarter'].apply(quarter_to_q)
    
    test_year = '2019'
    val_year = '2018'
    encoder_length = 90
    prediction_length = 3

    train = []
    test = []
    val = []
    all_examples = []

    num_i = 0
    ticker_to_num = {t: i for i, t in enumerate(data['ticker_exchange'].unique())}
    print("Data loaded.")

    for i, row in data.iterrows():
        year = row['date'][:4]
        start = row['date']
        y = row['Y']
        vol_before = row['vol_before']
        vol_after = row['vol_after']
        
        if(len(vol_before) == encoder_length):
            if(year == test_year):
                test.append({
                    'start': start,
                    'target': vol_before + vol_after,
                    'feat_static_cat': [],
                    'feat_static_real': [],
                    'item_id': i,
                    'y': y
                })
            elif(year == val_year):
                val.append({
                    'start': start,
                    'target': vol_before + vol_after,
                    'feat_static_cat': [],
                    'feat_static_real': [],
                    'item_id': i,
                    'y': y
                })
            if(year != test_year and year != val_year):
                train.append({
                    'start': start,
                    'target': vol_before + vol_after,
                    'feat_static_cat': [],
                    'feat_static_real': [],
                    'item_id': i,
                    'y': y
                })
            # else:
            #     train.append({
            #         'start': start,
            #         'target': vol_before,
            #         'feat_static_cat': [num_i],
            #         'feat_static_real': [],
            #         'item_id': i,
            #         'y': y
            #     })
            all_examples.append({
                    'start': start,
                    'target': vol_before,
                    'feat_static_cat': [num_i],
                    'feat_static_real': [],
                    'item_id': i,
                    'y': y
                })
                    
            num_i += 1

    train = Dataset.from_pandas(pd.DataFrame(train))
    val = Dataset.from_pandas(pd.DataFrame(val))
    test = Dataset.from_pandas(pd.DataFrame(test))

    dataset = DatasetDict({
        'train': train,
        'validation': val,
        'test': test
    })
    train_examples = dataset["train"]
    val_examples = dataset["validation"]
    test_examples = dataset["test"]
    
    freq='1D'

    train_examples.set_transform(partial(transform_start_field, freq=freq))
    test_examples.set_transform(partial(transform_start_field, freq=freq))
    val_examples.set_transform(partial(transform_start_field, freq=freq))

    lags_seq = [i for i in get_lags_for_frequency(freq) if i <= encoder_length]
    time_features = []
    
    config = TimeSeriesTransformerConfig(
        prediction_length=prediction_length,
        context_length=encoder_length,
        lags_sequence=lags_seq,
        num_time_features=len(time_features) + 1,
        num_static_categorical_features=0,
        # cardinality=[len(train_examples)],
        # embedding_dimension=[4],
        
        encoder_layers=4,
        decoder_layers=4,
        d_model=128,
        n_head=4,
        decoder_ffn_dim=512,
        encoder_ffn_dim=512,
        dropout=0.1,
    )
    
    train_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=train_examples,
        batch_size=256,
        num_batches_per_epoch=len(train_examples) // 256,
    )

    all_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=train_examples,
        batch_size=256,
        num_batches_per_epoch=24
    )

    test_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=test_examples,
        batch_size=64
    )

    val_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=val_examples,
        batch_size=64
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accelerator = Accelerator(cpu=False)
    device = accelerator.device

    model = TimeSeriesTransformerForPrediction(config)
    model = model.to(device)
    optimiser = optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    model, optimiser, train_dataloader = accelerator.prepare(
        model,
        optimiser,
        train_dataloader
    )
    
    print('Len train: {}'.format(len(train_examples)))
    print('Len val: {}'.format(len(val_examples)))
    print('Len test: {}'.format(len(test_examples)))

    best_test_loss = float('inf')
    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            optimiser.zero_grad()
            outputs = model(
                past_values=batch['past_values'].to(device),
                past_time_features=batch['past_time_features'].to(device),
                past_observed_mask=batch['past_observed_mask'].to(device),
                static_categorical_features=batch['static_categorical_features'].to(device) if(config.num_static_categorical_features > 0) else None,
                static_real_features=batch['static_real_features'].to(device) if(config.num_static_real_features > 0) else None,
                future_values=batch['future_values'].to(device),
                future_time_features=batch['future_time_features'].to(device),
                future_observed_mask=batch['future_observed_mask'].to(device),
                output_hidden_states=True,
            )
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimiser.step()
            
            if(idx % 100 == 0):
                print(f'Epoch: {epoch}, Batch: {idx}, Loss: {loss.item()}')
                        
        model.eval()
        forecasts = []
        for batch in test_dataloader:
                outputs = model.generate(
                    past_values=batch['past_values'].to(device),
                    past_time_features=batch['past_time_features'].to(device),
                    past_observed_mask=batch['past_observed_mask'].to(device),
                    static_categorical_features=batch['static_categorical_features'].to(device) if(config.num_static_categorical_features > 0) else None,
                    static_real_features=batch['static_real_features'].to(device) if(config.num_static_real_features > 0) else None,
                    future_time_features=batch['future_time_features'].to(device),
                    output_hidden_states=True,
                )
                forecasts += [outputs.sequences.cpu().numpy()]
                
        forecasts = np.concatenate(forecasts)

        forecast_median = np.median(forecasts,1)[:,-1]
        errors = []
        for item_id, ts in enumerate(test_examples):
            forecast = forecast_median[item_id]
            ground_truth = ts['y']

            errors += [(forecast - ground_truth)**2]
            
        print(f"test MSE: {np.array(errors).mean()}")
        if(np.array(errors).mean() < best_test_loss):
            best_test_loss = np.array(errors).mean()
            torch.save(model.state_dict(), 'best_test_model.pt')

        forecasts = []
        for batch in val_dataloader:
                outputs = model.generate(
                    past_values=batch['past_values'].to(device),
                    past_time_features=batch['past_time_features'].to(device),
                    past_observed_mask=batch['past_observed_mask'].to(device),
                    static_categorical_features=batch['static_categorical_features'].to(device) if(config.num_static_categorical_features > 0) else None,
                    static_real_features=batch['static_real_features'].to(device) if(config.num_static_real_features > 0) else None,
                    future_time_features=batch['future_time_features'].to(device),
                    output_hidden_states=True,
                )
                forecasts += [outputs.sequences.cpu().numpy()]
                
        forecasts = np.concatenate(forecasts)

        forecast_median = np.median(forecasts,1)[:,-1]
        errors = []
        for item_id, ts in enumerate(val_examples):
            forecast = forecast_median[item_id]
            ground_truth = ts['y']

            errors += [(forecast - ground_truth)**2]
            
        print(f"val MSE: {np.array(errors).mean()}")
        if(np.array(errors).mean() < best_val_loss):
            best_val_loss = np.array(errors).mean()
            torch.save(model.state_dict(), 'best_val_model.pt')
            
        
    torch.save(model.state_dict(), args.output_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)