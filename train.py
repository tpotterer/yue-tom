import argparse
import pandas as pd
import torch
import numpy as np

from model.volatility_regressor import VolatilityRegressor

def train(model, device, train_loader, optimiser, criterion):
    model.train()
    training_loss = 0
    for (X, y) in train_loader:
        (a,b,c,d) = X
        X = (a.to(device), b.to(device), c.to(device), d.to(device))
        y = y.unsqueeze(-1).to(device)
        
        optimiser.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimiser.step()
        
        training_loss += loss.item()
    return training_loss / len(train_loader)

def validate(model, device, val_loader, criterion):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for (X, y) in val_loader:
            (a,b,c,d) = X
            X = (a.to(device), b.to(device), c.to(device), d.to(device))
            y = y.unsqueeze(-1).to(device)
            
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            validation_loss += loss.item()
    return validation_loss / len(val_loader)

def main(args):
    data = pd.read_pickle(args.input_path)
    
    data['year'] = data['date'].str[:4]
    max_len = max(data['role_tokens'].apply(lambda x: len(x)))
    
    print('Padding sequences to length {}'.format(max_len))
    def pad(seq):
        seq = np.array(seq)
        if(seq.shape[0] == max_len):
            return seq
        return np.append(seq, [seq[-1]] * (max_len - len(seq)), axis=0)
    
    data['role_tokens'] = data['role_tokens'].apply(lambda x: np.append(x, [0] * (max_len - len(x)), axis=0))
    data['part_tokens'] = data['part_tokens'].apply(lambda x: np.append(x, [0] * (max_len - len(x)), axis=0))
    data['utterance_emb'] = data['utterance_emb'].apply(pad)
    
    train_data = data[(data['year'] != args.val_year) & (data['year'] != args.test_year)]
    val_data = data[data['year'] == args.val_year]
    test_data = data[data['year'] == args.test_year]
    
    del data
    
    train_data = train_data[['seq_emb', 'utterance_emb', 'role_tokens', 'part_tokens', 'Y']]
    val_data = val_data[['seq_emb', 'utterance_emb', 'role_tokens', 'part_tokens', 'Y']]
    test_data = test_data[['seq_emb', 'utterance_emb', 'role_tokens', 'part_tokens', 'Y']]
    
    X_train = []
    y_train = []
    for (i, row) in train_data.iterrows():
        X_train.append((
            torch.tensor(row['seq_emb']).float(),
            torch.tensor(row['utterance_emb']).float(),
            torch.tensor(row['role_tokens']).long(),
            torch.tensor(row['part_tokens']).float()
        ))
        y_train.append(
            torch.tensor(row['Y']).float()
        )
        
    X_val = []
    y_val = []
    for (i, row) in val_data.iterrows():
        X_val.append((
            torch.tensor(row['seq_emb']).float(),
            torch.tensor(row['utterance_emb']).float(),
            torch.tensor(row['role_tokens']).long(),
            torch.tensor(row['part_tokens']).float()
        ))
        y_val.append(
            torch.tensor(row['Y']).float()
        )
        
    X_test = []
    y_test = []
    test_mse = []
    for (i, row) in test_data.iterrows():
        X_test.append((
            torch.tensor(row['seq_emb']).float(),
            torch.tensor(row['utterance_emb']).float(),
            torch.tensor(row['role_tokens']).long(),
            torch.tensor(row['part_tokens']).float()
        ))
        y_test.append(
            torch.tensor(row['Y']).float()
        )
        
    model = VolatilityRegressor(utt_emb_dim=768, seq_emb_dim=100)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-1)
    criterion = torch.nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(list(zip(X_val, y_val)), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=args.batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print('Epoch: {}'.format(epoch))
        train_loss = train(model, device, train_loader, optimiser, criterion)
        print('Training loss: {}'.format(train_loss))
        val_loss = validate(model, device, val_loader, criterion)
        print('Validation loss: {}'.format(val_loss))
        test_loss = validate(model, device, test_loader, criterion)
        print('Test loss: {}'.format(test_loss))
        
        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.output_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--input_path', type=str)
    argparser.add_argument('--output_path', type=str)
    argparser.add_argument('--val_year', type=str, default='2018')
    argparser.add_argument('--test_year', type=str, default='2019')
    
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epochs', type=int, default=100)
    argparser.add_argument('--lr', type=float, default=5e-4)
    
    args = argparser.parse_args()
    
    main(args)