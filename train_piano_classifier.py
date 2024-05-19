import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import dataset
from model.piano_classifier import PianoClassifier

def train_piano_classifier(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data, piano_type in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, piano_type)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='config json file', default='config.json')
    parser.add_argument('-d_out', help='output directory', default='../output')
    parser.add_argument('-d_dataset', help='dataset directory', default='./dataset')
    parser.add_argument('-epoch', help='number of epochs(10)', type=int, default=10)
    parser.add_argument('-batch', help='batch size(8)', type=int, default=8)
    parser.add_argument('-lr', help='learning rate(1e-04)', type=float, default=1e-4)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d_dataset = args.d_dataset.rstrip('/')
    dataset_train = dataset.MyDataset(d_dataset+'/feature/train.pkl',
                                      d_dataset+'/label_onset/train.pkl',
                                      d_dataset+'/label_offset/train.pkl',
                                      d_dataset+'/label_mpe/train.pkl',
                                      d_dataset+'/label_velocity/train.pkl',
                                      d_dataset+'/idx/train.pkl',
                                      d_dataset+'/piano_type/train.pkl',
                                      config,
                                      args.n_slice)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch, shuffle=True)

    input_size = config['feature']['n_bins']
    num_piano_types = 8
    model = PianoClassifier(input_size, num_piano_types).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_piano_classifier(model, dataloader_train, criterion, optimizer, epochs=args.epoch)

    torch.save(model.state_dict(), os.path.join(args.d_out, 'piano_classifier.pth'))
