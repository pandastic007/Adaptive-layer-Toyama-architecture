import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import dataset
from model.adaptive_amt_model import AdaptiveAMTModel
from model.piano_classifier import PianoClassifier
from model.model_spec2midi import Encoder_SPEC2MIDI, Decoder_SPEC2MIDI

def train_adaptive_model(model, classifier, train_loader, criterion, optimizer, epochs=10):
    model.train()
    classifier.eval()
    for epoch in range(epochs):
        for data, target, piano_type in train_loader:
            with torch.no_grad():
                piano_type_pred = classifier(data).argmax(dim=1)
            
            optimizer.zero_grad()
            output = model(data, piano_type_pred)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='config json file', default='config.json')
    parser.add_argument('-d_out', help='output directory', default='../output')
    parser.add_argument('-d_dataset', help='dataset directory', default='./dataset')
    parser.add_argument('-n_div_train', help='num of train dataset division(1)', type=int, default=1)
    parser.add_argument('-n_div_valid', help='num of valid dataset division(1)', type=int, default=1)
    parser.add_argument('-n_div_test', help='num of test dataset division(1)', type=int, default=1)
    parser.add_argument('-n_slice', help='dataset slice(0: num_frame, 1>=: this number)(16)', type=int, default=16)
    parser.add_argument('-epoch', help='number of epochs(100)', type=int, default=100)
    parser.add_argument('-resume_epoch', help='number of epoch to resume(-1)', type=int, default=-1)
    parser.add_argument('-resume_div', help='number of div to resume(-1)', type=int, default=-1)
    parser.add_argument('-batch', help='batch size(8)', type=int, default=8)
    parser.add_argument('-lr', help='learning rate(1e-04)', type=float, default=1e-4)
    parser.add_argument('-dropout', help='dropout parameter(0.1)', type=float, default=0.1)
    parser.add_argument('-clip', help='clip parameter(1.0)', type=float, default=1.0)
    parser.add_argument('-seed', type=int, default=1234, help='seed value(1234)')
    parser.add_argument('-cnn_channel', help='number of cnn channel(4)', type=int, default=4)
    parser.add_argument('-cnn_kernel', help='number of cnn kernel(5)', type=int, default=5)
    parser.add_argument('-hid_dim', help='size of hidden layer(256)', type=int, default=256)
    parser.add_argument('-pf_dim', help='size of position-wise feed-forward layer(512)', type=int, default=512)
    parser.add_argument('-enc_layer', help='number of layer of transformer(encoder)(3)', type=int, default=3)
    parser.add_argument('-dec_layer', help='number of layer of transformer(decoder)(3)', type=int, default=3)
    parser.add_argument('-enc_head', help='number of head of transformer(encoder)(4)', type=int, default=4)
    parser.add_argument('-dec_head', help='number of head of transformer(decoder)(4)', type=int, default=4)
    parser.add_argument('-weight_A', help='loss weight for 1st output(1.0)', type=float, default=1.0)
    parser.add_argument('-weight_B', help='loss weight for 2nd output(1.0)', type=float, default=1.0)
    parser.add_argument('-valid_test', help='validation with test data', action='store_true')
    parser.add_argument('-v', help='verbose(print debug)', action='store_true')
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
    classifier = PianoClassifier(input_size, num_piano_types).to(device)
    classifier.load_state_dict(torch.load(os.path.join(args.d_out, 'piano_classifier.pth')))

    encoder = Encoder_SPEC2MIDI(config['input']['margin_b'],
                                config['input']['num_frame'],
                                config['feature']['n_bins'],
                                args.cnn_channel,
                                args.cnn_kernel,
                                args.hid_dim,
                                args.enc_layer,
                                args.enc_head,
                                args.pf_dim,
                                args.dropout,
                                device)
    decoder = Decoder_SPEC2MIDI(config['input']['num_frame'],
                                config['feature']['n_bins'],
                                config['midi']['num_note'],
                                config['midi']['num_velocity'],
                                args.hid_dim,
                                args.dec_layer,
                                args.dec_head,
                                args.pf_dim,
                                args.dropout,
                                device)
    model = AdaptiveAMTModel(encoder, decoder, num_piano_types, config).to(device)
    model.apply(initialize_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_adaptive_model(model, classifier, dataloader_train, criterion, optimizer, epochs=args.epoch)

    torch.save(model.state_dict(), os.path.join(args.d_out, 'adaptive_amt_model.pth'))
