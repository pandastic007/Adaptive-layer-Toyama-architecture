import torch
import pickle

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, f_feature, f_label_onset, f_label_offset, f_label_mpe, f_label_velocity, f_idx, f_piano_type, config, n_slice):
        super().__init__()

        with open(f_feature, 'rb') as f:
            feature = pickle.load(f)

        with open(f_label_onset, 'rb') as f:
            label_onset = pickle.load(f)
        with open(f_label_offset, 'rb') as f:
            label_offset = pickle.load(f)
        with open(f_label_mpe, 'rb') as f:
            label_mpe = pickle.load(f)
        if f_label_velocity is not None:
            self.flag_velocity = True
            with open(f_label_velocity, 'rb') as f:
                label_velocity = pickle.load(f)
        else:
            self.flag_velocity = False

        with open(f_idx, 'rb') as f:
            idx = pickle.load(f)

        with open(f_piano_type, 'rb') as f:
            piano_type = pickle.load(f)

        self.feature = torch.from_numpy(feature)
        self.label_onset = torch.from_numpy(label_onset)
        self.label_offset = torch.from_numpy(label_offset)
        self.label_mpe = torch.from_numpy(label_mpe)
        if self.flag_velocity:
            self.label_velocity = torch.from_numpy(label_velocity)
        if n_slice > 1:
            idx_tmp = torch.from_numpy(idx)
            self.idx = idx_tmp[:int(len(idx_tmp) / n_slice) * n_slice][::n_slice]
        else:
            self.idx = torch.from_numpy(idx)
        self.piano_type = torch.from_numpy(piano_type)
        self.config = config
        self.data_size = len(self.idx)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        idx_feature_s = self.idx[idx] - self.config['input']['margin_b']
        idx_feature_e = self.idx[idx] + self.config['input']['num_frame'] + self.config['input']['margin_f']

        idx_label_s = self.idx[idx]
        idx_label_e = self.idx[idx] + self.config['input']['num_frame']

        spec = (self.feature[idx_feature_s:idx_feature_e]).T
        label_onset = self.label_onset[idx_label_s:idx_label_e]
        label_offset = self.label_offset[idx_label_s:idx_label_e]
        label_mpe = self.label_mpe[idx_label_s:idx_label_e].float()
        piano_type = self.piano_type[idx]

        if self.flag_velocity:
            label_velocity = self.label_velocity[idx_label_s:idx_label_e].long()
            return spec, label_onset, label_offset, label_mpe, label_velocity, piano_type
        else:
            return spec, label_onset, label_offset, label_mpe, piano_type
