import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from main import audio_file_dict

class EmotionDataset(Dataset):
    def __init__(self, audio_file_dict):
        self.audio_fie_dict = audio_file_dict

    def __getitem__(self, index):
        img = list(audio_file_dict.index)[index]
        img, _ = torchaudio.load(img)
        img = torch.mean(img, dim=0).unsqueeze(0)
        img = torchaudio.transforms.Spectrogram()(img)
        img = F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])

        label = pd.get_dummies(audio_file_dict.emotion)[index]
        label = np.array(label)
        label = torch.from_numpy(label)
        return (img, label)

    def __len__(self):
        count = len(audio_file_dict)
        return count