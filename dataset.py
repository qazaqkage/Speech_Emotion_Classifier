import pandas as pd
import os
import torchaudio
import torch.nn.functional as F

modality = {'01':'full_av','02':'video_only','03':'audio_only'}
vocal_channel = {'01':'speech','02':'song'}
emotion = {'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}
emotional_intensity = {'01':'normal','02':'strong'}
statement = {'01':'Kids are talking by the door','02':'Dogs are sitting by the door'}
reptition = {'01':'first_repitition','02':'second_repetition'}

def actor_f(num):
    if int(num)%2==0: return('female')
    else: return('male')

def creating_df():
    actors = sorted(os.listdir('C:/Users/LEGION/Desktop/Jupyret/datasets/RAVDESS/'))
    audio_file_dict = {}
    for actor in actors:
        actor_dir = os.path.join('C:/Users/LEGION/Desktop/Jupyret/datasets/RAVDESS/',actor)
        actor_files = os.listdir(actor_dir)
        actor_dict = [i.replace(".wav","").split("-") for i in actor_files]
        dict_entry = {os.path.join(actor_dir,i):j for i,j in zip(actor_files,actor_dict)}
        audio_file_dict.update(dict_entry)

    audio_file_dict = pd.DataFrame(audio_file_dict).T
    audio_file_dict.columns = ['modality','vocal_channel','emotion','emotional_intensity','statement','repetition','actor']

    audio_file_dict.modality = audio_file_dict.modality.map(modality)
    audio_file_dict.vocal_channel = audio_file_dict.vocal_channel.map(vocal_channel)
    audio_file_dict.emotion = audio_file_dict.emotion.map(emotion)
    audio_file_dict.emotional_intensity = audio_file_dict.emotional_intensity.map(emotional_intensity)
    audio_file_dict.statement = audio_file_dict.statement.map(statement)
    audio_file_dict.repetition = audio_file_dict.repetition.map(reptition)
    audio_file_dict['actor_sex'] = audio_file_dict.actor.apply(actor_f)

    return audio_file_dict

def load_audio(audio_file_dict):
    audio_files = []
    for i in list(audio_file_dict.index):
        i, _ = torchaudio.load(i)
        audio_files.append(i)
    return audio_files



def make_spec(audio_files):
    spectrograms = []
    for i in audio_files:
        specgram = torchaudio.transforms.Spectrogram()(i)
        spectrograms.append(specgram)
    return spectrograms

def get_minmax(spectrograms):
    max_width, max_height = max([i.shape[2] for i in spectrograms]), max([i.shape[1] for i in spectrograms])
    return max_width, max_height

def img_batch(max_width, max_height, spectrograms):
    image_batch = [
        # The needed padding is the difference between the
        # max width/height and the image's actual width/height.
        F.pad(img, [0, max_width - img.size(2), 0, max_height - img.size(1)])
        for img in spectrograms
    ]
    return image_batch

def cleaning(audio_files, spectrograms):
    del audio_files, spectrograms