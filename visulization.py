import matplotlib.pyplot as plt
import seaborn as sns

def stats(audio_file_dict):
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 8))
    ax1[0].barh(y=audio_file_dict.emotion.value_counts().index, width=audio_file_dict.emotion.value_counts().values)
    ax1[0].set_title('Emotion')
    ax1[1].bar(x=audio_file_dict.actor_sex.value_counts().index, height=audio_file_dict.actor_sex.value_counts().values)
    ax1[1].set_title('Actor Sex')
    ax2[0].bar(x=audio_file_dict.emotional_intensity.value_counts().index,
               height=audio_file_dict.emotional_intensity.value_counts().values)
    ax2[0].set_title('Emotional Intensity')
    ax2[1].bar(x=audio_file_dict.statement.value_counts().index, height=audio_file_dict.statement.value_counts().values)
    plt.xticks(rotation=45)
    ax2[1].set_title('Statement')
    fig.tight_layout()
    return plt.figshow()