# Importing modules
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import moviepy.editor as mpe

file_path = 'thunderridge_at_madison_audio.wav'
path = 'data/'

x, sr = librosa.load(path+file_path)

int(librosa.get_duration(x)/60)

max_slice = 5
window_length = max_slice * sr

a = x[21*window_length:22*window_length]
ipd.Audio(a, rate=sr)

energy = sum(abs(a**2))
len(a)


fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(a)
plt.show()

energy = np.array([sum(abs(x[i:i+window_length]**2)) for i in range(0, len(x), window_length)])
plt.hist(energy)
plt.show()

df = pd.DataFrame(columns=['energy', 'start', 'end'])
thresh = 400
row_index = 0
for i in range(len(energy)):
    value=energy[i]
    if value >= thresh:
        i=np.where(energy==value)[0]
        df.loc[row_index, 'energy'] = value
        df.loc[row_index, 'start'] = i[0] * 5
        df.loc[row_index, 'end'] = (i[0] + 1) * 5
        row_index += 1

temp=[]
i=0
j=0
n=len(df) - 2
m=len(df) - 1
while(i<=n):
    j=i+1
    while(j<=m):
        if(df['end'][i] == df['start'][j]):
            df.loc[i,'end'] = df.loc[j,'end']
            temp.append(j)
            j=j+1
        else:
            i=j
            break

df.drop(temp,axis=0,inplace=True)


from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
start=np.array(df['start'])
end=np.array(df['end'])
for i in range(len(df)):
    if(i!=0):
        start_lim = start[i] - 5
    else:
        start_lim = start[i]
    end_lim   = end[i]
    filename="data/" + str(i+1) + ".mp4"
    ffmpeg_extract_subclip(path+"thunderridge_at_madison.mp4",start_lim,end_lim,targetname=filename)

# Adding background music
my_clip = mpe.VideoFileClip('data/thunderridge_at_madison.mp4')
audio_background = mpe.AudioFileClip('data/swoope.mp3')
final_audio = mpe.CompositeAudioClip([my_clip.audio, audio_background])
final_clip = my_clip.set_audio(final_audio)
final_clip.write_videofile('data/thunderridge_at_madison2.mp4')