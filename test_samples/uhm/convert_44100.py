import os
folder = '.'
for audio in os.listdir(folder):
    if audio.split('.')[-1] == 'wav':
        os.system("sox "+audio+" -r 44100 -c 1 -b 32 "+audio.split('.')[0]+"44100_mono_32bit.wav")