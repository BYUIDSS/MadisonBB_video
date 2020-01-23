# Loading modules

from __future__ import unicode_literals
import youtube_dl

url = 'https://www.youtube.com/watch?v=Jt58svr-JxM'

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=Jt58svr-JxM'])