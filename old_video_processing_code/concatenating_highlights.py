from moviepy.editor import VideoFileClip, concatenate_videoclips

vids = []

for i in range(1,38):
    vids.append(VideoFileClip("data/{}.mp4".format(i)))

final_clip = concatenate_videoclips(vids)
final_clip.write_videofile("data/highlights_thunderridge_at_madison.mp4")