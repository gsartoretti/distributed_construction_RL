import numpy as np
import sys


#This code allows gifs to be saved of the training episode for use in the Control Center.
def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    def make_mask(t):
        try:
            x = salIMGS[int(len(salIMGS)/duration*t)]
        except:
            x = salIMGS[-1]
        return x

    clip = mpy.VideoClip(make_frame, duration=duration)
    if salience == True:
        mask = mpy.VideoClip(make_mask, ismask=True,duration= duration)
        clipB = clip.set_mask(mask)
        clipB = clip.set_opacity(0)
        mask = mask.set_opacity(0.1)
        mask.write_gif(fname, fps = len(images) / duration,verbose=False)
    else:
        clip.write_gif(fname, fps = len(images) / duration,verbose=False)


if __name__ == '__main__':
    assert len(sys.argv) > 2, "Missing pickle file names"

    for i in range(1, len(sys.argv)):
        pickle_name = sys.argv[i]
        episode_frames = pickle.load(open(pickle_name, "rb"))

        time_per_step = 0.1
        images = np.array(episode_frames)

        make_gif(images, './gifs/'+pickle_name.replace('.frames', '.gif'), duration=len(images)*time_per_step,true_image=True,salience=False)
