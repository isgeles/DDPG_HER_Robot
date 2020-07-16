import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from IPython.display import display
from PIL import Image

def animate_frames(frames, jupyter=True):
    """
    Animate frames from array (with ipython HTML extension for jupyter).
    @param frames: list of frames
    @param jupyter: (bool), for using jupyter notebook extension to display animation
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('off')
    cmap = None if len(frames[0].shape) == 3 else 'Greys'
    patch = plt.imshow(frames[0], cmap=cmap)

    anim = animation.FuncAnimation(plt.gcf(),
        lambda x: patch.set_data(frames[x]), frames=len(frames), interval=30)

    if jupyter:
        display(HTML(anim.to_jshtml()))  # ipython extension
    else:
        plt.show()
    plt.close()


def save_gif(frames):
    """
    Save animation of frames to gif-file in working directory.
    @param frames: list of frames
    """
    images = [Image.fromarray(frames[i]) for i in range(len(frames))]
    with open('./openai_gym.gif', 'wb+') as f:  # change the path if necessary
        im = Image.new('RGB', images[0].size)
        im.save(f, save_all=True, append_images=images, optimize=False)
    pass






