1. First, you have to download the thought viz dataset pickle file and its equivalent images (images are not required for feature extraction learning, just class names are required).
2. Then run train.py code and it will run for around 3000 epochs.
3. After train, now run the find_bestckpt.py code, it will give you the path of best checkpoint with better test acc.
4. Finally you can run inference.py for generating tsne plot.
