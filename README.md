## EEG2Image
<br/>

EEG2IMAGE: Image Reconstruction from EEG Brain Signals [ICASSP 2023]

[[Paper](https://arxiv.org/abs/2302.10121)]

<br/>

* EEG2Image best_ckpt [[Link](https://drive.google.com/file/d/1gdmm_qlGGUF0AM8X0a3JDg0Dc2HOpn7k/view?usp=share_link)]
* Inception Score [[Link](https://drive.google.com/file/d/1nQWX3eYSLH1LX56HJ1fQLIgzmiYKgpY_/view?usp=share_link)]
* Preprocessed thoughtviz EEG Data [[Link](https://drive.google.com/file/d/1j_vNNXROc3MKe4lW7DwwLaVfpXguD0A8/view?usp=share_link)]


## Abstract
Reconstructing images using brain signals of imagined visuals may provide an augmented vision to the disabled, leading to the advancement of Brain-Computer Interface (BCI) technology. The recent progress in deep learning has boosted the study area of synthesizing images from brain signals using Generative Adversarial Networks (GAN). In this work, we have proposed a framework for synthesizing the images from the brain activity recorded by an electroencephalogram (EEG) using small-size EEG datasets. This brain activity is recorded from the subject's head scalp using EEG when they ask to visualize certain classes of Objects and English characters. We use a contrastive learning method in the proposed framework to extract features from EEG signals and synthesize the images from extracted features using conditional GAN. We modify the loss function to train the GAN, which enables it to synthesize $128 \times 128$ images using a small number of images. Further, we conduct ablation studies and experiments to show the effectiveness of our proposed framework over other state-of-the-art methods using the small EEG dataset.

## Architecture

<img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/EEG2Image_Architecture.png"/>


## Results

| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/eeg_lstm_classification.png"/> | <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/embedding_space.png"/> |
|-|-|
| t-SNE visualization of Object test dataset EEG feature space which is learned using label supervision with test classification accuracy 0.75 and k-means accuracy 0.18. | t-SNE visualization of Object test dataset EEG feature space which is learned using triplet loss with test k-means accuracy 0.53. Each cluster’s equivalent EEG-based generated images are also visualized in this plot. |


| ThoughtViz  |   EEG2Image (Ours)      |
|----------|:-------------:|
| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/comparison.png"/> |  <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/our_image.jpg"/> |
| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/comparision_alphabet.png"/> |  <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/ours_alphabet.jpg"/> |


## Note

* Unstructured code.


## References

* ThoughtViz [[Link](https://github.com/ptirupat/ThoughtViz)]
