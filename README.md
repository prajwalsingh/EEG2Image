## EEG2Image
<br/>

EEG2IMAGE: Image Reconstruction from EEG Brain Signals [ICASSP 2023]

[[Paper](https://arxiv.org/abs/2302.10121)]

<br/>

* EEG2Image best_ckpt [[Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EWC0lT5vEN1c206cJ0tdmdQBkVhvCL5TVnNhBI7cWSTKFg?e=jrpnh9)]
* Inception Score [[Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/EfWLlhNk0CxXqgMnsKgt8k8BxSqflp98ACpl9ZLScWSHtA?e=cEfq0R)]
* Preprocessed thoughtviz EEG Data [[Link](https://iitgnacin-my.sharepoint.com/:u:/g/personal/19210048_iitgn_ac_in/Ea4Sp2UH__ZbRQGZXu9o-6cByJK4E6E4GtxrcVony9_Q8g?e=bVdyIJ)]

## Updates

Follow up work: Learning Robust Deep Visual Representations from EEG Brain Recordings (WACV 2024) [ [Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Singh_Learning_Robust_Deep_Visual_Representations_From_EEG_Brain_Recordings_WACV_2024_paper.pdf) | [Code](https://github.com/prajwalsingh/EEGStyleGAN-ADA) ]

## Abstract
Reconstructing images using brain signals of imagined visuals may provide an augmented vision to the disabled, leading to the advancement of Brain-Computer Interface (BCI) technology. The recent progress in deep learning has boosted the study area of synthesizing images from brain signals using Generative Adversarial Networks (GAN). In this work, we have proposed a framework for synthesizing the images from the brain activity recorded by an electroencephalogram (EEG) using small-size EEG datasets. This brain activity is recorded from the subject's head scalp using EEG when they ask to visualize certain classes of Objects and English characters. We use a contrastive learning method in the proposed framework to extract features from EEG signals and synthesize the images from extracted features using conditional GAN. We modify the loss function to train the GAN, which enables it to synthesize $128 \times 128$ images using a small number of images. Further, we conduct ablation studies and experiments to show the effectiveness of our proposed framework over other state-of-the-art methods using the small EEG dataset.

## Architecture

<img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/EEG2Image_Architecture.png"/>


## Results

| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/eeg_lstm_classification.png"/> | <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/embedding_space.png"/> |
|-|-|
| t-SNE visualization of Object test dataset EEG feature space, which is learned using label supervision with test classification accuracy 0.75 and k-means accuracy 0.18. | t-SNE visualization of Object test dataset EEG feature space, which is learned using triplet loss with test k-means accuracy 0.53. Each cluster’s equivalent EEG-based generated images are also visualized in this plot. |


| ThoughtViz  |   EEG2Image (Ours)      |
|----------|:-------------:|
| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/comparison.png"/> |  <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/our_image.jpg"/> |
| <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/comparision_alphabet.png"/> |  <img src="https://github.com/prajwalsingh/EEG2Image/blob/main/results/ours_alphabet.jpg"/> |


## Note

* Unstructured code.
* You can find anaconda environment yml file in anaconda folder.


## References

* ThoughtViz [[Link](https://github.com/ptirupat/ThoughtViz)]
