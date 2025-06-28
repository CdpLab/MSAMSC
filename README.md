#  <p align="center">Visually Impaired People Recognize Virtual Objects Through 3D Reconstructed Shapes</p>

 <p align="center">Jia Liu, Yongze Li, Peng Gao, Jiaxu Ning, Lina Wei, and Dapeng Chen*</p>
  <p align="center">Nanjing University of Information Science and Technology</p>

## <p align="center">ABSTRACT</p>
Recent studies have shown that multimodal sentiment analysis is effective for depression diagnosis. However, semantic conflicts across modalities still limit performance. To address this, we propose a semantic conflict-based multimodal sentiment analysis and depression diagnosis method. First, we used the independent component analysis decoupling model to separate the features extracted by the pre-trained model into sentiment-related and non-sentiment-related components. It enables effective extraction of emotion-relevant features while minimizing noise. Next, to tackle the issue of multimodal "semantic conflict", we design a multimodal co-fusion strategy. This strategy simulates the human brain's process of synthesizing emotional judgments based on trimodal inputs. It leverages the complementarity of semantic features to eliminate "deceptive" information across modalities, and combines interactive information between different modalities to fully extract emotional cues. Furthermore, we introduce an Emotionally constrained loss function, through which optimized training yields improved multimodal sentiment analysis performance and robustness. Experimental results show that on the CMU-MOSI dataset, our method improves the Pearson correlation coefficient (Corr) and F1-score by 5.5% and 1%, respectively. On the CMU-MOSEI dataset, it enhances Corr and F1-score by 3.6% and 1.1%, respectively. Additionally, experiments on the DAIC-WOZ dataset demonstrate that our model outperforms the best baseline. Thus, the proposed model effectively handles dynamic emotional variations in long sequences.

## MSASC

This is the implementation of [A Multimodal Sentiment Analysis and Diagnostic Method of Depression based on Semantic Conflict]in PyTorch (Version 1.8.1).

## Requirements

```
   pytorch-transformers==1.2.0
   numpy==1.20.2
   torch==1.8.1
   torchvision==0.9.1
   tqdm==4.58.0
   transformers==3.0.2
   scikit-learn==0.24.2
   six==1.15.0
   wandb==0.10.25
```

This repository contains the source code of our paper, using the following datasets:

- [CMU-MOSI](https://drive.google.com/file/d/1FDDMuPK_r_2HSpc0GNyDucYYUId_UD_8/view?usp=sharing)
Download the preprocessed datasets and put them into `dataset/`.

- [CMU-MOSEI](https://drive.google.com/file/d/1IsCctXAjVqxvoeYhgUEVslYm-5t0vOG0/view?usp=sharing)
Download the preprocessed datasets and put them into `dataset/`.

- [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)
  Download the preprocessed datasets and put them into `dataset/`.
  Please note that the dataset is available for academic research only and require approval before downloading.
  Once approved, you will be provided with a download link and further usage instructions.

## Train
Please follow the steps below in order to be able to train our models:

1 - Install Requirements

```
pip install -r requirements.txt
```

2 - Download the preprocessed datasets and put them into `dataset/`.

 

3 -  Usage:
```
python train.py
```

 ## Document Description
 
- `\config\global_configs.py`: Different data sizes trained on three datasets.
-  For instance, to test on MOSI using the sizes:

ACOUSTIC_DIM = 74    VISUAL_DIM = 27    TEXT_DIM = 768

to test on MOSEI using the sizes:

ACOUSTIC_DIM = 74    VISUAL_DIM = 35    TEXT_DIM = 768

to test on DAIC-WOZ using the sizes:

ACOUSTIC_DIM = 39    VISUAL_DIM = 49     TEXT_DIM = 768



If you find this material useful, please cite the following article:

## Citation
```

```

## Contact
Should you have any questions, please feel free to contact me at [202412220510@nuist.edu.cn]

<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fguangyizhangbci%2FPARSE&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
