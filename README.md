# quantative_style

This is the source code for the WACV 2020 paper "Improving style tranfer with calibrated metrics"

https://arxiv.org/pdf/1910.09447.pdf

<img src='teasor.png' width=400>

Codes have three parts:

## Evaluation of base E statistics

The code is implemneted in E_base.ipynb.

We calculate the minus log KL distance between feature statistics of symthesized image and style image. For the evaluation of KL distance, We also use the PCA to reduce the rank of feature channels. Please see our paper for the details. 


## C-Value evaluation

We use the off-the-shelf contour detection method by Arbelaezet al. [1],which estimates Pb from an image.   We use the standardmetric,(the F-score, which is a harmonic mean of precisionand recall between Pb and human-drawn contour map). Thefinal contour detection score is the Maximum F-score of aprecision-recall curve.   We  compute  the  final  contour  de-tection scores with the transferred images’ Pb and groundtruth contours from the content images.  The resulting con-tour detection scores are the base C statistics.  
For sourcecontent images and human annotated ground truth contourmaps we choose 200 test images from BSDS500[1].

References
[1]  P.  Arbelaez,  M.  Maire,  C.  Fowlkes,  and  J.  Malik. Con-tour detection and hierarchical image segmentation.IEEEtransactions  on  pattern  analysis  and  machine  intelligence,33(5):898–916, 2011.

## Calibrating E statistic and Calibrating C statistic

[50styles_1.pdf](https://github.com/stringtron/quantative_style/files/4075093/50styles_1.pdf)

[50styles_2.pdf](https://github.com/stringtron/quantative_style/files/4075094/50styles_2.pdf)

The code will be uploaded soon.

## Citation
Please cite our paper for any purpose of usage.
```
@article{yeh2019improving,
  title={Improving Style Transfer with Calibrated Metrics},
  author={Yeh, Mao-Chuang and Tang, Shuai and Bhattad, Anand and Zou, Chuhang and Forsyth, David},
  journal={arXiv preprint arXiv:1910.09447},
  year={2019}
}
```
