# Nash-Detect

Code for KDD 2020 paper **Robust Spammer Detection by Nash Reinforcement Learning**.  
[Yingtong Dou](http://ytongdou.com/), [Guixiang Ma](https://scholar.google.com/citations?user=CbKihaUAAAAJ&hl=en), [Philip S. Yu](https://www.cs.uic.edu/PSYu/), [Sihong Xie](http://www.cse.lehigh.edu/~sxie/).  
\[[Paper](http://arxiv.org/abs/2006.06069)\]\[[Slides](http://ytongdou.com/files/kdd20slides.pdf)\]\[[Video](https://youtu.be/Pa13fabSGVw)\]\[[Toolbox](https://github.com/safe-graph/UGFraud)\]\[[Chinese Blog](https://mp.weixin.qq.com/s?__biz=MzU1Mjc5NTg5OQ==&mid=2247485268&idx=1&sn=451d137496829d1405c28808ec0bb0b2&chksm=fbfdecc0cc8a65d651600a437043cbb5483bfa16e35c3b1cb3ca60d7ac47e64a3bdd514e357f&token=1158859151&lang=zh_CN#rd)\]

## Overview

<p align="center">
    <br>
    <a href="https://github.com/YingtongDou/Nash-Detect">
        <img src="https://github.com/YingtongDou/Nash-Detect/blob/master/overview.png" width="600"/>
    </a>
    <br>
<p>

**Nash-Detect** is an algorithm proposed by the above paper to train a robust spam review detector using reinforcement learning. The robust detector is composed of five base detectors and is trained through playing a minimax game between the spammer and the defender. There are five base spamming strategies used by the spammer to synthesize the mixed spamming strategy.

This repo includes the spamming attack implementation and generation code, the detector implementation code, and the training & testing code for Nash-Detect and all baselines. 

Note that we only investigate the shallow graph and behavior-based spam detectors in this paper; there is no text or deep neural network involved. Nonetheless, there is no hurdle to apply Nash-Detect to train robust neural networks or text-based spam detectors.

## Setup

To run the code, you need the [Yelp Spam Review Datasets](http://odds.cs.stonybrook.edu/yelpchi-dataset/). Please send email with the title `Yelp Dataset Request` to [ytongdou@gmail.com](mailto:ytongdou@gmail.com) to download the file with metadata and ground truth. You can unzip the dataset file under the root directory of the project.

You can download the project and install required packages using following commands:

```bash
git clone https://github.com/YingtongDou/Nash-Detect.git
cd Nash-Detect
pip3 install -r requirements.txt
```

To run the code, you need to have **Python 3.6** or later version. 

## Running

1. Run `attack_generation.py` with `mode = "Training"` to generate fake reviews for training
2. Run `worst_case.py` to compute the worst-case performance of single attacks vs. single detectors
3. Run `training.py` to train a robust detector configuration using Nash-Detect
4. Run `attack_generation.py` with `mode = "Testing"` to generate fake reviews for testing
5. Run `testing.py` to test the performance of the optimal detector trained by Nash-Detect and other baselines

To facilitate the training and testing, we have stored all generated fake reviews in directories `/Training` and `/Testing`. So you can skip Step 1 and 4 to play the game and evaluation code directly. Moreover, you can play each single detector using the `eval_XXX.py` under the `/Detector` repository or using our [UGFraud](https://github.com/safe-graph/UGFraud) toolbox.

To experimental settings and model parameters can be found at the beginning of the `main` functions of `training.py` and `testing.py`.

### Repo Structure
The repository is organized as follows:
- `Attack/` contains the implementations of four spamming attack strategies, the `Singleton` attack is implemented in `attack_generation.py`;
- `Detector/` contains the implementations and evaluations of five spam detectors;
- `Testing/` contains generated fake reviews for testing;
- `Training/` contains generated fake reviews for training;
- `Utils/` contains:
    * functions for loading graphs/features from dataset/manifest files  (`iohelper.py`);
    * utility functions for training and testing (`eval_helper.py`);
    * functions for extracting and updating features and prior beliefs (`yelpFeatureExtraction.py`);
    * the manifest file for features (`feature_configuration.py`).

## Citation
```bibtex
@inproceedings{dou2020robust,
  title={Robust Spammer Detection by Nash Reinforcement Learning},
  author={Dou, Yingtong and Ma, Guixiang and Yu, Philip S and Xie, Sihong},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  year={2020}
}
```
