# Hands-on Tutorial of Once-for-All Network [[youtube]](https://www.youtube.com/watch?v=wrsid5tvuSM), [[bilibili]](https://www.bilibili.com/video/BV1oK411p797/)

This is a hands-on tutorial for **Once for All: Train One Network and Specialize it for Efficient Deployment** [[arXiv]](https://arxiv.org/abs/1908.09791).

```bibtex
@inproceedings{
  cai2020once,
  title={Once for All: Train One Network and Specialize it for Efficient Deployment},
  author={Han Cai and Chuang Gan and Tianzhe Wang and Zhekai Zhang and Song Han},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/pdf/1908.09791.pdf}
}
```

In this notebook, we will demonstrate 
- how to use pretrained specialized OFA sub-networks for efficient inference on diverse hardware platforms
- how to get new specialized neural networks on ImageNet with the OFA network within minutes.

Required packages:
```bash
pip install --upgrade pip
pip install --upgrade jupyter notebook
```

Then, please clone this repository to your computer using:

```bash
git clone https://github.com/mit-han-lab/once-for-all.git
```

After cloning is finished, you may go to the directory of this tutorial and run

```bash
jupyter notebook --port 8888
```

to start a jupyter notebook and access it through the browser. Finally, let's explore the notebook `ofa.ipynb` prepared by us!


