## 🪨️ Making Retrieval-Augmented Language Models Robust to Irrelevant Context

### RetRobust Overview
By training RALMs on 1K examples we can make them robust to irrelevant context and improve QA performance
[**[Paper]**](http://arxiv.org/abs/2310.01558).

![Alt text](images/retrobust_fig_1.png?raw=true "Retrobust examples")


###  🤗 Data and Models
Our models and data are available at the [**RetRobust HuggingFace Collection**](https://huggingface.co/collections/Ori/retrobust-65198eef2b4fffcb4100e163).

### 🧗🏽 Experiments framework
LLama-2 inference servers were set using [**lm-sys/FastChat**](https://github.com/lm-sys/FastChat). Experiments were run using the framework from [**reasoning-on-cots**](https://github.com/oriyor/reasoning-on-cots). More details coming soon...

### ✍ Citation
```
bibtex
@misc{yoran2023making,
      title={Making Retrieval-Augmented Language Models Robust to Irrelevant Context}, 
      author={Ori Yoran and Tomer Wolfson and Ori Ram and Jonathan Berant},
      year={2023},
      eprint={2310.01558},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
