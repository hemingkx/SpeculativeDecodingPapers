<div align="center">
  <h2><img src="assets/logo.png" height="28px"/><i>Unlocking Efficiency in Large Language Model Inference:</i><br>A Comprehensive Survey of Speculative Decoding</h2> 
</div>

<div align="center">
<b>Heming Xia</b><sup>1</sup>,
<b>Zhe Yang</b><sup>2</sup>,
<b>Qingxiu Dong</b><sup>2</sup>,
<b>Peiyi Wang</b><sup>2</sup>,
<b>Yongqi Li</b><sup>1</sup>,
<b>Tao Ge</b><sup>3</sup>,
<b>Tianyu Liu</b><sup>4</sup>,
<b>Wenjie Li</b><sup>1</sup>,
<b>Zhifang Sui</b><sup>2</sup>
</div>

<div align="center">
<sup>1</sup>Department of Computing, The Hong Kong Polytechnic University
</div>
<div align="center">
<sup>2</sup>National Key Laboratory for Multimedia Information Processing, Peking University
</div>
<div align="center">
<sup>3</sup>Microsoft Research Asia <sup>4</sup>Alibaba Group
</div>

![timeline](./assets/timeline.png)

This repository contains a regularly updated paper list for **Speculative Decoding**.

[![Arxiv](https://img.shields.io/badge/Arxiv-2401.07851-orange.svg)](https://arxiv.org/abs/2401.07851) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](./LICENSE) ![GitHub last commit (branch)](https://img.shields.io/github/last-commit/hemingkx/SpeculativeDecodingPapers/main?logo=github&color=blue)

## Content

- [Keywords Convention](#keywords-convention)
- [Papers](#papers)
  - [Survey](#survey)
  - [Speculative Decoding for Seq2Seq](#speculative-decoding-for-seq2seq)
  - [Speculative Decoding for LLMs](#speculative-decoding-for-llms)
  - [Applications](#applications)
  - [Benchmarks](#benchmarks)
  - [Analysis](#analysis)
- [Blogs](#blogs)
- [Contribution](#contribution)
  - [Contributors](#contributors)
  - [Contributing to this paper list](#contributing-to-this-paper-list)
- [Citation](#citation)

## Keywords Convention

![](https://img.shields.io/badge/SpecDec-blue) Abbreviation

![](https://img.shields.io/badge/ACL2022-orange) Conference

![](https://img.shields.io/badge/Drafter:_small_LM-green) Drafting Methods in Speculative Decoding

![](https://img.shields.io/badge/Batching-lightgray) Main Features

## Papers

### Survey

- **Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding**  
  *Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui.* [[pdf](https://arxiv.org/pdf/2401.07851.pdf)], [[code](https://github.com/hemingkx/SpeculativeDecodingPapers)], 2024.01. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Survey_on_Speculative_Decoding-lightgray)

### Speculative Decoding for Seq2Seq

- **Blockwise Parallel Decoding for Deep Autoregressive Models**  
  *Mitchell Stern, Noam Shazeer, Jakob Uszkoreit*. [[pdf](https://arxiv.org/pdf/1811.03115.pdf)], 2018.11. ![](https://img.shields.io/badge/NIPS2018-orange) ![](https://img.shields.io/badge/Self--Draft:_specialized_FFN_heads-green) ![](https://img.shields.io/badge/Blockwise_Decoding-blue)

- **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**  
  *Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui*. [[pdf](https://arxiv.org/abs/2203.16487)], [[code](https://github.com/hemingkx/SpecDec)], 2022.03. ![](https://img.shields.io/badge/EMNLP2023--findings-orange) ![](https://img.shields.io/badge/Drafter:_specialized_Non--Auto_LM-green) ![](https://img.shields.io/badge/SpecDec-blue)

- **Speculative Decoding with Big Little Decoder**  
  *Sehoon Kim, Karttikeya Mangalam, Suhong Moon, John Canny, Jitendra Malik, Michael W. Mahoney, Amir Gholami, Kurt Keutzer*. [[pdf](https://arxiv.org/pdf/2302.07863.pdf)], [[code](https://github.com/kssteven418/BigLittleDecoder)], 2023.02. ![](https://img.shields.io/badge/NIPS2023-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/BiLD-blue)

- **Accelerating Transformer Inference for Translation via Parallel Decoding**  
  *Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, Emanuele Rodolà*. [[pdf](https://aclanthology.org/2023.acl-long.689.pdf)], 2023.05. ![](https://img.shields.io/badge/ACL2023-orange) ![](https://img.shields.io/badge/Self--Draft:_mask--predict-green)
  
- **SPEED: Speculative Pipelined Execution for Efficient Decoding**  
  *Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt Keutzer, Amir Gholami, Sophia Shao*. [[pdf](https://arxiv.org/pdf/2310.12072.pdf)], 2023.10. ![](https://img.shields.io/badge/ENLSP_at_NIPS2023-orange) ![](https://img.shields.io/badge/Self--Draft:_early--existing-green) ![](https://img.shields.io/badge/SPEED-blue)

### Speculative Decoding for LLMs

- **Fast Inference from Transformers via Speculative Decoding**  
  *Yaniv Leviathan, Matan Kalman, Yossi Matias*. [[pdf](https://arxiv.org/pdf/2211.17192.pdf)], 2022.11. ![](https://img.shields.io/badge/ICML2023--Oral-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green)

- **Accelerating Large Language Model Decoding with Speculative Sampling**  
  *Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper*. [[pdf](http://arxiv.org/abs/2302.01318)], 2023.02. ![](https://img.shields.io/badge/Technical_Report-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/SpS-blue)

- **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification**  
  *Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia.* [[pdf](https://arxiv.org/pdf/2305.09781.pdf)], [[code](https://github.com/flexflow/FlexFlow/)], 2023.05. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_boost--tuned_small_LMs-green) ![](https://img.shields.io/badge/SpecInfer-blue)

- **Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding**  
  *Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, Kangwook Lee*. [[pdf](https://arxiv.org/pdf/2307.05908.pdf)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Self--Draft:_early--existing-green) ![](https://img.shields.io/badge/PPD-blue)

- **Accelerating LLM Inference with Staged Speculative Decoding**  
  *Benjamin Spector, Chris Re*. [[pdf](https://arxiv.org/pdf/2308.04623.pdf)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green)

- **SpecTr: Fast Speculative Decoding via Optimal Transport**   
  *Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix Yu, Michael Riley, Sanjiv Kumar*. [[pdf](https://openreview.net/pdf?id=d0mGsaheuT)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/SpecTr-blue)

- **Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding**  
  *Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra*. [[pdf](https://arxiv.org/pdf/2309.08168.pdf)], [[code](https://github.com/dilab-zju/self-speculative-decoding)], 2023.09. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_early--existing-green) ![](https://img.shields.io/badge/Self--Speculative-blue)

- **Online Speculative Decoding**  
  *Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Ion Stoica, Zhijie Deng, Alvin Cheung, Hao Zhang*. [[pdf](https://arxiv.org/pdf/2310.07177.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Online_Distillation_Strategy-lightgray)

- **DistillSpec: Improving Speculative Decoding via Knowledge Distillation**  
  *Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, Rishabh Agarwal.* [[pdf](https://arxiv.org/pdf/2310.08461.pdf)], 2023.10. ![](https://img.shields.io/badge/ICLR_2024-orange) ![](https://img.shields.io/badge/Drafter:_distilled_small_LM-green) ![](https://img.shields.io/badge/DistillSpec-blue) ![](https://img.shields.io/badge/Distillation_Strategy-lightgray)

- **REST: Retrieval-Based Speculative Decoding**  
  *Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D Lee, Di He.* [[pdf](https://arxiv.org/pdf/2311.08252.pdf)], [[code](https://github.com/FasterDecoding/REST)], 2023.11. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Draft:_retrieved_Docs-green) ![](https://img.shields.io/badge/REST-blue)

- **Speculative Contrastive Decoding**  
  *Hongyi Yuan, Keming Lu, Fei Huang, Zheng Yuan, Chang Zhou.* [[pdf](https://arxiv.org/pdf/2311.08981.pdf)], 2023.11. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/SCD-blue)

- **PaSS: Parallel Speculative Sampling**  
  *Giovanni Monea, Armand Joulin, Edouard Grave.* [[pdf](https://arxiv.org/pdf/2311.13581.pdf)], 2023.11. ![](https://img.shields.io/badge/ENLSP_at_NIPS2023-orange) ![](https://img.shields.io/badge/Self--Draft:_prompt--tuning-green) ![](https://img.shields.io/badge/PaSS-blue)

- **Cascade Speculative Drafting for Even Faster LLM Inference**  
  *Ziyi Chen, Xiaocong Yang, Jiacheng Lin, Chenkai Sun, Jie Huang, Kevin Chen-Chuan Chang.* [[pdf](https://arxiv.org/pdf/2312.11462.pdf)], 2023.12. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_cascaded_LMs-green)

- **Multi-Candidate Speculative Decoding**  
  *Sen Yang, Shujian Huang, Xinyu Dai, Jiajun Chen.* [[pdf](https://arxiv.org/pdf/2401.06706.pdf)], [[code](https://github.com/NJUNLP/MCSD)], 2024.01. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green)

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**  
  *Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao.* [[pdf](https://arxiv.org/pdf/2401.10774.pdf)], [[code](https://github.com/FasterDecoding/Medusa)], 2024.01. ![](https://img.shields.io/badge/Technical_Report-orange) ![](https://img.shields.io/badge/Self--Draft:_specialized_FFN_heads-green) ![](https://img.shields.io/badge/Medusa-blue)

- **BiTA: Bi-Directional Tuning for Lossless Acceleration in Large Language Models**  
  *Feng Lin, Hanling Yi, Hongbin Li, Yifan Yang, Xiaotian Yu, Guangming Lu, Rong Xiao*. [[pdf](https://arxiv.org/pdf/2401.12522.pdf)], [[code](https://github.com/linfeng93/BiTA)], 2024.01. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_prompt--tuning-green) ![](https://img.shields.io/badge/BiTA-blue)
  
- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**  
  *Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang*. [[pdf](https://arxiv.org/pdf/2401.15077.pdf)], [[code](https://github.com/SafeAILab/EAGLE)], 2024.01. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_auto--regression_heads-green) ![](https://img.shields.io/badge/EAGLE-blue)
  
- **GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding**  
  *Cunxiao Du, Jing Jiang, Xu Yuanchen, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, Yang You*. [[pdf](https://arxiv.org/pdf/2402.02082.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/GLIDE-blue)
  
- **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**  
  *Yichao Fu, Peter Bailis, Ion Stoica, Hao Zhang*. [[pdf](https://arxiv.org/pdf/2402.02057.pdf)], [[code](https://github.com/hao-ai-lab/LookaheadDecoding)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_mask--predict-green) ![](https://img.shields.io/badge/Lookahead-blue)
  
- **Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding**  
  *Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, William Brandon*. [[pdf](https://arxiv.org/pdf/2402.05109.pdf)], [[code](https://github.com/zankner/hydra)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_specialized_FFN_heads-green) ![](https://img.shields.io/badge/Hydra-blue)
  
- **Speculative Streaming: Fast LLM Inference without Auxiliary Models**  
  *Nikhil Bhendawade, Irina Belousova, Qichen Fu, Henry Mason, Mohammad Rastegari, Mahyar Najibi*. [[pdf](https://arxiv.org/pdf/2402.11131.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_multi--stream_attention-green) 

- **Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding**  
  *Hanling Yi, Feng Lin, Hongbin Li, Peiyang Ning, Xiaotian Yu, Rong Xiao*. [[pdf](https://arxiv.org/pdf/2402.11809.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:_semi--autoregressive_finetuning-green) ![](https://img.shields.io/badge/SPACE-blue)

- **Sequoia: Scalable, Robust, and Hardware-aware Speculative Decoding**  
  *Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, Beidi Chen*. [[pdf](https://arxiv.org/pdf/2402.12374.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Token_Tree_Optimization-lightgray) ![](https://img.shields.io/badge/Sequoia-blue)
  
- **ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding**  
  *Shuzhang Zhong, Zebin Yang, Meng Li, Ruihao Gong, Runsheng Wang, Ru Huang*. [[pdf](https://arxiv.org/pdf/2402.13485.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Token_Tree_Pruning_&_Generation-lightgray) ![](https://img.shields.io/badge/ProPD-blue)
  
- **Ouroboros: Speculative Decoding with Large Model Enhanced Drafting**  
  *Weilin Zhao, Yuxiang Huang, Xu Han, Chaojun Xiao, Zhiyuan Liu, Maosong Sun*. [[pdf](https://arxiv.org/pdf/2402.13720.pdf)], [[code](https://github.com/thunlp/Ouroboros)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Phrase_Candidate_Pool-lightgray) ![](https://img.shields.io/badge/Ouroboros-blue)

### Applications

- **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**  
  *Xin Sun, Tao Ge, Furu Wei, Houfeng Wang*. [[pdf](https://aclanthology.org/2021.acl-long.462.pdf)], [[code](https://github.com/AutoTemp/Shallow-Aggressive-Decoding)], 2021.07. ![](https://img.shields.io/badge/ACL2021-orange) ![](https://img.shields.io/badge/Draft:_original_input-green) ![](https://img.shields.io/badge/SAD-blue)

- **Inference with Reference: Lossless Acceleration of Large Language Models**  
  *Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei.* [[pdf](https://arxiv.org/pdf/2304.04487.pdf)], 2023.04. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Draft:_the_reference-green) ![](https://img.shields.io/badge/LLMA-blue)
  
- **LLMCad: Fast and Scalable On-device Large Language Model Inference**  
  *Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu.* [[pdf](https://arxiv.org/pdf/2309.04255.pdf)], 2023.09. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/LLMCad-blue)
  
- **Accelerating Retrieval-Augmented Language Model Serving with Speculation**  
  *Zhihao Zhang, Alan Zhu, Lijie Yang, Yihua Xu, Lanting Li, Phitchaya Mangpo Phothilimthana, Zhihao Jia.* [[pdf](https://arxiv.org/pdf/2401.14021.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/RaLMSpec-blue)
  
- **Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy**  
  *Yao Zhao, Zhitian Xie, Chenyi Zhuang, Jinjie Gu.* [[pdf](https://arxiv.org/pdf/2312.12728.pdf)], [[code](https://github.com/alipay/PainlessInferenceAcceleration)], 2023.12. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Draft:_retrieved_Docs-green)
  
- **A Simple Framework to Accelerate Multilingual Language Model for Monolingual Text Generation**  
  *Jimin Hong, Gibbeum Lee, Jaewoong Cho.* [[pdf](https://arxiv.org/pdf/2401.10660.pdf)], 2024.01. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_specialized_FFN_heads-green) ![](https://img.shields.io/badge/MuMo-blue)![](https://img.shields.io/badge/Korean_&_Japanese-lightgray)

### Benchmarks

- **Spec-Bench: A Comprehensive Benchmark for Speculative Decoding**  
  *Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, Zhifang Sui.* [[pdf](https://arxiv.org/pdf/2401.07851.pdf)], [[code](https://github.com/hemingkx/Spec-Bench)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Spec--Bench-blue)

### Analysis

- **The Synergy of Speculative Decoding and Batching in Serving Large Language Models**  
  *Qidong Su, Christina Giannoula, Gennady Pekhimenko.* [[pdf](https://arxiv.org/pdf/2310.18813.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Optimal_Speculation_Length_in_Batching-lightgray)
  
- **Decoding Speculative Decoding**  
  *Minghao Yan, Saurabh Agarwal, Shivaram Venkataraman.* [[pdf](https://arxiv.org/pdf/2402.01528.pdf)], 2024.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter_Selection_to_Maximize_Throughput-lightgray)

## Blog & Project

**Assisted Generation: a new direction toward low-latency text generation.** Huggingface. 2023.05. [[Blog](https://huggingface.co/blog/assisted-generation)] [[Code](https://github.com/huggingface/transformers/blob/849367ccf741d8c58aa88ccfe1d52d8636eaf2b7/src/transformers/generation/utils.py#L4064)]

**Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads.** Princeton, UIUC. 2023.09. [[Blog](https://sites.google.com/view/medusa-llm)] [[Code](https://github.com/FasterDecoding/Medusa)]

**Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.** LMSys. 2023.11. [[Blog](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)] [[Code](https://github.com/hao-ai-lab/LookaheadDecoding)]

**Accelerating Generative AI with PyTorch II: GPT, Fast.** Pytorch. 2023.11. [[Blog](https://pytorch.org/blog/accelerating-generative-ai-2/)] [[Code](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L76)]

**Prompt Lookup Decoding.** Apoorv Saxena. 2023.11. [[Code](https://github.com/apoorvumang/prompt-lookup-decoding)] [[Colab](https://colab.research.google.com/drive/1ovjH1sg3lXWdm5Rx5EEukB9H_PFJVpJ4?usp=sharing)]

**EAGLE: Lossless Acceleration of LLM Decoding by Feature Extrapolation.** Vector Institute, University of Waterloo, Peking University. 2023.12. [[Blog](https://sites.google.com/view/eagle-llm)] [[Code](https://github.com/SafeAILab/EAGLE)]

## Contributors

<a href="https://github.com/hemingkx/SpeculativeDecodingPapers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hemingkx/SpeculativeDecodingPapers" />
</a>

## Contributing to this paper list

-  There are cases where we miss important works in this field, please feel free to contribute and promote your awesome work or other related works here! Thanks for the efforts in advance.

## Citation

If you find the resources in this repository useful, please cite our paper:

```
@misc{xia2024unlocking,
      title={Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding}, 
      author={Heming Xia and Zhe Yang and Qingxiu Dong and Peiyi Wang and Yongqi Li and Tao Ge and Tianyu Liu and Wenjie Li and Zhifang Sui},
      year={2024},
      eprint={2401.07851},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

