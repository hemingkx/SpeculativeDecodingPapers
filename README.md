<div style="text-align: center;">
    <h1><img src="assets/logo.png" height="28px" /> Speculative Decoding Papers </h1>
</div>


This is a paper list about **Speculative Decoding**.

## Content

- [Keywords Convention](#keywords-convention)
- [Papers](#papers)
  - [Speculative Decoding for Seq2Seq](#speculative-decoding-for-seq2seq)
  - [Speculative Decoding for LLMs](#speculative-decoding-for-llms)
  - [Applications](#applications)
- [Blogs](#blogs)
- [Contribution](#contribution)
  - [Contributors](#contributors)
  - [Contributing to this paper list](#contributing-to-this-paper-list)

## Keywords Convention

![](https://img.shields.io/badge/SpecDec-blue) Abbreviation

![](https://img.shields.io/badge/ACL2022-orange) Conference

![](https://img.shields.io/badge/Drafter:_small_LM-green) Drafting Methods in Speculative Decoding

## Papers

### Speculative Decoding for Seq2Seq

- **Blockwise Parallel Decoding for Deep Autoregressive Models**

  *Mitchell Stern, Noam Shazeer, Jakob Uszkoreit*. [[pdf](https://arxiv.org/pdf/1811.03115.pdf)], 2018.11. ![](https://img.shields.io/badge/NIPS2018-orange) ![](https://img.shields.io/badge/Drafter:_tuned_FFN_heads-green) ![](https://img.shields.io/badge/Blockwise_Decoding-blue)

- **Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation**

  *Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, Zhifang Sui*. [[pdf](https://arxiv.org/abs/2211.17192], 2022.03. ![](https://img.shields.io/badge/EMNLP2023--findings-orange) ![](https://img.shields.io/badge/Drafter:_specialized_NAT-green) ![](https://img.shields.io/badge/SpecDec-blue)

- **Speculative Decoding with Big Little Decoder**

  *Sehoon Kim, Karttikeya Mangalam, Suhong Moon, John Canny, Jitendra Malik, Michael W. Mahoney, Amir Gholami, Kurt Keutzer*. [[pdf](https://arxiv.org/pdf/2302.07863.pdf)], 2023.02. ![](https://img.shields.io/badge/NIPS2023-orange) ![](https://img.shields.io/badge/BiLD-blue)

- **Accelerating Transformer Inference for Translation via Parallel Decoding**

  *Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, Emanuele Rodolà*. [[pdf](https://aclanthology.org/2023.acl-long.689.pdf)], 2023.05. ![](https://img.shields.io/badge/ACL2023-orange) ![](https://img.shields.io/badge/Self--Draft:mask--predict-green)
  
- **SPEED: Speculative Pipelined Execution for Efficient Decoding**

  *Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt Keutzer, Amir Gholami, Sophia Shao*. [[pdf](https://arxiv.org/pdf/2310.12072.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:early--existing-green) ![](https://img.shields.io/badge/SPEED-blue)

### Speculative Decoding for LLMs

- **Fast Inference from Transformers via Speculative Decoding**

  *Yaniv Leviathan, Matan Kalman, Yossi Matias*. [[pdf](https://arxiv.org/pdf/2308.00675.pdf)], 2022.11. ![](https://img.shields.io/badge/ICML2023--Oral-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green)

- **Accelerating Large Language Model Decoding with Speculative Sampling**

  *Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper*. [[pdf](http://arxiv.org/abs/2302.01318)], 2023.02. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/SpS-blue)

- **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification**

  *Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, Zhihao Jia.* [[pdf](https://arxiv.org/abs/2305.09781)], 2023.05. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_boost--tuned_small_LMs-green) ![](https://img.shields.io/badge/SpecInfer-blue)

- **Predictive Pipelined Decoding: A Compute-Latency Trade-off for Exact LLM Decoding**

  *Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, Kangwook Lee*. [[pdf](https://arxiv.org/pdf/2307.05908.pdf)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Self--Draft:early--existing-green) ![](https://img.shields.io/badge/PPD-blue)

- **Accelerating LLM Inference with Staged Speculative Decoding**

  *Benjamin Spector, Chris Re*. [[pdf](https://arxiv.org/abs/2308.04623)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green)

- **SpecTr: Fast Speculative Decoding via Optimal Transport** 

  *Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, Felix Yu, Michael Riley, Sanjiv Kumar*. [[pdf](https://openreview.net/pdf?id=d0mGsaheuT)], 2023.08. ![](https://img.shields.io/badge/ES--FOMO_at_ICML2023-orange) ![](https://img.shields.io/badge/Drafter:_smaller_LM-green) ![](https://img.shields.io/badge/SpecTr-blue)

- **Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding**

  *Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Sharad Mehrotra*. [[pdf](https://arxiv.org/pdf/2309.08168.pdf)], 2023.09. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Self--Draft:early--existing-green) ![](https://img.shields.io/badge/Self--Speculative-blue)
  
- **Online Speculative Decoding**

  *Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Ion Stoica, Zhijie Deng, Alvin Cheung, Hao Zhang*. [[pdf](https://arxiv.org/pdf/2310.07177.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange)
  
- **DistillSpec: Improving Speculative Decoding via Knowledge Distillation**

  *Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, Rishabh Agarwal.* [[pdf](https://arxiv.org/pdf/2310.08461.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Drafter:_distilled_LM-green) ![](https://img.shields.io/badge/DistillSpec-blue)

### Applications

- **Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding**

  *Xin Sun, Tao Ge, Furu Wei, Houfeng Wang*. [[pdf](https://aclanthology.org/2021.acl-long.462.pdf)], 2021.07. ![](https://img.shields.io/badge/ACL2021-orange) ![](https://img.shields.io/badge/Draft:_original_input-green) ![](https://img.shields.io/badge/SAD-blue)

- **Inference with Reference: Lossless Acceleration of Large Language Models**

  *Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, Furu Wei.* [[pdf](https://arxiv.org/pdf/2304.04487.pdf)], 2023.04. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/Draft:_the_reference-green) ![](https://img.shields.io/badge/LLMA-blue)
  
- **LLMCad: Fast and Scalable On-device Large Language Model Inference**

  *Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu.* [[pdf](https://arxiv.org/pdf/2309.04255.pdf)], 2023.09. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/LLMCad-blue)
  
- **Accelerating Retrieval-augmented Language Model Serving with Speculation**

  *ICLR 2024 Conference Submission.* [[pdf](https://openreview.net/forum?id=vkzPuZJ80a)], 2023.10. ![](https://img.shields.io/badge/ICLR2024--submission-orange) ![](https://img.shields.io/badge/RaLMSpec-blue)
  
- **An Emulator for Fine-Tuning Large Language Models using Small Language Models**

  *Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, Christopher D. Manning*. [[pdf](https://arxiv.org/pdf/2310.12962.pdf)], 2023.10. ![](https://img.shields.io/badge/Arxiv-orange) ![](https://img.shields.io/badge/EFT-blue)

## Blog & Project

**Assisted Generation: a new direction toward low-latency text generation**. Huggingface 2023.05. [[Link](https://huggingface.co/blog/assisted-generation)]

**Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads.** 2023.09. [[Project]](https://github.com/FasterDecoding/Medusa)

## Contributors

<a href="https://github.com/hemingkx/SpeculativeDecodingPapers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=hemingkx/SpeculativeDecodingPapers" />
</a>

## Contributing to this paper list

-  There are cases where we miss important works in this field, please feel free to contribute and promote your awesome work or other related works here! Thanks for the efforts in advance.
