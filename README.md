# Out-Of-Distribution Is Not a Magic: The Clash Between Rejection Rate and Model Success


This repository contains code and results for the research paper "Out-Of-Distribution Is Not a Magic: The Clash Between Rejection Rate and Model Success". This study aimed to evaluate the performance of different out-of-distribution (OOD) detection techniques in the context of network traffic classification. 

## Abstract
Network traffic classification plays a crucial role in ensuring the security and efficiency of computer networks. However, unknown or out-of-distribution traffic poses a significant challenge to accurate classification. In this study, we explore the effectiveness of three OOD detection methods - ODIN, GradBP, and K+1 - in enhancing the performance of traffic classification models. We conduct experiments on both binary (benign vs. malicious) and multiclass (well-known applications) classification tasks and evaluate the trade-offs between accuracy and rejection rates. Our findings highlight the task-dependent nature of OOD detection and identify the most suitable method for each classification scenario.


## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [How to Use](#how-to-use)
- [References](#references)

## Introduction
Network traffic classification is essential for various network management tasks, including intrusion detection, quality of service, and traffic engineering. However, accurately classifying traffic becomes challenging when confronted with unknown or out-of-distribution samples. In this study, we investigate the effectiveness of three OOD detection techniques - ODIN, GradBP, and K+1 - in improving the performance of traffic classification models. We evaluate these methods on both binary and multiclass classification tasks and analyze the trade-offs between accuracy and rejection rates.

## Dataset
We used a publicly available network traffic dataset for our experiments. The dataset includes labeled traffic samples representing both benign and malicious traffic as well as well-known applications.

## Experimental Setup
Our experiments were conducted on a machine with an AMD Ryzen 5 1600 SixCore Processor, 16 GB RAM, and an NVIDIA GeForce GTX 1060 6GB GPU. We performed two main sets of experiments: 
1. Base model evaluation: We evaluated the base classification model without OOD detection on a dataset of known classes, both in the binary (benign vs. malicious) and multiclass (well-known applications) tasks.
2. OOD detection evaluation: We assessed the performance of the base model with OOD detection techniques on a dataset containing both known and unknown classes. We implemented ODIN, GradBP, and K+1 methods and evaluated their effectiveness at different rejection rates.

## Results
The results of our experiments, including confusion matrices, tables, and threshold graphs, can be found the images section. 

- ODIN demonstrated promising results in both binary and multiclass classification tasks. However, it exhibited weaknesses in detecting misclassified known applications as OOD samples.
- GradBP performed well in the binary classification task, outperforming ODIN in various metrics. It effectively rejected OOD samples while maintaining high accuracy rates.
- K+1, which introduced a rejection class, showed mixed results. It achieved high purity rates but lower accuracy compared to ODIN and GradBP.
- The choice of rejection rate had a significant impact on the performance of OOD detection methods. GradBP consistently improved as the rejection rate increased, indicating a higher likelihood of accurately classifying passed samples.
- Different methods showed varying performance depending on the classification task and the metrics evaluated.

## Conclusion
Our study demonstrates the efficacy of ODIN, GradBP, and K+1's efficacy in enhancing traffic classification models' performance with OOD detection. The choice of the most suitable method depends on the classification task. ODIN showed superior performance in multiclass classification, while GradBP emerged as the preferable choice for binary classification. K+1 exhibited its own strengths but also limitations. Overall, the rejection rate was crucial in balancing accuracy and data loss.

## Future Work
In future research, we aim to explore the performance of multiple models and additional OOD detection methods in different domains. By conducting comprehensive experiments and evaluating various models, we can better understand the nuances of network traffic classification and potentially identify novel techniques for improved performance across diverse classification tasks.

## How to Use
Running this code requires the datasets we use, which are not public as of right now.
