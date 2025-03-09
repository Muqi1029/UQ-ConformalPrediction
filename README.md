# Uncertainty Quantification for Large Language Models in the Medical Domain



## :tada: News

:partying_face: We’re thrilled to announce the launch of our demo for quantifying uncertainty in LLM outputs within the medical domain.

Now, you can upvote**👍** or downvote**👎** individual sentences based on their confidence. Your feedback is invaluable to our team and plays a crucial role in helping us make LLMs more **reliable and trustworthy**.

Thank you for contributing to our mission of advancing uncertainty quantification in large language models!



## :robot:Model

### HuatuoGPT2-7B

HuatuoGPT2 leverages an innovative domain adaptation approach to significantly enhance its medical knowledge and dialogue capabilities. It achieves state-of-the-art performance across various medical benchmarks, notably outperforming GPT-4 in expert evaluations and recent medical licensing exams.

Explore the model [here](https://huggingface.co/FreedomIntelligence/HuatuoGPT2-7B).



## :books:Dataset

### MedQA

**Multiple Choice Question Answering Based on USMLE**

This dataset is derived from professional medical board exams, specifically the United States Medical License Exams (USMLE). It spans three languages—English, Simplified Chinese, and Traditional Chinese—containing 12,723, 34,251, and 14,123 questions respectively.

The data is structured as follows:

```json
{'answer': '非甾体抗炎药',
 'answer_idx': 'D',
 'meta_info': '第一部分\u3000历年真题',
 'options': {'A': '苯溴马隆', 'B': '别嘌呤醇', 'C': '抗生素', 'D': '非甾体抗炎药', 'E': '甲氟蝶呤'},
 'question': '男，50岁。吃海鲜后夜间突发左足第一跖趾关节剧烈疼痛1天。查体：关节局部红肿，'}
```

Explore the dataset [here](https://paperswithcode.com/dataset/medqa-usmle), or download it directly using this [link](https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view).



## :100:Confidence Scoring

 **1. Creating the Calibration Dataset**

- **Benchmarking**: Evaluate the model on the **MedQA dataset**.
- **Selection**: Retain only those questions the model successfully answers.
- **Log Probability**: Compute the model's log probabilities for the retained questions.
- **Simulating Distribution**: Use these probabilities to model the distribution of the random variable representing the likelihood of correct answers, forming the calibration dataset.



**2.  Inference and Confidence Scoring**

- **Assumption**: The model's probability of answering correctly follows an **i.i.d. distribution**.
- **New Instance Prediction**: Each new instance is expected to align with this distribution.
- **Log Probability Computation**: After generating a response, compute its log probability.



**Confidence Score**: Locate the computed log probability within the calibration dataset's distribution and output its percentile as the final confidence score.



## :first_quarter_moon:TODO List

- [x] **Basic Demo**: Generate confidence scores using the log probability of outputs.

- [x] **LLM Output Streaming**: Implement real-time streaming of model outputs.

- [ ] **Enhanced Algorithms**: Design more rigorous algorithms or train a model to generate confidence scores effectively.



## :email:Reach Out

If you’re interested in exploring uncertainty quantification in this form and have any questions, feel free to reach out to me at **muqi1029@gmail.com**.



## :book:Citation

```
@misc{li2024uncertainty,
  author = {Muqi Li},
  title = {Uncertainty Quantification for Large Language Models in the Medical Domain},
  year = {2024},
  howpublished = {https://github.com/Muqi1029/UQ-CP},
  note = {Contact: muqi1029@gmail.com}
}
```
