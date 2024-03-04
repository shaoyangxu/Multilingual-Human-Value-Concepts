# Multilingual-Human-Value-Concepts

Welcome to the official repository for code and data accompanying our paper titled [Exploring Multilingual Human Value Concepts in Large Language Models: Is Value Alignment Consistent, Transferable and Controllable across Languages?](https://arxiv.org/abs/2402.18120)

# Preparing

Instead of collecting multilingual concept vectors and recognizing multilingual concepts manually, you can download precomputed concept vectors and concept recognition results of all concepts, languages and LLMs from [here](https://drive.google.com/drive/folders/1-wVEEx3luRDAjG-e531nRy_2Dd-s7yFa?usp=drive_link). To use these precomputed data, simply download the `data` and the `res` folders in the above link and put them into this directory.

# Collecting Multilingual Concept Vectors

Collect vectors for llama2-chat-7B as:
```bash
python collect_vector.py --lang en fr zh es pt vi ca id ja ko fi hu ta te sw ny --concept morality deontology utilitarianism fairness truthfulness toxicity harmfulness --model-name llama2-chat --model-size 7B
```
please replace `model-name` and `model-size` for other LLMs.

# Recognizing Multilingual Concepts

Perform concept recognition for llama2-chat-7B as:
```bash
python recognize_concept.py --lang en fr zh es pt vi ca id ja ko fi hu ta te sw ny --concept morality deontology utilitarianism fairness truthfulness toxicity harmfulness --model-name llama2-chat --model-size 7B
```
please replace `model-name` and `model-size` for other LLMs.

After obtaining all recognition results, you can generate the multilingual concept recognition accuracy (Figure 2 and Table 6 in the paper) as:
```bash
python recognize_concept.py --lang en fr zh es pt vi ca id ja ko fi hu ta te sw ny --concept morality deontology utilitarianism fairness truthfulness toxicity harmfulness --cross-model llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1
```

# Cross-lingual Consistency and Transferability Analysis

Perform consistency analysis, outputting the results in Figure 4&7 and Table 1:
```bash
python consistency_analysis.py --lang en fr zh es pt vi ca id ja ko fi hu ta te sw ny --concept morality deontology utilitarianism fairness truthfulness toxicity harmfulness --cross-model llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1
```

Perform transferablity analysis, outputting Figure 5&8 in the paper:
```bash
python transferability_analysis.py --lang en fr zh es pt vi ca id ja ko fi hu ta te sw ny --concept morality deontology utilitarianism fairness truthfulness toxicity harmfulness --cross-model llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1
```

# Cross-Lingual Value Alignment Control

TBC