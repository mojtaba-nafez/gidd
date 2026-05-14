### 1. Performance Improvement




### 2. Sample Generation Step:
Normal Case:
```
sum total_changed: 512
sum mask_to_word: 512
sum word_to_mask: 0
sum word_to_word: 0
```
N4 Case:
```
sum total_changed: 517
sum mask_to_word: 513
sum word_to_mask: 1
sum word_to_word: 3
```

**GIDD statement:** GIDD’s strength does not lie in likelihood estimation or perplexity—on these metrics, MDLM may perform better. However, GIDD excels in text quality, achieving stronger results in generative perplexity as well as clarity, grammaticality, factuality, writing style, and creativity, as evaluated by GPT-4o.




### 3. Sample From  Hellaswag (Likelihood Base Benchmarking)

**Input:**  "Roof shingle removal: A man is sitting on a roof. He"
**Output**: " is using wrap to wrap a pair of skis."

**Input**:  "Roof shingle removal: A man is sitting on a roof. He"
**Output**: " is ripping level tiles off."

**Input**:  "Roof shingle removal: A man is sitting on a roof. He"
**Output**: " is holding a rubik's cube."

**Input**:  "Roof shingle removal: A man is sitting on a roof. He"
**Output**: " starts pulling up roofing on a roof."


### 4. Sample Checking Using GPT and Gemeni Conclusion(10 Sample): Self-Correction Step.

#### Without Compare to initial text:
Both are weak, but Original LLM barely refines the text and mostly preserves the corrupted original. N4 at least improves readability, structure, and grammar in several places, even though it introduces some risky meaning changes. So the stronger choice is LLM N4, but with low confidence because the source is extremely corrupted.

#### Without Compare to initial text:
LLM N4 shows more consistent improvement in readability and coherence, while LLM original only occasionally beats N4 on local fluency or conservatism. For practical refinement, N4 is the slightly more useful model, though both remain low-quality and unreliable.