# InkBench

InkBench is a corpus designed to evaluate the ability of multimodal LLMs with vision inputs to transcribe historical documents. 


**Character Accuracy Results (1-CER).** *September 10, 2025.* 

|                                     | Overall | Book Page | Handwritten | Mixed | Other Typed/Printed | Cost per 1K |
|:-----------------------------------:|:-------:|:---------:|:-----------:|:-----:|:-------------------:|:-----------:|
| openai-gpt-4.1-mini                 | 0.88    | 0.88      | 0.91        | 0.83  | 0.93                | $0.11       |
| google-gemini-2.5-flash             | 0.87    | 0.87      | 0.88        | 0.84  | 0.89                | $1.22       |
| openai-gpt-5-mini                   | 0.87    | 0.86      | 0.89        | 0.82  | 0.91                | $0.13       |
| openai-gpt-5                   | 0.84    | 0.82      | 0.88        | 0.80  | 0.85                | $1.88       |
| google-gemini-2.5-flash-lite        | 0.83    | 0.82      | 0.83        | 0.82  | 0.85                | $0.30       |
| qwen-qwen2.5-vl-72b-instruct        | 0.82    | 0.89      | 0.76        | 0.83  | 0.71                | $4.47       |
| mistralai-mistral-medium-3.1        | 0.80    | 0.82      | 0.81        | 0.75  | 0.84                | $4.21       |
| openai-gpt-4o-mini                  | 0.78    | 0.80      | 0.80        | 0.67  | 0.98                | $0.59       |
| anthropic-claude-3.5-haiku          | 0.77    | 0.80      | 0.75        | 0.71  | 0.80                | $6.12       |
| qwen-qwen-vl-plus                   | 0.74    | 0.83      | 0.58        | 0.75  | 0.85                | $1.43       |
| meta-llama-llama-4-scout            | 0.69    | 0.81      | 0.45        | 0.73  | 0.86                | $0.96       |
| amazon-nova-pro-v1                  | 0.69    | 0.67      | 0.66        | 0.72  | 0.80                | $6.78       |
| google-gemma-3-27b-it               | 0.69    | 0.74      | 0.61        | 0.69  | 0.71                | $0.31       |
| microsoft-phi-4-multimodal-instruct | 0.66    | 0.68      | 0.63        | 0.63  | 0.72                | $0.15       |
| mistralai-pixtral-large-2411        | 0.62    | 0.66      | 0.57        | 0.53  | 0.75                | $25.44      |
| openai-gpt-5-nano                   | 0.55    | 0.75      | 0.25        | 0.48  | 0.77                | $0.06       |
| qwen-qwen2.5-vl-32b-instruct        | 0.53    | 0.63      | 0.45        | 0.31  | 0.76                | $12.68      |
| meta-llama-llama-4-maverick         | 0.47    | 0.80      | 0.00        | 0.46  | 0.56                | $1.94       |
| google-gemma-3-12b-it               | 0.46    | 0.61      | 0.49        | 0.18  | 0.33                | $0.17       |
| Tesseract                           | 0.34    | 0.54      | 0.01        | 0.29  | 0.63                | $0.00       |
| mistralai-pixtral-12b               | -0.09   | 0.10      | -0.04       | 0.31  | -1.78               | $0.65       |


## Corpus
The corpus is constructed of a random sample of 400 images drawing from nine collections of Library of Congress images that were transcribed by volunteers in their By the People project. Creation dates range from the late 1700s to early 1900s. Images were restricted to those containing a single page of more than 25 words. It consists of four kinds of documents:
* Book pages (n=160, median of 171 words per page)
* Handwritten pages (n=120, median of 171 words per page)
* Other printed materials (n=40, median of 168 words per page)
* Documents with print and handwritten (n=80, median of 90 words per page)

## Transcription
Each image is encoded to base64 and paired with a detailed system prompt based on the [LoC guidelines](https://crowd.loc.gov/get-started/how-to-transcribe/) that sets strict transcription rules (*e.g.,* preserve spelling, mark deletions, handle illegible text). Two sample image–transcription pairs are also included as conversation history to provide the model with concrete examples of how to transcribe. Each model is accessed through OpenRouter.

## Accuracy
We evaluate each model’s transcription against the reference text using three error measures (all lower is better): WER (Word Error Rate), which counts substitutions, insertions, and deletions at the word level after normalizing by lowercasing, collapsing spaces, stripping ends, and tokenizing into words; CER (Character Error Rate), which applies the same substitution/insertion/deletion logic at the character level after stripping and lowercasing; and CER (alphanumeric-only, lowercase), which first removes all non-alphanumeric characters ([^0-9a-z]) and lowercases before computing character errors, making it less sensitive to punctuation and diacritics and serving as the default for headline comparisons. Final accuracy displayed is 1 – CER.





