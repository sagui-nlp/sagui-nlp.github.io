---
title: "DeBERTinha: A Multistep Approach to Adapt DebertaV3 XSmall for Brazilian Portuguese Natural Language Processing Task"
link: https://huggingface.co/collections/sagui-nlp/debertinha-6516fbc717cd8716dda6be89
year: 2023
---
This paper presents an approach for adapting the DebertaV3 XSmall model pre-trained in English for Brazilian Portuguese natural language processing (NLP) tasks. A key aspect of the methodology involves a multistep training process to ensure the model is effectively tuned for the Portuguese language. Initial datasets from Carolina and BrWac are preprocessed to address issues like emojis, HTML tags, and encodings. A Portuguese-specific vocabulary of 50,000 tokens is created using SentencePiece. Rather than training from scratch, the weights of the pre-trained English model are used to initialize most of the network, with random embeddings, recognizing the expensive cost of training from scratch. The model is fine-tuned using the replaced token detection task in the same format of DebertaV3 training. The adapted model, called DeBERTinha, demonstrates effectiveness on downstream tasks like named entity recognition, sentiment analysis, and determining sentence relatedness, outperforming BERTimbau-Large in two tasks despite having only 40M parameters.

