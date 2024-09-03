# Counting-component-for-VQA
Improvement of vqa counting method

A key aspect of VQA is its ability to compute objects in images, which is crucial for solving problems involving quantity. For the counting method, Yan Zhang proposed a novel approach to enhance the VQA system's ability to count objects in natural images. By combining object counting with visual question answering tasks, researchers have proposed a model that can better identify and count objects in images, thereby improving the accuracy of VQA systems for quantity problems. Despite improvements, text processing in VQA systems is still limited, especially for problems with complex language structures or context sensitive queries. The aim of this study is to improve the VQA system by proposing enhancements to text processing and fusion components, and to address the counting problem in visual question answering using different combinations and variations of language and visual models. 

## Single-model results on VQA v2 test-std split

Through research, our main contribution is the introduction of Long-Short Term Memory(LSTM) with attention mechanism for improvement, and we have developed a low rank version of the Multimodal Bilinear Pooling(MLB) fusion module to reduce computational complexity and adapt to the low memory GPU. We have also found that the actual BERT model is not suitable for VQA tasks.


|   Metrics  |  RoBERTa  | DisBert | LSTM   | Origin   |
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:  | 
| Original fusion | 55.38 | 56.22 | *66.25 | **66.55** | 
| MLB  | 52.19 | 54.70 | **60.03** | 53.09 | 
| Concatenate | 55.59 | 55.52 | **61.93** | 55.12 | 



UPDATE: 
