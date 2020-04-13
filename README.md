# Data-to-Text-Generation

## Content

1. [Papers](https://github.com/DrLiLiang/Data-to-Text-Generation#1-papers)
2. [Datasets](https://github.com/DrLiLiang/Data-to-Text-Generation#2-datasets)
3. [Evaluation Metrics](https://github.com/DrLiLiang/Data-to-Text-Generation#3-evaluation-metrics)

## 1. Papers

### 2016
1. [Neural Text Generation from Structured Data with Application to the Biography Domain](https://arxiv.org/abs/1603.07771) **EMNLP2016**
   - Code:[Official](https://github.com/DavidGrangier/wikipedia-biography-dataset)

### 2017
1. [Challenges in Data-to-Document Generation](https://arxiv.org/abs/1707.08052) **EMNLP2017**
   - Code: [Official](https://github.com/harvardnlp/data2text)
2. [Order-planning neural text generation from structured data](https://arxiv.org/abs/1709.00155) **AAAI2018**
3. [Table-to-text Generation by Structure-aware Seq2seq Learning](https://arxiv.org/abs/1711.09724) **AAAI2018**
   - Code:[Official](https://github.com/tyliupku/wiki2bio)

4. [Table-to-Text: Describing Table Region with Natural Language](https://arxiv.org/abs/1805.11234) **AAAI2018**

5. [Generating Descriptions from Structured Data Using a Bifocal Attention Mechanism and Gated Orthogonalization](https://www.aclweb.org/anthology/N18-1139/) **NAACL2018**
   - Code: [Official](https://github.com/PrekshaNema25/StructuredData_To_Descriptions)

6. [A mixed hierarchical attention based encoder-decoder approach for standard summarizaion](https://arxiv.org/abs/1804.07790) **NAACL2018**

### 2018

1. [Operation-guided Neural Networks for High Fidelity Data-To-Text Generation](https://arxiv.org/abs/1809.02735v1) **EMNLP2018**
2. [Learning Neural Templates for Text Generation](https://arxiv.org/abs/1808.10122) **EMNLP2018**
   - Code: [Official](https://github.com/harvardnlp/neural-template-gen)
3. [Learning Latent Semantic Annotations for Grounding Natural Language to Structured Data](https://www.aclweb.org/anthology/D18-1411/) **EMNLP2018**
   - Code: [Official](https://github.com/hiaoxui/D2T-Grounding)
4. [Data-to-Text Generation with Content Selection and Planning](https://arxiv.org/abs/1809.00582) **AAAI2019**
   - Code: [Official](https://github.com/ratishsp/data2text-plan-py)
5. [Hierarchical Encoder with Auxiliary Supervision for Neural Table-to-Text Generation: Learning Better Representation for Tables](https://www.aaai.org/ojs/index.php/AAAI/article/view/4653) **AAAI2019**
6. [Key Fact as Pivot: A Two-Stage Model for Low Resource Table-to-Text Generation](https://arxiv.org/abs/1908.03067) **ACL2019**
7. [Learning to Select, Track, and Generate for Data-to-Text](https://www.aclweb.org/anthology/P19-1202/) **ACL2019**
   - Code: [Official](https://github.com/aistairc/sports-reporter)
8. [Towards Comprehensive Description Generation from Factual Attribute-value Tables ](https://www.aclweb.org/anthology/P19-1600/) **ACL2019**
9. [Data-to-text Generation with Entity Modeling](https://www.aclweb.org/anthology/P19-1195/) **ACL2019**
10. [Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081) **ACL2019** 
   - Code: [Official](https://github.com/google-research/language/tree/master/language/table_text_eval)
11. [Step-by-Step: Separating Planning from Realization in Neural Data-to-Text Generation](https://arxiv.org/abs/1904.03396) **NAACL2019**
    - Code: [Official](https://github.com/AmitMY/chimera)
12. [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/abs/1904.02342) **NAACL2019**
    - Code: [Official](https://github.com/rikdz/GraphWriter)
13. [Deep Graph Convolutional Encoders for Structured Data to Text Generation](http://aclweb.org/anthology/W18-6501) **INLG2018**
    - Code:  [Official](https://github.com/diegma/graph-2-text)
14. ...

### 2019

1. [Enhancing Neural Data-To-Text Generation Models with External Background Knowledge](https://www.aclweb.org/anthology/D19-1299/) **EMNLP2019**
   - Code: [Official](https://github.com/hitercs/WikiInfo2Text)
2. [Neural data-to-text generation: A comparison between pipeline and end-to-end architectures](https://arxiv.org/abs/1908.09022) **EMNLP2019**
   - Code: [Official](https://github.com/ThiagoCF05/DeepNLG/)
3. [Table-to-Text Generation with Effective Hierarchical Encoder on Three dimensions (Row, Column and Time)](https://arxiv.org/abs/1909.02304?context=cs) **EMNLP2019**
4. [Enhanced Transformer Model for Data-to-Text Generation](https://www.aclweb.org/anthology/D19-5615/) **EMLP-WGNT2019**
   - Code: [Official](https://github.com/gongliym/data2text-transformer)
5. [Selecting, Planning, and Rewriting: A Modular Approach for Data-to-Document Generation and Translation](https://www.aclweb.org/anthology/D19-5633/) **EMNLP2019-short**
6. [An Encoder with non-Sequential Dependency for Neural Data-to-Text Generation](https://www.aclweb.org/anthology/W19-8619/) **INLG2019**
7. [Controlling Contents in Data-to-Document Generation with Human-Designed Topic Labels](https://www.aclweb.org/anthology/W19-8640/) **INLG2019**
8. [Revisiting Challenges in Data-to-Text Generation with Fact Grounding](https://www.aclweb.org/anthology/W19-8639/) **INLG2019**
   - Code: [Official](https://github.com/wanghm92/rw_fg)
9. [Learning to Select Bi-Aspect Information for Document-Scale Text Content Manipulation](https://arxiv.org/abs/2002.10210) **AAAI2020**
   - Code: [Official](https://github.com/syw1996/SCIR-TG-Data2text-Bi-Aspect)
10. [Variational Template Machine for Data-to-Text Generation](https://openreview.net/forum?id=HkejNgBtPB) **ICLR2020**

## 2. DataSets

### 1. WikiBio

Source: [Neural text generation from structured data with application to the biography domain.](https://arxiv.org/abs/1603.07771) **EMNLP2016**

Code: [Official](https://github.com/DavidGrangier/wikipedia-biography-dataset)

#### Related Papers
1. [Order-planning neural text generation from structured data](https://arxiv.org/abs/1709.00155) **AAAI2018**
2. [Table-to-text Generation by Structure-aware Seq2seq Learning](https://arxiv.org/abs/1711.09724) **AAAI2018**
3. [Table-to-Text: Describing Table Region with Natural Language](https://arxiv.org/abs/1805.11234) **AAAI2018**
4. [Generating Descriptions from Structured Data Using a Bifocal Attention Mechanism and Gated Orthogonalization](https://www.aclweb.org/anthology/N18-1139/) **NAACL2018**
5. [Learning Neural Templates for Text Generation](https://arxiv.org/abs/1808.10122) **EMNLP2018**
6. [An Encoder with non-Sequential Dependency for Neural Data-to-Text Generation](https://www.aclweb.org/anthology/W19-8619/) **INLG2019**
7. [Hierarchical Encoder with Auxiliary Supervision for Neural Table-to-Text Generation: Learning Better Representation for Tables](https://www.aaai.org/ojs/index.php/AAAI/article/view/4653) **AAAI2019**
8. [Enhancing Neural Data-To-Text Generation Models with External Background Knowledge](https://www.aclweb.org/anthology/D19-1299/) **EMNLP2019**
9. [Variational Template Machine for Data-to-Text Generation](https://openreview.net/forum?id=HkejNgBtPB) **ICLR2020**

### 2. ROTOWIRE

Source:[Challenges in Data-to-Document Generation](https://arxiv.org/abs/1707.08052) **EMNLP2017**
Code: [Official](https://github.com/harvardnlp/boxscore-data)

#### Related Papers:

1. [Operation-guided Neural Networks for High Fidelity Data-To-Text Generation](https://arxiv.org/abs/1809.02735v1) **EMNLP2018**
2. [Data-to-Text Generation with Content Selection and Planning](https://arxiv.org/abs/1809.00582) **AAAI2019**
3. [Data-to-text Generation with Entity Modeling](https://www.aclweb.org/anthology/P19-1195/) **ACL2019**
4. [Learning to Select, Track, and Generate for Data-to-Text](https://www.aclweb.org/anthology/P19-1202/) **ACL2019**
5. [Table-to-Text Generation with Effective Hierarchical Encoder on Three dimensions (Row, Column and Time)](https://arxiv.org/abs/1909.02304?context=cs) **EMNLP2019**
6. [Selecting, Planning, and Rewriting: A Modular Approach for Data-to-Document Generation and Translation](https://www.aclweb.org/anthology/D19-5633/) **EMNLP2019-short**
7. [Enhanced Transformer Model for Data-to-Text Generation](https://www.aclweb.org/anthology/D19-5615/) **EMLP-WGNT2019**

### 3. WIKITABLETEXT

Source: [Table-to-Text: Describing Table Region with Natural Language](https://arxiv.org/abs/1805.11234) **AAAI2018**

Code: **None**

### 4. WebNLG challenge

Source: [The WebNLG Challenge: Generating Text from DBPedia Data](https://www.aclweb.org/anthology/W16-6626/)

#### Related Papers

1. [Deep Graph Convolutional Encoders for Structured Data to Text Generation](http://aclweb.org/anthology/W18-6501) **INLG2018**
2. [Step-by-Step: Separating Planning from Realization in Neural Data-to-Text Generation](https://arxiv.org/abs/1904.03396) **NAACL2019**

### 5. The Augmented WebNLG corpus

Source: [Enriching the WebNLG corpus](https://www.aclweb.org/anthology/W18-6521/) **ACL2018**

Code: [Official](https://github.com/ThiagoCF05/webnlg)

#### Related Papers
1. [Neural data-to-text generation: A comparison between pipeline and end-to-end architectures](https://arxiv.org/abs/1908.09022) **EMNLP2019**

### 6. ROTOWIRE-MODIFIED

Source: [Learning to Select, Track, and Generate for Data-to-Text](https://www.aclweb.org/anthology/P19-1202/) **ACL2019**

Code: [Official](https://github.com/aistairc/sports-reporter)

### 7. ESPN

Source: [Operation-guided Neural Networks for High Fidelity Data-To-Text Generation](https://arxiv.org/abs/1809.02735v1) **EMNLP2018**

Code: [Official](https://github.com/janenie/espn-nba-data)

#### Related Papers

1. [An Encoder with non-Sequential Dependency for Neural Data-to-Text Generation](https://www.aclweb.org/anthology/W19-8619/) **INLG2019**

### 8. WEATHERGOV

Source: [Learning semantic correspondences with less supervision](https://cs.stanford.edu/~pliang/papers/semantics-acl2009.pdf) **ACL2009**

#### Related Papers

1. [A mixed hierarchical attention based encoder-decoder approach for standard summarizaion](https://arxiv.org/abs/1804.07790) **NAACL2018**

### 9. RW-FG

Source: [Revisiting Challenges in Data-to-Text Generation with Fact Grounding](https://www.aclweb.org/anthology/W19-8639/) **INLG2019**

Code: [Official](https://github.com/wanghm92/rw_fg)

### 10. AGENDA

Source: [Text Generation from Knowledge Graphs with Graph Transformers](https://arxiv.org/abs/1904.02342) **NAACL2019**

Code: [Official](https://github.com/rikdz/GraphWriter)

### 11. SPNLG

Source: [Can neural generators for dialogue learn sentence planning and discourse structuring?](https://arxiv.org/abs/1809.03015) **INLG2018**

Code: [Official](https://nlds.soe.ucsc.edu/sentence-planning-NLG)

#### Related Papers

1. [Variational Template Machine for Data-to-Text Generation](https://openreview.net/forum?id=HkejNgBtPB) **ICLR2020**

## 3. Evaluation Metrics
1. **PARENT**: [Handling Divergent Reference Texts when Evaluating Table-to-Text Generation](https://arxiv.org/abs/1906.01081) **ACL2019**
   - Code: [Official](https://github.com/google-research/language/tree/master/language/table_text_eval)
# Updating ......

