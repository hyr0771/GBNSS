### Abstract
Biological network similarity search plays a crucial role in the analysis of biological networks for human disease research, and drug discovery. The biological network similarity search aims to efficiently identify novel networks that are biologically homologous to the query networks. Great progress has been achieved in the biological network similarity search. However, it remains a challenge to fully mine the biological network information to improve the accuracy of query results without increasing time overheads. In this study, we propose a biological network similarity search method based on graph neural networks named GBNSS, which combines topological and biological information (GO annotations) of biological networks into the graph neural networks to find topologically and biologically similar biological networks in the database. Additionally, GBNSS is a topology-free biological network similarity search method with arbitrary network structure. The experimental results on four benchmark datasets show that GBNSS outperforms the existing methods in terms of computational efficiency and search accuracy. Case studies further demonstrate that GBNSS is capable of searching similar networks in real-world biological networks.


### Requirements
The codebase is implemented in Python 3.10.6. package versions used for development are listed in requirment.txt


### Datasets
The Dataset contain data that used for GBNSS.
rawData: the initial data for each networks from species, in each network, it has the following structure:

"""
sce:YHR186C	sce:YKL203C
sce:YHR186C	sce:YGL180W
sce:YHR186C	sce:YPR185W
sce:YHR186C	sce:YPR049C
sce:YHR186C	sce:YMR028W
sce:YJR066W	sce:YNL006W
sce:YJR066W	sce:YGL180W
"""

bio_feature.csv: each node's biological features. 
topo_feature.csv: each node's topological features.

data: the code takes pairs of graphs for training and testing. the columns = ["graph1", "graph2", "ec", "lccs", "s3", "go"]
"""
sce00130.txt	sce00100.txt	1.0	6	1.0	9.808
sce00190.txt	sce00010.txt	0.235294	12	0.0788177	149.27
sce00190.txt	sce00020.txt	0.220588	15	0.135135	108.479
sce00190.txt	sce00030.txt	0.529412	36	0.428571	59.498
sce00190.txt	sce00040.txt	0.833333	5	0.454545	8.744
sce00190.txt	sce00051.txt	0.220588	15	0.157895	55.102
"""


### Model options
```
  --filters-1             INT         Number of filter in 1st GCN layer.       Default is 128.
  --filters-2             INT         Number of filter in 2nd GCN layer.       Default is 64. 
  --filters-3             INT         Number of filter in 3rd GCN layer.       Default is 32.
  --tensor-neurons        INT         Neurons in tensor network layer.         Default is 16.
  --bottle-neck-neurons   INT         Bottle neck layer neurons.               Default is 16.
  --bins                  INT         Number of histogram bins.                Default is 16.
  --batch-size            INT         Number of pairs processed per batch.     Default is 128. 
  --epochs                INT         Number of SimGNN training epochs.        Default is 5.
  --dropout               FLOAT       Dropout rate.                            Default is 0.5.
  --learning-rate         FLOAT       Learning rate.                           Default is 0.001.
  --weight-decay          FLOAT       Weight decay.                            Default is 10^-5.
  --histogram             BOOL        Include histogram features.              Default is False.
```


### Examples

Training a GBNSS model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```

Increasing the learning rate and the dropout.
```
python src/main.py --learning-rate 0.01 --dropout 0.9
'''

You can save the trained model by adding the `--save-path` parameter.
```
python src/main.py --save-path /Method/save_path/model-name
```

You can get the final rank by 
```
python src/rank.py


### Result format
Here is an example for the output file "ec_result.csv" by using EC as label:
graph1  graph2  result
sce04392.txt	sce00062.txt	0.5393388867378235
sce04392.txt	sce00563.txt	0.5333682894706726
sce04392.txt	sce00860.txt	0.5308347344398499
sce04392.txt	sce00513.txt	0.5282118320465088
sce04392.txt	sce00440.txt	0.5258874297142029

### Funding support
This work is supported by the National Natural Science Foundation of China (No. 61862006 and No.62261003) and the Natural Science Foundation of Guangxi Prov-ince (No.2020GXNSFAA159074).


