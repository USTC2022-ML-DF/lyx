# USTC-ML-Lab 2022

我们实验分成三个模块，分别是三种算法的尝试
1. 词袋贝叶斯分类器
2. Finetuned BERT分类器
3. 在无标签数据集上MLM Finutined BERT

## 目录结构
```
bert_vulnerablity/      
    code/               # 基于BERT的分类器
    code_mlm/           # 基于BERT及MLM无监督训练的分类器
    data/               # 数据 (需要创建并下载数据)
        train.json      
        valid.json
        test_a.json
    save/               # 模型权重保存路径
    submissions/        # 预测结果保存路径
word_bag/               
    word_bag_bayes.py   # 基于词袋的贝叶斯分类器
    data/               # 数据 (需要创建并并下载数据)
        train.json      
        valid.json
        test_a.json
    submissions/        # 预测结果保存路径
```

## 运行环境

* 系统 Ubuntu 18.04
* Python 环境: (latest versions are okay)
    * scikit-learn
    * nltk    
    * pytorch
    * transformers
    * xlswriter
    * pandas
    * tqdm

## 运行指令

* word_bag

将数据放在指定位置，运行算法，输出结果在word_bag/submissions/*.xls中

```bash
cd word_bag
python word_bag_bayes.py    # 训练、验证、测试、预测一体
```

* bert_vulnerability

将数据放在指定位置，在 `bert_vulnerability/code/config.py` 中修改输入与输出路径。输出结果在bert_vulnerability/submissions/*.xls中

```bash
cd bert_vulnerability/code
# 或 cd bert_vulnerability/code_mlm
python train.py             # 训练、验证
python infer.py             # 预测
``` 
