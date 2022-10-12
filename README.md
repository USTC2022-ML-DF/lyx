# lyx
构造如下目录
--root
    --code
    --data
        --train.json
        --valid.json
        --test_a.json

然后需要修改/code/config.py里datapath、--savedmodel_path、--ckpt_file对应的路径, 以及其余训练时超参数
* train时运行 python /code/train.py
* predict时运行python /code/infer.py

