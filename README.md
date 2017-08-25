# the_machine
Machine built to monitor public opinion. Models based on `stacked LSTM`(three layers)  was trained by `written language` from blog and article.


### Requirements
- `Keras >= 2.0`
- `tensorflow >= 1.1.0`
- `jieba`
- `PyYAML`
- `python2.7`

### How to use

- **Install Package**
```bash
$ pip install -U sentinews
```

- **sentiment**

```python
# test.py
# coding: utf-8
from sentinews import SentiNews

conts = [
    u'在绚丽之后，一条路慢慢展开，指南针引导东方与西方的握手，这是我们与世界的昨天。',
    u'下面就完全陷于一种扯皮，机组与机场之间互相推诿指责，我们弄不清楚，只好旁观。'
]


if __name__ == '__main__':
    # it will load model automatically
    senti = SentiNews()
    # receive params of unicode string
    result = senti.sentiment(conts[0])
    # result:
    # {
    # "labels": "中性",
    # "probs": {
    #     "中性": "0.954392",
    #     "消极": "0.0124916",
    #     "积极": "0.0331167"
    #     }
    # }

    # also receive list of unicode string
    results = senti.sentiment(conts)
    # results:
    # [{
    # "labels": "中性",
    # "probs": {
    #     "中性": "0.954392",
    #     "消极": "0.0124916",
    #     "积极": "0.0331167"
    #     }
    # },
    # {
    # "labels": "消极",
    # "probs": {
    #     "中性": "0.357558",
    #     "消极": "0.529143",
    #     "积极": "0.113299"
    #     }
    # }]
```


### License

GNU (for details, please refer to [LICENSE](https://github.com/KillersDeath/the_machine/blob/master/LICENSE))

Copyright (c) 2017 LiujieZhang
