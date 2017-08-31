# sentinews
Machine built to monitor public opinion. Models based on `bidirectial LSTM with attention`  was trained by `online ugc-comment` from 大众点评.


### Requirements
- `Keras >= 2.0`
- `tensorflow >= 1.1.0`
- `h5py`
- `matplotlib`
- `jieba`
- `PyYAML`
- `python2.7`

### How to use

- **Install Package**
```bash
$ pip install -U ugc-sentiment
```

- **sentiment**
```python
# test.py
# coding: utf-8
from ugc_sentiment import UgcSentiment
conts = [
  u'服务态度太差了。团购去吃的，中午到的时候不到12点吧，手机在12点3分收到了已使用团购券。到了12点35的时候只上了一个菜，我实在是受不了了，就催服务员，服务员说让我稍等一下，饭菜20分钟左右就能好，我都等了40分钟了，才上一个菜，旁边的桌比我们晚来，菜上的都比我们多，你TMD能不开团购啊，能就开不能就别TM凑热闹 。。。。',
  u'店里的食物不算太全但食物味道还算可以，店员服务态度很好，价格比起同类的有些小贵，店面还是比较整洁的'
]
if __name__ == '__main__':
    # it will load model automatically
    senti = UgcSentiment()
    # receive params of unicode string
    result = senti.sentiment(conts[0])
```
+ **result：**
```json
    {
    "labels": "2",
    "probs_json": {
        "1": "0.43460175,,",
        "2": "0.44080707,",
        "3": "0.0980910137296",
        "4": "0.007562303,",
        "5": "0.0075623029843",
        },
    "probs": "0.43460175395 0.440807074308 0.0980910137296 0.0189378540963 0.0075623029843"
    }
```
+ **Support list of unicode string**
```python
    # also receive list of unicode string
    results = senti.sentiment(conts)
```
+ **result:**
```json
    [{
    "labels": "2",
    "probs_json": {
        "1": "0.43460175,,",
        "2": "0.44080707,",
        "3": "0.0980910137296",
        "4": "0.007562303,",
        "5": "0.0075623029843",
        },
    "probs": "0.43460175395 0.440807074308 0.0980910137296 0.0189378540963 0.0075623029843"
  },
    {
    "labels": "3",
    "probs_json": {
        "1": "0.0150475967675,,",
        "2": "0.118314310908,",
        "3": "0.647881686687",
        "4": "0.167027801275,",
        "5": "0.0517285950482",
        },
    "probs": "0.0150475967675 0.118314310908 0.647881686687 0.167027801275 0.0517285950482"
    }]
```


### License

GNU (for details, please refer to [LICENSE](https://github.com/KillersDeath/ugc-sentiment/blob/master/LICENSE))

Copyright (c) 2017 LiujieZhang
