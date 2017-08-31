# coding: utf-8
from ugc_sentiment import UgcSentiment

conts = [
    u'服务态度太差了。团购去吃的，中午到的时候不到12点吧，手机在12点3分收到了已使用团购券。到了12点35的时候只上了一个菜，我实在是受不了了，就催服务员，服务员说让我稍等一下，饭菜20分钟左右就能好，我都等了40分钟了，才上一个菜，旁边的桌比我们晚来，菜上的都比我们多，你TMD能不开团购啊，能就开不能就别TM凑热闹 。。。。',
    u'店里的食物不算太全但食物味道还算可以，店员服务态度很好，价格比起同类的有些小贵，店面还是比较整洁的',
    u'这家菜真的是大爱，就喜欢这种小巧精致的菜，看着既舒服，又好吃，是一家比较老字号的店了，店里生意很好，每次来的时候人都是挺多的，应为确实口味不错，好吃，对于吃的东西，我觉得好吃就是对是无的最高评价了。经常会和家人一块过来吃，本来就喜欢吃带面粉的食物，吃小笼包，大赞啊，还有玉脂冰清，嫩嫩的，滑滑的。'
]

if __name__ == '__main__':
    senti = UgcSentiment()
    result = senti.sentiment(conts)
    print(result)
    pre = [res['labels'] for res in result]
    for i in range(len(conts)):
        print(conts[i])
        print('predict: %s' % pre[i])
        s = [
            '%s: %s' % (result[i]['probs'].keys()[j],
                        result[i]['probs_json'].values()[j]) for j in range(5)
        ]
        print(' '.join(s))
