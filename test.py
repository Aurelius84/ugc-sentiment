# coding: utf-8
from sentinews import SentiNews

conts = [
    u'在绚丽之后，一条路慢慢展开，指南针引导东方与西方的握手，这是我们与世界的昨天。',
    u'我们用低宛的音乐和华丽的服饰，演绎了中国古代的含蓄文明，低调而绚丽。',
    u'下面就完全陷于一种扯皮，机组与机场之间互相推诿指责，我们弄不清楚，只好旁观。',
    u'这个道理谁都懂，谁都惜命，不敢拿生命开玩笑。'
]


if __name__ == '__main__':
    senti = SentiNews()
    result = senti.sentiment(conts)
    pre = [res['labels'] for res in result]
    for i in range(len(conts)):
        print(conts[i])
        print('predict: %s' % pre[i])
        s = ['%s: %s' % (result[i]['probs'].keys()[j], result[i]['probs'].values()[j]) for j in range(3)]
        print(' '.join(s))
