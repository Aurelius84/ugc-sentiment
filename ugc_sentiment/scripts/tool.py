# coding: utf-8


def loadSentimentVector(file_name):
    """
    Load sentiment vector
    [Surprise, Sorrow, Love, Joy, Hate, Expect, Anxiety, Anger]
    """
    contents = [
        line.strip('\n').split() for line in open(file_name, 'r').readlines()
    ]
    sentiment_dict = {
        line[0].decode('utf-8'): [float(w) for w in line[1:]]
        for line in contents
    }
    return sentiment_dict


if __name__ == '__main__':
    data_dir = '/Users/zhangliujie/Documents/data/all_vocab'
    # load sentiment_dict
    sentiment_dict = loadSentimentVector('../docs/sentiment/extend_dict.txt')
    cnt = 0
    with open(data_dir+'/40.vocab', 'r') as f:
        for line in f:
            words = line.decode('utf8').split()
            filter_wds = filter(lambda x: x in sentiment_dict, words)

            print('*'*60)
            print(line)
            print(' '.join(filter_wds))

            cnt += 1
            if cnt == 100:
                exit()
