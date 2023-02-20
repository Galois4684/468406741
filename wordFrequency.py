import re
from collections import Counter
N_all = 0
N_5 = 0
word_5_dict = {}
word_all_dict = {}
path = [r"C:\Users\26271\Desktop\冰与火之歌：权力的游戏 A Game of Thrones.txt",
        r"C:\Users\26271\Desktop\冰与火之歌_冰雨的风暴 A Storm of Swords.txt",
        r"C:\Users\26271\Desktop\冰与火之歌：A Feast for Crows 群鸦的盛宴.txt",
        r"C:\Users\26271\Desktop\冰与火之歌_A CLASH OF KINGS 列王的纷争.txt",
        r"C:\Users\26271\Desktop\冰与火之歌_魔龙的狂舞 A Dance with Dragons.txt",
        r"C:\Users\26271\Desktop\Pride And prejudice傲慢与偏见.txt",
        r"C:\Users\26271\Desktop\不平静的坟墓 An Unquiet Grave.txt",
        r"C:\Users\26271\Desktop\The City of Dreadful Night8章节.txt",
        r"C:\Users\26271\Desktop\Scenes of Clerical Life教区生活场景.txt"]
for i in range(len(path)):
    # 读取文章
    with open(path[i], encoding='utf-8') as f:
        text = f.read()

    # 清洗文章
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = re.sub(r'\d+', '', text)  # 去除数字

    # 将文章转化为单词列表
    words = text.lower().split()

    # 统计单词出现的次数
    word_count = Counter(words)
    for word in word_count.keys():
        N_all = N_all + word_count[word]
        if word in word_all_dict:
            word_all_dict[word] = word_all_dict[word] + word_count[word]
        else:
            word_all_dict[word] = word_count[word]
        if len(word) == 5:
            if word in word_5_dict:
                word_5_dict[word] = word_5_dict[word] + word_count[word]
                N_5 = N_5 + word_count[word]
            else:
                word_5_dict[word] = word_count[word]
                N_5 = N_5 + word_count[word]
# print(word_5_dict)
# print(word_all_dict)
# print("diffrent len5 words: ",len(word_5_dict))
# print("diffrent all words: ",len(word_all_dict))
# print("N_5: ",N_5)
# print("N_all: ",N_all)

word_familiar_frequency_5_dict = {}
word_imformation_frequency_5_dict = {}
for word in word_5_dict:
    word_familiar_frequency_5_dict[word] = word_5_dict[word] / N_all
    word_imformation_frequency_5_dict[word] = word_5_dict[word] / N_5
# print(word_imformation_frequency_5_dict[word])
# print(word_familiar_frequency_5_dict[word])

word_order=sorted(word_5_dict.items(),key=lambda x:x[1],reverse=True)
# word_order_dict = {}
# for i in range(len(word_order)):
#     word_order_dict[word_order[i][0]] = i+1
print(word_order)

