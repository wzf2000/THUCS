from .datareader import DataReader
from .config import *
import json
from pypinyin import lazy_pinyin
import jieba
import os

data_reader = DataReader()

def output_tables():
    word_table, pinyin_table = data_reader.read_table(TABLE)
    with open(WORD_TABLE, 'w', encoding = 'utf8') as f:
        json.dump(word_table, f, ensure_ascii = False)
    with open(PINYIN_TABLE, 'w', encoding = 'utf8') as f:
        json.dump(pinyin_table, f, ensure_ascii = False)

def output_sentences():
    for month in MONTHS:
        if os.path.exists(SENTENCES % month):
            print("File already existed!")
            continue
        print('start month:', month)
        with open(SENTENCES % month, 'w', encoding = 'utf8') as f:
            pass
        sina_news = data_reader.read_sina_news(SINA_NEWS % month)
        sentences = []
        percent = 1
        for new in sina_news:
            pinyin_sentence = lazy_pinyin(new, errors = lambda item : ['*' for i in range(len(item))])
            sentences.append((new, pinyin_sentence))
            while len(sentences) / len(sina_news) >= percent / 100:
                print('processing: ', str(percent) + '%')
                percent += 1
        with open(SENTENCES % month, 'w', encoding = 'utf8') as f:
            json.dump(sentences, f, ensure_ascii = False)
        print('finished month:', month)

def count_frequency():
    with open(PINYIN_TABLE, 'r', encoding = 'utf8') as f:
        pinyin_table = json.load(f)
    pinyin_table = pinyin_table.keys()
    count = {}
    binary_count = {}
    for month in MONTHS:
        print('start month:', month)
        percent = 1
        cnt = 0
        with open(SENTENCES % month, 'r', encoding = 'utf8') as f:
            sentences = json.load(f)
        for sentence in sentences:
            for i in range(len(sentence[0])):
                if sentence[0][i] not in pinyin_table:
                    continue
                now = sentence[0][i] + sentence[1][i]
                if now not in count:
                    count[now] = 1
                else:
                    count[now] += 1
                if i == 0:
                    continue
                if sentence[0][i - 1] in pinyin_table:
                    last = sentence[0][i - 1] + sentence[1][i - 1]
                    if last not in binary_count:
                        binary_count[last] = {}
                    if now not in binary_count[last]:
                        binary_count[last][now] = 0
                    binary_count[last][now] += 1
            cnt += 1
            while cnt / len(sentences) >= percent / 100:
                print('processing: ', str(percent) + '%')
                percent += 1
        print('finished month:', month)
    with open(FREQUENCY % 1, 'w', encoding = 'utf8') as f:
        json.dump(count, f, ensure_ascii = False)
    with open(FREQUENCY % 2, 'w', encoding = 'utf8') as f:
        json.dump(binary_count, f, ensure_ascii = False)

def count_ternary_frequency():
    with open(PINYIN_TABLE, 'r', encoding = 'utf8') as f:
        pinyin_table = json.load(f)
    pinyin_table = pinyin_table.keys()
    ternary_count = {}
    for month in MONTHS:
        print('start month:', month)
        percent = 1
        cnt = 0
        with open(SENTENCES % month, 'r', encoding = 'utf8') as f:
            sentences = json.load(f)
        for sentence in sentences:
            for i in range(len(sentence[0])):
                if i == 0 or i == 1:
                    continue
                if sentence[0][i] not in pinyin_table:
                    continue
                if sentence[0][i - 1] not in pinyin_table:
                    continue
                if sentence[0][i - 2] in pinyin_table:
                    second_last = sentence[0][i - 2] + sentence[1][i - 2]
                    last = sentence[0][i - 1] + sentence[1][i - 1]
                    now = sentence[0][i] + sentence[1][i]
                    if second_last not in ternary_count:
                        ternary_count[second_last] = {}
                    if last not in ternary_count[second_last]:
                        ternary_count[second_last][last] = {}
                    if now not in ternary_count[second_last][last]:
                        ternary_count[second_last][last][now] = 0
                    ternary_count[second_last][last][now] += 1
            cnt += 1
            while cnt / len(sentences) >= percent / 100:
                print('processing: ', str(percent) + '%')
                percent += 1
        print('finished month:', month)
    with open(FREQUENCY % 3, 'w', encoding = 'utf8') as f:
        json.dump(ternary_count, f, ensure_ascii = False)

def check_word(word : str, pinyin_table):
    for character in word:
        if character not in pinyin_table:
            return False
    return True

def count_word_frequency():
    with open(PINYIN_TABLE, 'r', encoding = 'utf8') as f:
        pinyin_table = json.load(f)
    pinyin_table = pinyin_table.keys()
    count = {}
    binary_count = {}
    for month in MONTHS:
        print('start month:', month)
        percent = 1
        cnt = 0
        with open(SENTENCES % month, 'r', encoding = 'utf8') as f:
            sentences = json.load(f)
        for sentence in sentences:
            word_sentence = jieba.lcut(sentence[0])
            l, r = 0, 0
            for i in range(len(word_sentence)):
                last_l, last_r = l, r
                l, r = r, r + len(word_sentence[i])
                if not check_word(word_sentence[i], pinyin_table):
                    continue
                now = word_sentence[i] + ''.join(sentence[1][l : r])
                if now not in count:
                    count[now] = 1
                else:
                    count[now] += 1
                if i == 0:
                    continue
                if not check_word(word_sentence[i - 1], pinyin_table):
                    continue
                last = word_sentence[i - 1] + ''.join(sentence[1][last_l : last_r])
                if last not in binary_count:
                    binary_count[last] = {}
                if now not in binary_count[last]:
                    binary_count[last][now] = 0
                binary_count[last][now] += 1
            cnt += 1
            while cnt / len(sentences) >= percent / 100:
                print('processing: ', str(percent) + '%')
                percent += 1
        print('finished month:', month)
    with open(WORD_FREQUENCY % 1, 'w', encoding = 'utf8') as f:
        json.dump(count, f, ensure_ascii = False)
    with open(WORD_FREQUENCY % 2, 'w', encoding = 'utf8') as f:
        json.dump(binary_count, f, ensure_ascii = False)

if __name__ == '__main__':
    count_word_frequency()