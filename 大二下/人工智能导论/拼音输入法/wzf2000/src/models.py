from os import terminal_size
from .config import *
import json
import math

class BinaryModel(object):
    def __init__(self, alpha : float = 0.99999):
        with open(FREQUENCY % 1, 'r', encoding = 'utf8') as f:
            self.unigram_frequency = json.load(f)
        with open(FREQUENCY % 2, 'r', encoding = 'utf8') as f:
            self.binary_frequency = json.load(f)
        with open(WORD_TABLE, 'r', encoding = 'utf8') as f:
            self.word_table = json.load(f)
        self.alpha = alpha
        self.ratio = 100000.0

    def forward(self, sentence : str):
        sentence = sentence.split(' ')
        sentence = list(map(lambda item : item.lower(), sentence))
        new_sentence = list(map(lambda item : item.replace('lue', 'lve'), sentence))
        new_sentence = list(map(lambda item : item.replace('nue', 'nve'), new_sentence))
        dp = [{} for i in range(len(sentence))]
        last_word = [{} for i in range(len(sentence))]
        for word in self.word_table[sentence[0]]:
            if (word + new_sentence[0]) not in self.unigram_frequency:
                continue
            dp[0][word + new_sentence[0]] = math.log(self.unigram_frequency[word + new_sentence[0]])
            last_word[0][word + new_sentence[0]] = ''
        for i in range(len(sentence) - 1):
            for last in dp[i].keys():
                for now_word in self.word_table[sentence[i + 1]]:
                    now = now_word + new_sentence[i + 1]
                    if now not in self.unigram_frequency:
                        continue
                    if last not in self.binary_frequency:
                        continue
                    P = self.alpha * (self.binary_frequency[last][now] if now in self.binary_frequency[last] else 0) / self.unigram_frequency[last] + (1 - self.alpha) * self.unigram_frequency[now] / self.ratio
                    if now not in dp[i + 1]:
                        dp[i + 1][now] = dp[i][last] + math.log(P)
                        last_word[i + 1][now] = last
                    elif dp[i + 1][now] < dp[i][last] + math.log(P):
                        dp[i + 1][now] = dp[i][last] + math.log(P)
                        last_word[i + 1][now] = last
        max_prob = -1e10
        max_word = ''
        for word in dp[len(sentence) - 1].keys():
            if max_word == '':
                max_prob = dp[len(sentence) - 1][word]
                max_word = word
            elif max_prob < dp[len(sentence) - 1][word]:
                max_prob = dp[len(sentence) - 1][word]
                max_word = word
        res = ''
        now = max_word
        i = len(sentence) - 1
        while i >= 0:
            res = now[0] + res
            now = last_word[i][now]
            i -= 1
        return res

class TernaryModel(BinaryModel):
    def __init__(self, alpha : float = 0.99999, beta : float = 0.95):
        super().__init__(alpha)
        with open(FREQUENCY % 3, 'r', encoding = 'utf8') as f:
            self.ternary_frequency = json.load(f)
        self.beta = beta

    def forward(self, sentence : str):
        sentence = sentence.split(' ')
        sentence = list(map(lambda item : item.lower(), sentence))
        new_sentence = list(map(lambda item : item.replace('lue', 'lve'), sentence))
        new_sentence = list(map(lambda item : item.replace('nue', 'nve'), new_sentence))
        dp = [{} for i in range(len(sentence) - 1)]
        last_word = [{} for i in range(len(sentence) - 1)]
        for word in self.word_table[sentence[0]]:
            last = word + new_sentence[0]
            if last not in self.unigram_frequency or last not in self.binary_frequency:
                continue
            dp[0][last] = {}
            last_word[0][last] = {}
            for second_word in self.word_table[sentence[1]]:
                now = second_word + new_sentence[1]
                if now not in self.binary_frequency[last]:
                    continue
                P = self.alpha * (self.binary_frequency[last][now] if now in self.binary_frequency[last] else 0) / self.unigram_frequency[last] + (1 - self.alpha) * self.unigram_frequency[now] / self.ratio
                dp[0][last][now] = math.log(P) + math.log(self.unigram_frequency[last])
                last_word[0][last][now] = ''
        for i in range(len(sentence) - 2):
            for second_last in dp[i].keys():
                for last in dp[i][second_last].keys():
                    for now_word in self.word_table[sentence[i + 2]]:
                        now = now_word + new_sentence[i + 2]
                        if now not in self.unigram_frequency:
                            continue
                        if last not in self.binary_frequency:
                            continue
                        P = self.beta * ((self.ternary_frequency[second_last][last][now] if second_last in self.ternary_frequency and last in self.ternary_frequency[second_last] and now in self.ternary_frequency[second_last][last] else 0) / self.binary_frequency[second_last][last] if last in self.binary_frequency[second_last] else 0) + (1 - self.beta) * (self.alpha * (self.binary_frequency[last][now] if now in self.binary_frequency[last] else 0) / self.unigram_frequency[last] + (1 - self.alpha) * self.unigram_frequency[now] / self.ratio)
                        if last not in dp[i + 1]:
                            dp[i + 1][last] = {}
                            last_word[i + 1][last] = {}
                        if now not in dp[i + 1][last]:
                            dp[i + 1][last][now] = dp[i][second_last][last] + math.log(P)
                            last_word[i + 1][last][now] = second_last
                        elif dp[i + 1][last][now] < dp[i][second_last][last] + math.log(P):
                            dp[i + 1][last][now] = dp[i][second_last][last] + math.log(P)
                            last_word[i + 1][last][now] = second_last
        max_word = ''
        max_second_word = ''
        for word in dp[len(sentence) - 2].keys():
            for second_word in dp[len(sentence) - 2][word].keys():
                if max_word == '':
                    max_prob = dp[len(sentence) - 2][word][second_word]
                    max_word = word
                    max_second_word = second_word
                elif max_prob < dp[len(sentence) - 2][word][second_word]:
                    max_prob = dp[len(sentence) - 2][word][second_word]
                    max_word = word
                    max_second_word = second_word
        res = max_second_word[0]
        next = max_second_word
        now = max_word
        i = len(sentence) - 2
        while i >= 0:
            res = now[0] + res
            now, next = last_word[i][now][next], now
            i -= 1
        return res
