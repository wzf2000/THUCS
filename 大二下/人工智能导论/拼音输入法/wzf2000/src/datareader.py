import json

class DataReader(object):
    def read_file(self, path : str):
        try:
            with open(path, 'r', encoding = 'gbk') as f:
                ret = f.readlines()
            return ret
        except Exception as e:
            print(e)
            return []

    def read_table(self, path : str):
        lines = self.read_file(path)
        word_table = {}
        pinyin_table = {}
        for line in lines:
            words = line.split()
            word_table[words[0]] = words[1 : ]
            for word in words[1 : ]:
                if word in pinyin_table:
                    pinyin_table[word].append(words[0])
                else:
                    pinyin_table[word] = [ words[0] ]
        return word_table, pinyin_table

    def read_sina_news(self, path : str):
        try:
            lines = self.read_file(path)
            ret = []
            for line in lines:
                news = json.loads(line)
                ret.append(news['html'])
                ret.append(news['title'])
            return ret
        except Exception as e:
            print(e)
            return []
