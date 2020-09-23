# -*- coding: utf-8 -*-
from scrapy import Selector
import json
import pathlib
import codecs

with codecs.open('movies.json', 'w', 'utf-8') as res:
    file_name = "E:\\Programming\\programs\\Python_Projects\\douban\\final_result.json"
    with open(file_name, 'r', encoding = 'utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            info = json.loads(line)
            if info['type'] == 'movie':
                res.write(line)