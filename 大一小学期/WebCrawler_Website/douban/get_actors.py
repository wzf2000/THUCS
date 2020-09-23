# -*- coding: utf-8 -*-
from scrapy import Selector
import json
import pathlib
import codecs
import requests
import time

def parse_actor_local(body, url):
    item = {}
    selector = Selector(text = body)
    item['type'] = 'actor'
    item['id'] = url.split('/')[-2]
    item['name'] = selector.xpath('//div[@id="content"]/h1//text()').extract()[0]
    info = selector.xpath('//div[@class="info"]/ul')
    sex = info.xpath('//span[contains(text(),"性别")]/..//text()').extract()
    if len(sex) > 0:
        item['sex'] = sex[2][1:].strip()
    constellation = info.xpath('//span[contains(text(),"星座")]/..//text()').extract()
    if len(constellation) > 0:
        item['constellation'] = constellation[2][1:].strip()
    birthday = info.xpath('//span[contains(text(),"出生日期")]/..//text()').extract()
    if len(birthday) > 0:
        item['birthday'] = birthday[2][1:].strip()
    birth_place = info.xpath('//span[contains(text(),"出生地")]/..//text()').extract()
    if len(birth_place) > 0:
        item['birth_place'] = birth_place[2][1:].strip()
    profession = info.xpath('//span[contains(text(),"职业")]/..//text()').extract()
    if len(profession) > 0:
        item['profession'] = profession[2][1:].strip()
    item['intro'] = '\n'.join(selector.xpath('//div[@id="intro"]/div[@class="bd"]/span[@class="all hidden"]//text()').extract()).strip()
    if item['intro'] == '':
        item['intro'] = '\n'.join(selector.xpath('//div[@id="intro"]/div[@class="bd"]//text()').extract()).strip()
    image_url = selector.xpath('//div[@id="headline"]/div[@class="pic"]/a//@href').extract()
    if len(image_url) > 0:
        item['image_url'] = image_url[0]
    return item

cookie = {
    'bid': 'm5Z8raNBHlc',
    ' douban-fav-remind': '1',
    ' __yadk_uid': 'JTkLKSvC7DNlYaE3v23t2Gne5irtBGLK',
    ' ll': '"118173"',
    ' _vwo_uuid_v2': 'DB27014BDADDDE09C5AEA35C696D94939|88eb979bfdc470b9d3b4b0c5e381a787',
    ' __gads': 'ID',
    ' gr_user_id': '4ebf221e-8307-4c48-9b1c-46faaa376e58',
    ' __utmv': '30149280.21320',
    ' viewed': '"2372674_1014144_25962249"',
    ' __utmc': '30149280',
    ' __utma': '30149280.792297130.1577857372.1599478830.1599491341.40',
    ' __utmz': '30149280.1599491341.40.32.utmcsr',
    ' push_noty_num': '0',
    ' push_doumail_num': '0',
    ' _pk_ref.100001.8cb4': '%5B%22%22%2C%22%22%2C1599492280%2C%22https%3A%2F%2Faccounts.douban.com%2Fpassport%2Flogin%3Fsource%3Dmovie%22%5D',
    ' _pk_ses.100001.8cb4': '*',
    ' __utmt': '1',
    ' dbcl2': '"213204958:Vh2PJPmOzXk"',
    ' ck': 'Hroe',
    ' _pk_id.100001.8cb4': 'd54edee5de78da23.1577857371.30.1599492781.1599487700.',
    ' __utmb': '30149280.9.10.1599491341'
}
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
}

with codecs.open('actors.json', 'w', 'utf-8') as res:
    for i in range(1000001, 1443685):
        celebrity = pathlib.Path("E:\\Programming\\programs\\Python_Projects\\douban\\html\\celebrity_" + str(i) + '.html')
        if celebrity.is_file():
            body = open("E:\\Programming\\programs\\Python_Projects\\douban\\html\\celebrity_" + str(i) + '.html', 'r', encoding = 'utf-8').read()
            url = 'https://movie.douban.com/celebrity/' + str(i) + '/'
            content = json.dumps(parse_actor_local(body, url), ensure_ascii = False) + '\n'
            res.write(content)
            print(i)
