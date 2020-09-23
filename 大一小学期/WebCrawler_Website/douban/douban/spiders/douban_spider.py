import scrapy
from scrapy import Selector
from douban.items import DoubanItem
from douban.items import ActorItem
import json
import time
import pathlib

class DoubanSpider(scrapy.Spider):
    name = "Douban"
    offset = 0
    start_url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&start=' + str(offset)
    actors = {}
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
    def start_requests(self):
        yield scrapy.Request(self.start_url, callback = self.parse, cookies = self.cookie)
    def parse(self, response):
        file_name = "E:\\Programming\\programs\\Python_Projects\\douban\\html\\" + response.request.url.split('/')[-2] + '_' + response.request.url.split('/')[-1].split('?')[0] + str(self.offset) + ".html"
        with open(file_name, "wb") as f:
            f.write(response.body)
        content_list = json.loads(response.body.decode())
        if (content_list['data'] == []):
            return
        for content in content_list['data']:
            time.sleep(1)
            path = pathlib.Path("E:\\Programming\\programs\\Python_Projects\\douban\\html\\" + content['url'].split('/')[-3] + '_' + content['url'].split('/')[-2] + ".html")
            if not path.is_file():
                yield scrapy.Request(url = content['url'], callback = self.parse_single, cookies = self.cookie)
        self.offset += 20
        if (self.offset > 1040):
            return
        url = 'https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&start=' + str(self.offset)
        yield scrapy.Request(url = url, callback = self.parse, cookies = self.cookie)
    def parse_single(self, response):
        file_name = "E:\\Programming\\programs\\Python_Projects\\douban\\html\\" + response.request.url.split('/')[-3] + '_' + response.request.url.split('/')[-2] + ".html"
        with open(file_name, "wb") as f:
            f.write(response.body)
        item = DoubanItem()
        selector = Selector(response)
        script = json.loads(selector.xpath('//script[@type="application/ld+json"]//text()').extract()[0])
        item['type'] = 'movie'
        item['title'] = selector.xpath('//span[@property="v:itemreviewed"]//text()').extract()[0]
        item['image_url'] = selector.xpath('//img[@rel="v:image"]//@src').extract()[0]
        item['intro'] = '\n'.join(selector.xpath('//div[@class="related-info"]/div[@class="indent"]/span[@class="all hidden"]//text()').extract()).strip()
        if item['intro'] == '':
            item['intro'] = '\n'.join(selector.xpath('//span[@property="v:summary"]//text()').extract()).strip()
        info = selector.xpath('//div[@id="info"]')
        item['director'] = info.xpath('//span[@class="attrs"]/a[@rel="v:directedBy"]//text()').extract()
        item['screenplay'] = info.xpath('//span[contains(text(),"编剧")]/../span[2]/a//text()').extract()
        cnt = 0
        item['actors'] = []
        for actor in script['actor']:
            id = actor['url'][11:-1]
            actor['id'] = id
            if id not in self.actors:
                self.actors[id] = actor
            item['actors'].append(actor)
            time.sleep(0.5)
            path = pathlib.Path("E:\\Programming\\programs\\Python_Projects\\douban\\html\\" + actor['url'].split('/')[-3] + '_' + actor['url'].split('/')[-2] + ".html")
            if not path.is_file():
                yield scrapy.Request(url = "https://movie.douban.com" + actor['url'], callback = self.parse_actor, cookies = self.cookie)
            cnt += 1
            if cnt >= 10:
                break
        item['genre'] = info.xpath('//span[@property="v:genre"]//text()').extract()
        item['initialReleaseDate'] = info.xpath('//span[@property="v:initialReleaseDate"]//text()').extract()[0]
        item['runtime'] = info.xpath('//span[@property="v:runtime"]//text()').extract()[0]
        item['review'] = selector.xpath('//div[@id="hot-comments"]/div/div/p/span//text()').extract()
        item['review_author'] = selector.xpath('//div[@id="hot-comments"]/div/div/h3/span[@class="comment-info"]/a//text()').extract()
        yield item
    def parse_actor(self, response):
        file_name = "E:\\Programming\\programs\\Python_Projects\\douban\\html\\" + response.request.url.split('/')[-3] + '_' + response.request.url.split('/')[-2] + ".html"
        with open(file_name, "wb") as f:
            f.write(response.body)
        item = ActorItem()
        selector = Selector(response)
        item['type'] = 'actor'
        item['id'] = response.request.url.split('/')[-2]
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
        yield item