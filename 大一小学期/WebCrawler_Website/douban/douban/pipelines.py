# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json
import codecs


class DoubanPipeline:
    def open_spider(self, spider):
        self.file = codecs.open('result.json', 'w', 'utf-8')
        self.num = 0
    def process_item(self, item, spider):
        self.num += 1
        content = json.dumps(dict(item), ensure_ascii = False) + '\n'
        self.file.write(content)
        return item
    def close_spider(self, spider):
        print("共" + str(self.num) + "条数据")
        self.file.close()
