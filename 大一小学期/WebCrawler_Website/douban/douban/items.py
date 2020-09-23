# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class DoubanItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    type = scrapy.Field()
    title = scrapy.Field()
    image_url = scrapy.Field()
    intro = scrapy.Field()
    director = scrapy.Field()
    screenplay = scrapy.Field()
    actors = scrapy.Field()
    genre = scrapy.Field()
    initialReleaseDate = scrapy.Field()
    runtime = scrapy.Field()
    review = scrapy.Field()
    review_author = scrapy.Field()

class ActorItem(scrapy.Item):
    type = scrapy.Field()
    id = scrapy.Field()
    name = scrapy.Field()
    sex = scrapy.Field()
    constellation = scrapy.Field()
    birthday = scrapy.Field()
    birth_place = scrapy.Field()
    profession = scrapy.Field()
    intro = scrapy.Field()
    image_url = scrapy.Field()

