<h2><center>影视爬虫与检索系统设计文档</center></h2>

<p style="text-align:right"><strong>2019011200 计 93 王哲凡</strong></p>

---

目前网站在 [http://douban.wzf2000.top](http:douban.wzf2000.top) 下运行。

### 1. 项目基本结构

#### 1.1. 影视爬虫部分预览

![预览1](https://img.wzf2000.top/image/2020/09/23/1ede4794d5ccf240f.png)

#### 1.2. 网站部分预览

![预览2](https://img.wzf2000.top/image/2020/09/23/2f9fec890c3601947.png)

![预览3](https://img.wzf2000.top/image/2020/09/23/32cdfd5f4c1235b9f.png)

（图中没有截到 `manage.py` 文件，并非没有）

#### 1.3. 影视爬虫部分简介

主要采用 `scrapy` 来进行爬虫，其中 `spiders` 文件夹下的 `douban_spider.py` 为主要爬虫文件，`items.py` 为输出格式文件（输出格式为 `.json`），`pipelines.py` 为输出控制文件，`settings.py` 为设置文件。

而 `html` 文件夹下则保存了所有爬取过的网页的源代码，方便爬虫结束后进行部分信息纠错。

最终爬虫导出的数据文件为 `final_result.json`，通过 `get_actors.py` 和 `get_movies.py` 进行数据分离得到 `actors.json` 和 `movies.json` ，其中出现的数据错误由 `deal.py` 纠正（根据本地 HTML 文件）。

由于网站采用了 MySQL 数据库，因此需要将输出的 `.json` 文件按数据表格式导入到数据库，这部分由 `input.py` 负责导入，其中包括表格式设计和利用 `pymysql` 库接口导入。

#### 1.4. 网站部分简介

`douban_movie` 为项目总文件夹，其下 `douban_movie` 文件夹中内容主要包括了基础的路径等信息设置。

主要的网站逻辑由 `app` 文件夹代表的 APP 完成，其下 `template` 文件夹用于存放和调取 HTML 模板文件，`templatetags` 保存了自定义的一个模板 `filter`，主要的网站生成由 `views.py` 完成，路径分析由 `urls.py` 完成。

`common_static` 文件夹用于存放静态文件，包括 `.css`、图片、字体文件，`collected_static` 则由 `python manage.py collectstatic` 自动生成。

### 2. 影视爬虫部分

#### 2.1. 影视爬虫

爬虫的主体为 `douban_spider.py`，其根据 `https://movie.douban.com/j/new_search_subjects?sort=U&range=0,10&tags=%E7%94%B5%E5%BD%B1&start=0` 网址，得出排行榜上电影的信息页 `url`，再进一步对信息页进行爬虫。

此网页一次会显示 $20$ 条电影信息，每次爬取完后，只须将 `start=` 后的数字增加 $20$ 即可获得之后 $20$ 部电影信息。

在电影信息页中，除了爬取基本信息外，还会根据源代码中 `<script type="application/ld+json">` 部分内容获取其演员的信息页地址等信息，然后进一步进入演员信息页爬取信息。

爬取过程中，对于 HTML 代码的分析主要利用 `scrapy` 的 `Selector`，并利用其中的 `xpath` 方法分析。

例如，爬取电影中的编剧信息就可以通过含“编剧”一词的标签得到，如下所示（第一步先获取 `id="info"` 的 `div` 元素）：

```python
info = selector.xpath('//div[@id="info"]')
item['screenplay'] = info.xpath('//span[contains(text(),"编剧")]/../span[2]/a//text()').extract()
```

电影和演员的大部分信息根据 `span` 标签的 `property` 属性即可定位，剩余也基本可以通过类似编剧的获取父元素下其他元素特点的方式得到。

特别注意到，电影和演员的简介信息在其长度不同的情况下，可能有不同的显示方式。

具体来说，长度较长时，可能会分在两个元素下，分别是带省略版，以及完整版。

这时需要先判断是否含有两个版本，再进行具体爬取，比如电影的简介爬取如下：

```python
item['intro'] = '\n'.join(selector.xpath('//div[@class="related-info"]/div[@class="indent"]/span[@class="all hidden"]//text()').extract()).strip()
if item['intro'] == '':
	item['intro'] = '\n'.join(selector.xpath('//span[@property="v:summary"]//text()').extract()).strip()
```

通过上面分析，就基本上可以解决信息的爬取。

对于一些中途出错的部分，也可以通过本地保存的 HTML 文件纠正，这部分主要由 `get_actors.py` 完成。

主要爬取的信息为：

- 电影信息：
  - 名称。
  - 图片网址。
  - 简介。
  - 导演。
  - 编剧。
  - 演员。
  - 类型。
  - 开映日期。
  - 时长。
  - 影评。
- 演员信息：
  - 豆瓣 ID。
  - 名字。
  - 性别。
  - 星座。
  - 生日。
  - 出生地。
  - 职业。
  - 简介。
  - 图片网址。

#### 2.2. 导入数据库

本项目采用的数据库为 MySQL，而导出的文件是 `.json` 文件，所以需要一个导入的过程，也就是 `input.py` 负责的部分。

首先通过终端创建了 `douban_movie` 这个数据库，在 `input.py` 通过 `pymysql` 连接此数据库，创建了四张表：

- 电影表（`movies`）：储存了其 ID，名称，图片地址，简介，开映日期，时长这些单一确定的信息。

- 演员表（`actors`）：储存了所有爬取得到的演员信息。

- 电影信息表（`movie_meta`）：通过 `movie_id` 确定信息属于哪部电影，通过 `meta_name` 确定信息类型：

  - `director`：导演。
  - `screenplay`：编剧。
  - `genre`：类型。
  - `review`：影评。
  - `actor`：演员。

  并通过 `meta_value` 确定信息对应的值。

  这些信息在一部电影中的数量不确定，所以另开了一张表记录。

- 演员合作表（`actor_cooperation`）：用于储存豆瓣 ID 分比为 `first_id` 和 `second_id` 的演员的合作次数 `count`。

  由于演员的合作相关信息难以在演员表中体现，所以新开一张表方便后续的查询。

对于 `pymysql` 库此处主要利用了其通过连接获取光标 `cursor()` 以及通过光标传入 SQL 指令 `execute(sql)` 两种方法，比如：

```python
c = pymysql.connect(host = 'localhost', user = 'root', password = '10900002', charset = 'utf8mb4', db = 'douban_movie')
cursor = c.cursor()
sql = '''
INSERT INTO movie_meta(movie_id, meta_name, meta_value)
VALUES
(%d, "%s", "%s");
''' % (id, "director", item)
cursor.execute(sql)
c.commit()
```

#### 2.3. 本地纠错

考虑到初始爬虫可能存在的部分小问题，需要通过后续对其进行修正。

`deal.py` 主要处理了，爬取影评过程中爬取的重复（简略版）或者无效（“展开”）信息。

`get_actors.py` 通过对于本地 HTML 文件的 `Selector` 处理了部分演员信息未能爬取到位的错误，并将演员信息重新输出到 `actors.json` 文件中。

`get_movies.py` 则负责将 `final_result.json` 中的电影数据部分提取输出到 `movies.json` 文件中。

#### 2.4. 爬虫数据总量

经过纠错、去重，最终共爬取了 $7834$ 条信息，其中：

- 电影 $1034$ 部，每部电影 $5$ 条短评，文件大小 $3682\,\text{KB}$。
- 演员 $6800$ 位，文件大小 $5678\,\text{KB}$。

### 3. 网站部分

#### 3.1. 网络路径分析

对于静态文件，通过 `settings.py` 文件设置：

```python
STATIC_URL = '/static/'

STATIC_ROOT = os.path.join(BASE_DIR, 'collected_static')

STATICFILES_DIRS = (
    os.path.join(BASE_DIR, "common_static"),
)

STATICFILES_FINDERS = (
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder"
)
```

其余路径均通过 `urls.py` 设置。

`douban_movie` 下的 `urls.py` 将空路径 `''` 映射给 `app` 中的 `urls.py`，而 `app` 中的 `urls.py` 则处理如下：

```python
urlpatterns = [
    path('', views.movie_index),
    path('movies/', views.movie_index),
    path('page/<int:id>/', views.movies),
    path('movies/page/<int:id>/', views.movies),
    path('actors/', views.actor_index),
    path('actors/page/<int:id>/', views.actors),
    path('movie/<int:id>/', views.movie),
    path('actor/<int:id>/', views.actor),
    path('movies/search/', views.movie_search),
    path('movies/search/page/<int:id>', views.movie_search_page),
    path('actors/search/', views.actor_search),
    path('actors/search/page/<int:id>', views.actor_search_page),
    path('reviews/search/', views.review_search),
    path('reviews/search/page/<int:id>', views.review_search_page),
]
```

即：

- 每个演员和电影拥有独一无二的 ID，用于其信息页，其中电影的 ID 为数据库自动生成，演员的 ID 沿用豆瓣 ID。
- 主页同样也就是电影列表页。
- 对于每种搜索，各有一个子路径以区分。
- 分页利用 `/page/<int:id>` 体现，而不是 GET 请求或者 POST 请求。

#### 3.2. 模板文件

在 `/app/template` 下存放所有用到的模板文件，在 `settings.py` 中设置好模板文件目录。

其中包含的模板文件主要为：

- `article.html`：用于包含其他模板文件，生成整个网页信息。
- `head.html`：用于负责 `<head>` 标签中的一些内容，包括静态文件引入与网页标题（浏览器标签处）设置。
- `css.html`：用于引入 `.css` 静态文件。
- `js.html`：用于引入 `.js` 静态文件，项目中只使用了 `jQuery` 库。
- `navigation-bar.html`：用于显示网站最上方的导航栏，以及搜索功能的部分。
- `title.html`：用于设置网页主体中的标题。
- `footer.html`：用于设置网页脚部信息。
- 其余文件：用于显示不同类型网页包括列表页、信息页、搜索结果页的网页主体。

#### 3.3. `templatetags` 文件夹

其下有一个 `rand.py`，注册了名为 `rand_style` 的 `filter`，用于在模板中使用到演员框或电影框时，随机为其选取一种样式。

也就是说，网页中每个信息框的样式是带随机的，会因浏览器刷新而改变。

#### 3.4. `common_static` 文件夹

储存了主要用到的静态文件，其中 `style.css` 是整个网页大部分元素的样式表，`css` 文件夹和 `webfonts` 文件夹则是用于 `fontawesome` 样式。

`images` 文件夹下包含了大部分用到的静态图片（包括网站背景、图标、搜索 Logo、搜索按钮图），除了用于显示无照片信息的演员照片文件 `none.png` 被直接放置在 `common_static` 下。

#### 3.5. 主要页面显示

通过 `/app/views.py` 生成网页。

对于每个网页，首先获取 `article.html` 模板，根据不同网页传入不同的 `type` 参数以方便其区分页面类型。

信息页根据网址中的 ID 查询数据库得到相关信息，对于不合法 ID 返回 $404$，对于合法者将其信息封装成字典传给模板方便模板生成网页。

例如：

```python
config = default
config.update({
    'type': 'movie',
    'title': movie[1],
    'url': request.get_host()
})
context = {
    'config': config,
    'movie': {
        'title': movie[1],
        'image_url': movie[2],
        'meta': meta,
        'intro': intro
    }
}
# ...
return HttpResponse(template.render(context, request))
```

列表页则根据 ID 获取数据库对应区间的内容，用以展示，对于不合法的 ID，同样返回 $404$。

例如：

```python
for i in range((id - 1) * actor_per_page, id * actor_per_page):
    if i == len(res):
        break
    image_url = res[i][9]
    if image_url.split('/')[-1] == 'avatar':
        image_url = '/static/none.png'
    actors.append({
        'url': '/actor/' + str(res[i][1]),
        'name': res[i][2],
        'image_url': image_url
    })
config = default
config.update({
    'type': 'actors',
    'title': '演员列表',
    'url': request.get_host()
})
l = range(1, all + 1) # all 是总页数
context = {
    'config': config,
    'all': all,
    'now': now,
    'list': l,
    'actors': actors
}
```

其中包括处理无图演员的信息部分。

对于搜索页，则通过 MySQL 的 `SELECT` 语句所带有的筛选功能进行查询。

#### 3.6. 分页

项目通过网址中的 `/page/<int:id>` 分页，获取信息在上面已经讲过。

对于网页中的页码栏部分，则通过传入模板参数解决，包括：

- `all`：总页数。
- `now`：当前要求页数，也就是 `id`。
- `list`：`range(1, all + 1)` 也就是所有合法页码的列表。
- `prev`：上一页页码。
- `next`：下一页页码。
- `prev_dots`：前面显示省略号的页码。
- `next_dots`：后面显示省略号的页码。

在模板中只须通过模板的 `if`、`for` 以及基本的关系判断运算符即可渲染得到页码栏。

对于跳转部分，通过增加 JavaScript 代码实现，例如对于电影列表页：

```html
<div class="any">
	<span>跳转至</span>
	<input type="number" id="jump-page" placeholder="1~{{all}}"/>
	<script>
		$('#jump-page').keydown(function(e) {
			if (e.keyCode == 13) {
				window.location.href = '/page/' + $('#jump-page').val();
			}
		})
	</script>
	<span>页</span>
</div>
```

其中利用了 `jQuery` 来获取对应元素。

#### 3.7. 搜索功能及其性能

搜索框架部分主体利用 `<form>` 表单提交 GET 请求，在后端通过 `request.GET.get()` 函数获取搜索信息。

对于不同字段的搜索，通过单选框的改变事件（JavaScript）来改变 `<form>` 表单的 `action` 地址，具体如下：

```html
<div class="nav-search">
	<select name="type" id="search-type">
	{% if config.search_type == 'movies' %}
		<option selected="selected" value="movies">电影</option>
	{% else %}
		<option value="movies">电影</option>
	{% endif %}
	{% if config.search_type == 'actors' %}
		<option selected="selected" value="actors">演员</option>
	{% else %}
		<option value="actors">演员</option>
	{% endif %}
	{% if config.search_type == 'reviews' %}
		<option selected="selected" value="reviews">影评</option>
	{% else %}
		<option value="reviews">影评</option>
	{% endif %}
	</select>
	<script>
		$('#search-type').change(function() {
			$('#search-form').attr('action', '/' + $(this).val() + '/search');
		})
	</script>
	<form action="/{{config.search_type}}/search" method="get" id="search-form">
		<fieldset>
			<div class="inp">
				<input id="inp-query" name="search_text" size="22" maxlength="20" placeholder="搜索电影、演员" value="{{search}}"/>
			</div>
			<div class="inp-btn">
				<input type="submit" value="搜索"/>
			</div>
		</fieldset>
	</form>
</div>
```

后端主要利用了 MySQL 的 `SELECT` 语句，一些使用如：

```mysql
select movie_id from movie_meta where meta_value like '%%%s%%' and (meta_name = 'directors' or meta_name = 'screenplay');
select b.movie_id from actors a, movie_meta b where a.douban_id = b.meta_value and b.meta_name = 'actor' and a.name like '%%%s%%';
```

其中 `%s` 用于填充搜索信息。

利用查询方式的优化以及数据库索引的建立，可以大大提升搜索效率，一般搜索均可以保证在 $0.2s$ 以内得出结果。

一些极端情况如（本项目特判了搜索内容空为不合法）：

- 电影搜索“a”，耗时约 $0.15s$。
- 演员搜索“a”，耗时 $0.12 \sim 0.19s$。
- 影评搜索“我”，耗时 $0.04s$。

#### 3.8. 数据库相关

由于 `pymysql` 库以及服务器的不稳定性，在连续多次访问服务器时，容易出现与 MySQL 服务器连接断开的情况，从而导致网页无法访问。

因此特别设计了 `MySQLManager` 类用于处理连接断开的情况：

```python
import pymysql
import traceback

class MySQLManager(object):

    def __init__(self, config):
        self.config = config
        self.c = pymysql.connect(**config)
        self.cur = self.c.cursor()

    def __del__(self):
        self.c.close()
        self.cur.close()

    def connect(self):
        self.c = pymysql.connect(**self.config)
        self.cur = self.c.cursor()

    def reconnect(self):
        try:
            self.c.ping()
        except:
            self.connect()

    def query(self, sql):
        try:
            self.reconnect()
            self.cur = self.c.cursor()
            with self.cur as cur:
                cur.execute(sql)
                result = cur.fetchall()
                self.c.commit()
                return result
        except Exception as e:
            print(e)
            traceback.print_exc()
```

其中利用 `query()` 进行查询时，先进行 `reconnect()` 也就是是否重连的判断，`reconnect()` 中通过 `try...except` 语句保证与数据库服务器的连接通畅。

#### 3.9. 网站总体界面

电影列表页：

![电影列表1](https://img.wzf2000.top/image/2020/09/23/1e9b39efa61e25131.png)

![电影列表2](https://img.wzf2000.top/image/2020/09/23/286100d3fe2e1c6b4.png)

演员列表页：

![演员列表2](https://img.wzf2000.top/image/2020/09/23/21629914d20e56605.png)

![演员列表1](https://img.wzf2000.top/image/2020/09/23/190fcd2e03b9432de.png)

电影信息页：

![电影信息1](https://img.wzf2000.top/image/2020/09/23/154e66777b02a7562.png)

![电影信息2](https://img.wzf2000.top/image/2020/09/23/2633b133ee533713a.png)

![电影信息3](https://img.wzf2000.top/image/2020/09/23/3446b079596e448a8.png)

演员信息页：

![演员信息1](https://img.wzf2000.top/image/2020/09/23/166e02d004bb36f2e.png)

![演员信息2](https://img.wzf2000.top/image/2020/09/23/2034b331342501104.png)

![演员信息3](https://img.wzf2000.top/image/2020/09/23/33588e13df813fb0d.png)

电影查询:

![电影查询](https://img.wzf2000.top/image/2020/09/23/e92b7313bb9f72a8405e8ff87c7bd586.png)

演员查询：

![演员查询](https://img.wzf2000.top/image/2020/09/23/bdabdd8bddafff17e0a2e1ca56a7c771.png)

影评查询：

![影评查询](https://img.wzf2000.top/image/2020/09/23/78b98e32d742423879ab285474a06307.png)

