from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from django.http import Http404
import pymysql
import timeit
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

manager = MySQLManager({
    'host': 'localhost',
    'user': 'root',
    'passwd': '10900002',
    'db': 'douban_movie',
    'charset': 'utf8mb4'
})

default = {
    'author': 'wzf2000',
    'search_type': 'movies'
}

movie_per_page = 20
actor_per_page = 25
review_per_page = 10

# Create your views here.

def movies(request, id):
    template = loader.get_template('article.html')
    sql = 'select count(*) from movies;'
    res = manager.query(sql)
    cnt = res[0][0]
    if (id - 1) * movie_per_page >= cnt or id <= 0:
        raise Http404("No more movies!")
    all = (cnt - 1) // movie_per_page + 1
    now = id
    sql = 'select * from movies;'
    res = manager.query(sql)
    movies = []
    for i in range((id - 1) * movie_per_page, id * movie_per_page):
        if i == len(res):
            break
        movies.append({
            'url': '/movie/' + str(res[i][0]),
            'title': res[i][1],
            'image_url': res[i][2]
        })
    config = default
    config.update({
        'type': 'movies',
        'title': '电影列表',
        'url': request.get_host()
    })
    l = range(1, all + 1)
    context = {
        'config': config,
        'all': all,
        'now': now,
        'list': l,
        'movies': movies
    }
    if id > 1:
        context['prev'] = id - 1
    context['prev_dots'] = id - 3
    context['next_dots'] = id + 3
    if id < all:
        context['next'] = id + 1
    return HttpResponse(template.render(context, request))

def movie_index(request):
    return movies(request, 1)

def actors(request, id):
    template = loader.get_template('article.html')
    sql = 'select count(*) from actors;'
    res = manager.query(sql)
    cnt = res[0][0]
    if (id - 1) * actor_per_page >= cnt or id <= 0:
        raise Http404("No more actors!")
    all = (cnt - 1) // actor_per_page + 1
    now = id
    sql = 'select * from actors;'
    res = manager.query(sql)
    actors = []
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
    l = range(1, all + 1)
    context = {
        'config': config,
        'all': all,
        'now': now,
        'list': l,
        'actors': actors
    }
    if id > 1:
        context['prev'] = id - 1
    context['prev_dots'] = id - 3
    context['next_dots'] = id + 3
    if id < all:
        context['next'] = id + 1
    return HttpResponse(template.render(context, request))

def actor_index(request):
    return actors(request, 1)

def movie(request, id):
    template = loader.get_template('article.html')
    sql = "select * from movies where id = " + str(id) +";"
    res = manager.query(sql)
    if len(res) == 0:
        raise Http404("Movie of this id does not exist")
    movie = res[0]
    meta = []
    info = movie[3].split('\n')
    intro = []
    for item in info:
        item = item.strip()
        intro.append(item)
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'director';" % id
    res = manager.query(sql)
    dir = []
    for item in res:
        dir.append(item[0])
    if len(dir) != 0:
        meta.append({
            'name': '导演',
            'value': ' / '.join(dir)
        })
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'screenplay';" % id
    res = manager.query(sql)
    scr = []
    for item in res:
        scr.append(item[0])
    if len(scr) != 0:
        meta.append({
            'name': '编剧',
            'value': ' / '.join(scr)
        })
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'genre';" % id
    res = manager.query(sql)
    gen = []
    for item in res:
        gen.append(item[0])
    if len(gen) != 0:
        meta.append({
            'name': '类型',
            'value': ' / '.join(gen)
        })
    if movie[4] != '':
        meta.append({
            'name': '上映日期',
            'value': movie[4]
        })
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'actor';" % id
    res = manager.query(sql)
    act = []
    for item in res:
        sql = "select name from actors where douban_id = %s;" % item[0]
        result = manager.query(sql)
        act.append(result[0][0])
    if len(act) != 0:
        meta.append({
            'name': '主演',
            'value': ' / '.join(act)
        })
    if movie[4] != '':
        meta.append({
            'name': '上映日期',
            'value': movie[4]
        })
    if movie[5] != '':
        meta.append({
            'name': '时长',
            'value': movie[5]
        })
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
    context['movie']['reviews'] = []
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'review';" % id
    res = manager.query(sql)
    for item in res:
        context['movie']['reviews'].append(item[0])
    context['movie']['actors'] = []
    sql = "select meta_value from movie_meta where movie_id = %d and meta_name = 'actor';" % id
    res = manager.query(sql)
    for item in res:
        sql = "select name, image_url from actors where douban_id = %s;" % item[0]
        res = manager.query(sql)
        ans = res[0]
        image_url = ans[1]
        if image_url.split('/')[-1] == 'avatar':
            image_url = '/static/none.png'
        context['movie']['actors'].append({
            'name': ans[0],
            'image_url': image_url,
            'url': '/actor/' + str(item[0])
        })
    return HttpResponse(template.render(context, request))

def actor(request, id):
    template = loader.get_template('article.html')
    sql = "select * from actors where douban_id = " + str(id) +";"
    res = manager.query(sql)
    if len(res) == 0:
        raise Http404("Actor of this id does not exist")
    actor = res[0]
    meta = []
    if actor[3] != '':
        meta.append({
            'name': '性别',
            'value': actor[3]
        })
    if actor[4] != '':
        meta.append({
            'name': '星座',
            'value': actor[4]
        })
    if actor[5] != '':
        meta.append({
            'name': '生日',
            'value': actor[5]
        })
    if actor[6] != '':
        meta.append({
            'name': '出生地',
            'value': actor[6]
        })
    if actor[7] != '':
        meta.append({
            'name': '职业',
            'value': actor[7]
        })
    info = actor[8].split('\n')
    intro = []
    for item in info:
        item = item.strip()
        intro.append(item)
    image_url = actor[9]
    if image_url.split('/')[-1] == 'avatar':
        image_url = '/static/none.png'
    config = default
    config.update({
        'type': 'actor',
        'title': actor[2],
        'url': request.get_host()
    })
    context = {
        'config': config,
        'actor': {
            'name': actor[2],
            'image_url': image_url,
            'meta': meta,
            'intro': intro
        }
    }
    context['actor']['movies'] = []
    sql = "select movie_id from movie_meta where meta_name= 'actor' and meta_value = '%d';" % id
    res = manager.query(sql)
    for item in res:
        sql = "select title, image_url from movies where id = %d;" % item[0]
        res = manager.query(sql)
        ans = res[0]
        context['actor']['movies'].append({
            'title': ans[0],
            'image_url': ans[1],
            'url': '/movie/' + str(item[0])
        })
    context['actor']['cooperators'] = []
    sql = "select second_id, count from actor_cooperation where first_id = %d order by count desc;" % id
    res = manager.query(sql)[:10]
    for item in res:
        sql = "select name, image_url from actors where douban_id = %d;" % item[0]
        res = manager.query(sql)
        ans = res[0]
        image_url = ans[1]
        if image_url.split('/')[-1] == 'avatar':
            image_url = '/static/none.png'
        context['actor']['cooperators'].append({
            'name': ans[0],
            'image_url': image_url,
            'url': '/actor/' + str(item[0]),
            'count': item[1]
        })
    return HttpResponse(template.render(context, request))

def movie_search_page(request, id):
    template = loader.get_template('article.html')
    req = request.GET.get('search_text', default = "").strip()
    req_orig = req
    req = req.replace('\\', '\\\\').replace('"','\\\"').replace("'",'\\\'')
    if req == "":
        config = default
        config.update({
            'type': 'search_movies',
            'search_type': 'movies',
            'title': '请输入内容再搜索！',
            'url': request.get_host(),
            'argv': '?search_text=' + req_orig
        })
        context = {
            'config': config,
            'search': req_orig,
            'error': True
        }
        return HttpResponse(template.render(context, request))
    start = timeit.default_timer()
    sql = "select id from movies where title like '%%%s%%';" % req
    res = manager.query(sql)
    movie_id = []
    for item in res:
        movie_id.append(item[0])
    sql = "select movie_id from movie_meta where meta_value like '%%%s%%' and (meta_name = 'directors' or meta_name = 'screenplay');" % req
    res = manager.query(sql)
    for item in res:
        movie_id.append(item[0])
    sql = "select b.movie_id from actors a, movie_meta b where a.douban_id = b.meta_value and b.meta_name = 'actor' and a.name like '%%%s%%';" % req
    res = manager.query(sql)
    for movie in res:
        movie_id.append(movie[0])
    movie_id = list(set(movie_id))
    cnt = len(movie_id)
    all = (cnt - 1) // movie_per_page + 1
    if (id > all or id <= 0) and all > 0:
        raise Http404("No more search result!")
    if all == 0:
        config = default
        config.update({
            'type': 'search_movies',
            'search_type': 'movies',
            'title': '没有找到相关结果！',
            'url': request.get_host()
        })
        end = timeit.default_timer()
        context = {
            'config': config,
            'search': req_orig,
            'count': cnt,
            'time': round(end - start, 2),
            'argv': '?search_text=' + req_orig
        }
        return HttpResponse(template.render(context, request))
    movies = []
    for movie in movie_id[(id - 1) * movie_per_page : id * movie_per_page]:
        sql = "select title, image_url from movies where id = %d;" % movie
        res = manager.query(sql)
        ans = res[0]
        movies.append({
            'title': ans[0],
            'image_url': ans[1],
            'url': '/movie/' + str(movie)
        })
    config = default
    config.update({
        'type': 'search_movies',
        'search_type': 'movies',
        'title': "“" + req_orig + "”" + '的搜索结果',
        'url': request.get_host()
    })
    l = range(1, all + 1)
    end = timeit.default_timer()
    context = {
        'config': config,
        'movies': movies,
        'search': req_orig,
        'time': round(end - start, 2),
        'all': all,
        'now': id,
        'list': l,
        'count': cnt,
        'argv': '?search_text=' + req_orig
    }
    if id > 1:
        context['prev'] = id - 1
    context['prev_dots'] = id - 3
    context['next_dots'] = id + 3
    if id < all:
        context['next'] = id + 1
    return HttpResponse(template.render(context, request))

def movie_search(request):
    return movie_search_page(request, 1)

def actor_search_page(request, id):
    template = loader.get_template('article.html')
    req = request.GET.get('search_text', default = "").strip()
    req_orig = req
    req = req.replace('\\', '\\\\').replace('"','\\\"').replace("'",'\\\'')
    if req == "":
        config = default
        config.update({
            'type': 'search_actors',
            'search_type': 'actors',
            'title': '请输入内容再搜索！',
            'url': request.get_host(),
            'argv': '?search_text=' + req_orig
        })
        context = {
            'config': config,
            'search': req_orig,
            'error': True
        }
        return HttpResponse(template.render(context, request))
    start = timeit.default_timer()
    sql = "select douban_id from actors where name like '%%%s%%';" % req
    res = manager.query(sql)
    actor_id = []
    for item in res:
        actor_id.append(item[0])
    sql = "select b.meta_value from movies a, movie_meta b where a.title like '%%%s%%' and a.id = b.movie_id and b.meta_name = 'actor';" % req
    res = manager.query(sql)
    for actor in res:
        actor_id.append(int(actor[0]))
    actor_id = list(set(actor_id))
    cnt = len(actor_id)
    all = (cnt - 1) // actor_per_page + 1
    if (id > all or id <= 0) and all > 0:
        raise Http404("No more search result!")
    if all == 0:
        config = default
        config.update({
            'type': 'search_actors',
            'search_type': 'actors',
            'title': '没有找到相关结果！',
            'url': request.get_host()
        })
        end = timeit.default_timer()
        context = {
            'config': config,
            'search': req_orig,
            'count': cnt,
            'time': round(end - start, 2),
            'argv': '?search_text=' + req_orig
        }
        return HttpResponse(template.render(context, request))
    actors = []
    for actor in actor_id[(id - 1) * actor_per_page : id * actor_per_page]:
        sql = "select name, image_url from actors where douban_id = %d;" % actor
        res = manager.query(sql)
        ans = res[0]
        image_url = ans[1]
        if image_url.split('/')[-1] == 'avatar':
            image_url = '/static/none.png'
        actors.append({
            'name': ans[0],
            'image_url': image_url,
            'url': '/actor/' + str(actor)
        })
    config = default
    config.update({
        'type': 'search_actors',
        'search_type': 'actors',
        'title': "“" + req_orig + "”" + '的搜索结果',
        'url': request.get_host()
    })
    l = range(1, all + 1)
    end = timeit.default_timer()
    context = {
        'config': config,
        'actors': actors,
        'search': req_orig,
        'time': round(end - start, 2),
        'all': all,
        'now': id,
        'list': l,
        'count': cnt,
        'argv': '?search_text=' + req_orig
    }
    if id > 1:
        context['prev'] = id - 1
    context['prev_dots'] = id - 3
    context['next_dots'] = id + 3
    if id < all:
        context['next'] = id + 1
    return HttpResponse(template.render(context, request))

def actor_search(request):
    return actor_search_page(request, 1)

def review_search_page(request, id):
    template = loader.get_template('article.html')
    req = request.GET.get('search_text', default = "").strip()
    req_orig = req
    req = req.replace('\\', '\\\\').replace('"','\\\"').replace("'",'\\\'')
    if req == "":
        config = default
        config.update({
            'type': 'search_reviews',
            'search_type': 'reviews',
            'title': '请输入内容再搜索！',
            'url': request.get_host(),
            'argv': '?search_text=' + req_orig
        })
        context = {
            'config': config,
            'search': req_orig,
            'error': True
        }
        return HttpResponse(template.render(context, request))
    start = timeit.default_timer()
    sql = "select movie_id, meta_value from movie_meta where meta_name = 'review' and meta_value like '%%%s%%';" % req
    res = manager.query(sql)
    cnt = len(res)
    all = (cnt - 1) // review_per_page + 1
    if (id > all or id <= 0) and all > 0:
        raise Http404("No more search result!")
    if all == 0:
        config = default
        config.update({
            'type': 'search_reviews',
            'search_type': 'reviews',
            'title': '没有找到相关结果！',
            'url': request.get_host()
        })
        end = timeit.default_timer()
        context = {
            'config': config,
            'search': req_orig,
            'count': cnt,
            'time': round(end - start, 2),
            'argv': '?search_text=' + req_orig
        }
        return HttpResponse(template.render(context, request))
    reviews = []
    for item in res[(id - 1) * review_per_page : id * review_per_page]:
        reviews.append({
            'text': item[1],
            'url': '/movie/' + str(item[0])
        })
    config = default
    config.update({
        'type': 'search_reviews',
        'search_type': 'reviews',
        'title': "“" + req_orig + "”" + '的搜索结果',
        'url': request.get_host()
    })
    l = range(1, all + 1)
    end = timeit.default_timer()
    context = {
        'config': config,
        'reviews': reviews,
        'search': req_orig,
        'time': round(end - start, 2),
        'all': all,
        'now': id,
        'list': l,
        'count': cnt,
        'argv': '?search_text=' + req_orig
    }
    if id > 1:
        context['prev'] = id - 1
    context['prev_dots'] = id - 3
    context['next_dots'] = id + 3
    if id < all:
        context['next'] = id + 1
    return HttpResponse(template.render(context, request))

def review_search(request):
    return review_search_page(request, 1)
