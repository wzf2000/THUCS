import pymysql
import json

c = pymysql.connect(host = 'localhost', user = 'root', password = '10900002', charset = 'utf8mb4', db = 'douban_movie')
cursor = c.cursor()
cnt = 0

for i in range(1, 21264):
    sql = "select meta_value from movie_meta where id=%d;" % i
    cursor.execute(sql)
    res = cursor.fetchall()
    if len(res) == 0:
        continue
    s1 = res[0][0]
    if len(s1) < 10:
        continue
    sql = "select meta_value from movie_meta where id=%d;" % (i + 1)
    cursor.execute(sql)
    res = cursor.fetchall()
    if len(res) == 0:
        continue
    s2 = res[0][0]
    s3 = s1[:-3]
    if s2[:len(s3)] == s3:
        sql = "delete from movie_meta where id=%d;" % i
        cursor.execute(sql)
        c.commit()