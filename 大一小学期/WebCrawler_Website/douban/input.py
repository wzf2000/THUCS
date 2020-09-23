import pymysql
import json

c = pymysql.connect(host = 'localhost', user = 'root', password = '10900002', charset = 'utf8mb4', db = 'douban_movie')
cursor = c.cursor()

sql = '''
CREATE TABLE movies( 
    id INT NOT NULL AUTO_INCREMENT, 
    title TINYTEXT NOT NULL, 
    image_url VARCHAR(200), 
    intro LONGTEXT, 
    initialReleaseDate TEXT,
    runtime TEXT,
    PRIMARY KEY ( id )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
'''

cursor.execute(sql)

sql = '''
CREATE TABLE actors( 
    id INT NOT NULL AUTO_INCREMENT, 
    douban_id INT,
    name TEXT NOT NULL, 
    sex TINYTEXT,
    constellation TEXT,
    birthday TEXT,
    birth_place TEXT,
    profession TEXT,
    intro LONGTEXT,
    image_url VARCHAR(200), 
    PRIMARY KEY ( id )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
'''

cursor.execute(sql)

sql = '''
CREATE TABLE movie_meta(
    id INT NOT NULL AUTO_INCREMENT, 
    movie_id INT NOT NULL,
    meta_name TINYTEXT NOT NULL,
    meta_value TEXT, 
    PRIMARY KEY ( id )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
'''

cursor.execute(sql)

sql = '''
CREATE TABLE actor_cooperation(
    id INT NOT NULL AUTO_INCREMENT, 
    first_id INT NOT NULL,
    second_id INT NOT NULL,
    count INT NOT NULL, 
    PRIMARY KEY ( id )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
'''

cursor.execute(sql)

file_name = "movies.json"

cnt = {}

with open(file_name, 'r', encoding = 'utf8') as f:
    while True:
        line = f.readline()
        if not line:
            break
        info = json.loads(line)
        if info['type'] == 'actor':
            sql = '''
            INSERT INTO actors(douban_id, name, sex, constellation, birthday, birth_place, profession, intro, image_url)
            VALUES
            (%s, "%s", "%s", "%s", "%s", "%s", "%s", "%s", "%s");
            ''' % ("" if 'id' not in info else info['id'], "" if 'name' not in info else info['name'], "" if 'sex' not in info else info['sex'], "" if 'constellation' not in info else info['constellation'], "" if 'birthday' not in info else info['birthday'], "" if 'birth_place' not in info else info['birth_place'], "" if 'profession' not in info else info['profession'], ("" if 'intro' not in info else info['intro']).replace('\\', '\\\\').replace('"','\\\"'), "" if 'image_url' not in info else info['image_url'])
            cursor.execute(sql)
            c.commit()
        elif info['type'] == 'movie':
            sql = '''
            INSERT INTO movies(title, image_url, intro, initialReleaseDate, runtime)
            VALUES
            ("%s", "%s", "%s", "%s", "%s");
            ''' % ("" if 'title' not in info else info['title'], "" if 'image_url' not in info else info['image_url'], ("" if 'intro' not in info else info['intro']).replace('\\', '\\\\').replace('"','\\\"'), "" if 'initialReleaseDate' not in info else info['initialReleaseDate'], "" if 'runtime' not in info else info['runtime'])
            cursor.execute(sql)
            c.commit()
            id = cursor.lastrowid
            if 'director' in info:
                for item in info['director']:
                    sql = '''
                    INSERT INTO movie_meta(movie_id, meta_name, meta_value)
                    VALUES
                    (%d, "%s", "%s");
                    ''' % (id, "director", item)
                    cursor.execute(sql)
                    c.commit()
            if 'screenplay' in info:
                for item in info['screenplay']:
                    sql = '''
                    INSERT INTO movie_meta(movie_id, meta_name, meta_value)
                    VALUES
                    (%d, "%s", "%s");
                    ''' % (id, "screenplay", item)
                    cursor.execute(sql)
                    c.commit()
            if 'genre' in info:
                for item in info['genre']:
                    sql = '''
                    INSERT INTO movie_meta(movie_id, meta_name, meta_value)
                    VALUES
                    (%d, "%s", "%s");
                    ''' % (id, "genre", item)
                    cursor.execute(sql)
                    c.commit()
            if 'review' in info:
                for item in info['review']:
                    sql = '''
                    INSERT INTO movie_meta(movie_id, meta_name, meta_value)
                    VALUES
                    (%d, "%s", "%s");
                    ''' % (id, "review", item.replace('\\', '\\\\').replace('"','\\\"'))
                    cursor.execute(sql)
                    c.commit()
            if 'actors' in info:
                for actor1 in info['actors']:
                    sql = '''
                    INSERT INTO movie_meta(movie_id, meta_name, meta_value)
                    VALUES
                    (%d, "%s", "%s");
                    ''' % (id, "actor", actor1['id'])
                    cursor.execute(sql)
                    c.commit()
                    for actor2 in info['actors']:
                        if actor1 == actor2:
                            continue
                        if actor1['id'] not in cnt:
                            cnt[actor1['id']] = {}
                        if actor2['id'] not in cnt[actor1['id']]:
                            cnt[actor1['id']][actor2['id']] = 0
                        cnt[actor1['id']][actor2['id']] += 1

for id1, item in cnt.items():
    for id2, Cnt in item.items():
        sql = '''
        INSERT INTO actor_cooperation(first_id, second_id, count)
        VALUES
        (%s, %s, %d);
        ''' % (id1, id2, Cnt)
        cursor.execute(sql)
        c.commit()


cursor.close()
c.close()