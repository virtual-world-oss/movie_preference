import json
import csv
from collections import defaultdict
from tqdm.auto import tqdm
from pprint import pprint
import numpy as np


def cal_num_tags():
    all_tags = set()
    with open('data/train/movie.json','r') as f:
        movie2tag = json.load(f)
        tags = movie2tag.values()
        for subset in tags:
            all_tags |= set(subset)
            # print(all_tags)
        num_tags = len(all_tags)
        tags = list(all_tags)
        return num_tags,tags

def calmatrixforuser2tags():
    user2tags = np.zeros((2022, cal_num_tags()))
    pass

def calmatrixfortags2tags():
    pass

def caluser2tagsfordislike():
    def str2tupleforkey(key):
        key = key.strip('"')
        key = key.lstrip('(')
        key = key.rstrip(')')
        key = key.split(',')
        key = [int(id) for id in key]
        return key
    user2disliketags = defaultdict(set)
    with open('data/train/tupleofuserandmovie2tags_from_rcttag.json', 'r') as f,open('data/train/movie.json') as movief:
        tupleofuserandmovie2tags = json.load(f)
        movie2tag = json.load(movief)
        # print(len(tupleofuserandmovie2tags.keys()))
        for item in tupleofuserandmovie2tags:
            # print(item)
            tupleitem = str2tupleforkey(item)
            # print(tupleitem)
            userid = tupleitem[0]
            movieid = tupleitem[1]
            tagsinthismovie = movie2tag[str(movieid)]
            tagsthisuserlike = tupleofuserandmovie2tags[item]
            # print(tagsthisuserlike)
            # print(tagsinthismovie)
            tagsuserdislikeinthismovie = set(tagsinthismovie) - set(tagsthisuserlike)
            # print(tagsuserdislikeinthismovie)
            user2disliketags[userid] |= tagsuserdislikeinthismovie
            # print(user2disliketags)
    for item in user2disliketags:
        user2disliketags[item] = list(user2disliketags[item])
    with open('data/train/user2tagsdislike.json', 'w') as f:
        json.dump(user2disliketags,f)
    pass

def caluser2tagslike():
    def str2tupleforkey(key):
        key = key.strip('"')
        key = key.lstrip('(')
        key = key.rstrip(')')
        key = key.split(',')
        key = [int(id) for id in key]
        return key
    pass
    user2liketags = defaultdict(set)
    with open('data/train/tupleofuserandmovie2tags_from_rcttag.json', 'r') as rctf, open('data/train/movie.json') as movief,open('data/train/tupleofuserandmovie2tags_from_obstag.json') as obsf:
        tupleofuserandmovie2tagsinobs = json.load(obsf)
        movie2tag = json.load(movief)
        tupleofuserandmovie2tagsinrct = json.load(rctf)
        # print(len(tupleofuserandmovie2tags.keys()))
        for item in tupleofuserandmovie2tagsinobs:
            # print(item)
            tupleitem = str2tupleforkey(item)
            # print(tupleitem)
            userid = tupleitem[0]
            movieid = tupleitem[1]
            tagsinthismovie = movie2tag[str(movieid)]
            tagsthisuserlike = tupleofuserandmovie2tagsinobs[item]
            # print(tagsthisuserlike)
            # print(tagsinthismovie)
            # print(tagsuserdislikeinthismovie)
            if len(tagsthisuserlike) == 1 and tagsthisuserlike[0] == -1:
                continue
            user2liketags[userid] |= set(tagsthisuserlike)
            # print(user2disliketags)
            # print(user2liketags)
        for item in tupleofuserandmovie2tagsinrct:
            # print(item)
            tupleitem = str2tupleforkey(item)
            # print(tupleitem)
            userid = tupleitem[0]
            movieid = tupleitem[1]
            tagsinthismovie = movie2tag[str(movieid)]
            tagsthisuserlike = tupleofuserandmovie2tagsinrct[item]
            # print(tagsthisuserlike)
            # print(tagsinthismovie)
            # print(tagsuserdislikeinthismovie)
            if len(tagsthisuserlike) == 1 and tagsthisuserlike[0] == -1:
                continue
            user2liketags[userid] |= set(tagsthisuserlike)
            # print(user2disliketags)
            # print(user2liketags)
    for item in user2liketags:
        user2liketags[item] = list(user2liketags[item])
    with open('data/train/user2tagslike.json', 'w') as f:
        json.dump(user2liketags,f)



def caltagsid2movieid():
    tags2movie = defaultdict(list)
    with open('data/train/movie.json','r') as f:
        movie2tags = json.load(f)
        for item in movie2tags:
            movieid = int(item)
            # print(movieid)
            tags = movie2tags[item]
            # print(tags)
            for tag in tags:
                tags2movie[tag].append(movieid)
            # print(tags2movie)
    for item in tags2movie:
        tags2movie[item] = list(set(tags2movie[item]))
    # print(len(tags2movie))
    # print(cal_num_tags())
    # print(len(tags2movie[0]))
    with open('data/train/tags2movie.json','w') as f:
        json.dump(tags2movie,f)
    pass

def caltagsmatrix():
    tagsmatrix = np.zeros((cal_num_tags()[0],cal_num_tags()[0]))
    print(type(tagsmatrix))
    # print(tagsmatrix.shape)
    with open('data/train/tags2movie.json','r') as f:
        tag2movies = json.load(f)
    for rowid in range(cal_num_tags()[0]):
        for columnid in range(cal_num_tags()[0]):
            if rowid == columnid:
                tagsmatrix[rowid][columnid] = -1
            else:
                common_movie_num = len(set(tag2movies[str(rowid)]) & set(tag2movies[str(columnid)]))
                # print(set(tag2movies[str(rowid)]))
                # print(set(tag2movies[str(columnid)]))
                # print(set(tag2movies[str(rowid)]) & set(tag2movies[str(columnid)]))
                tagsmatrix[rowid][columnid] = common_movie_num
    outfile = 'data/train/datatagsmatrix.npy'
    np.save(outfile,tagsmatrix)
    # print(tagsmatrix)
    # print(tagsmatrix.shape)
    # print(tagsmatrix[0][1])
    pass


if __name__ == '__main__':
    pass
    # print(cal_num_tags())


    # caltagsid2movieid()
    # outfile = 'data/train/datatagsmatrix.npy'
    # # caltagsmatrix()
    # tagsmatrix = np.load(outfile)
    # print(tagsmatrix)
    # print(tagsmatrix.shape)
    # print(tagsmatrix[0][1])




    # num_tags, tags = cal_num_tags()
    # print(tags)
    # print(num_tags)
    # print(max(tags))
    # caluser2tagsfordislike()
    # with open('data/train/user2tagsdislike.json','r') as f:
    #     dic = json.load(f)
    #     print(len(dic))
    #     for item in dic:
    #         noe = dic[item]
    #         print(len(noe))
    # caluser2tagslike()
    # with open('data/train/user2tagslike.json','r') as f:
    #     dic = json.load(f)
    #     print(len(dic))
    #     for item in dic:
    #         noe = dic[item]
    #         print(len(noe))





    # num_user = 2022
    # num_tags,tags = cal_num_tags()
    # len_matrix = num_user * num_tags
    # with open('data/train/user2tagslike.json', 'r') as likef,open('data/train/user2tagsdislike.json', 'r') as dislikef:
    #     user2liketags = json.load(likef)
    #     user2disliketags = json.load(dislikef)
    #     num_real = 0
    #     for userid in range(num_user):
    #         num_real += len(user2liketags[str(userid)])
    #         num_real += len(user2disliketags[str(userid)]) if str(userid) in user2disliketags else 0
    # print(f'total matrix elements: {len_matrix}, have data elements: {num_real}, ratio: {num_real / len_matrix}')


