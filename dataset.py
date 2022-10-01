import json
import csv
from collections import defaultdict
from tqdm.auto import tqdm
from pprint import pprint

datafile = 'data/train/rating.csv'
outfile = 'data/train/movie.json'






# if __name__ == '__main__':
#     cal_num_tags()




# with open(outfile,'r') as f:
# #     dic = json.load(f)
# #     print(dic['0']['138'])


# with open(datafile,'r') as f,open(outfile,'w') as ofile:
#     lines = f.readlines()[1:]
#     # print(lines[0])
#     user2movie2rating = defaultdict(defaultdict)
#     for line in lines:
#         line = line.strip().split(',')
#         tmpline = [int(id) for id in line]
#         user2movie2rating[tmpline[0]][tmpline[1]] = tmpline[2]
#         # print(user2movie2rating)
#     json.dump(user2movie2rating,ofile)




# with open(datafile,'r') as f,open(outfile,'w') as ofile:
#     lines = f.readlines()[1:]
#     # print(lines[0])
#     tupleofuserandmovie2tags = defaultdict(list)
#     for line in lines:
#         line = line.strip().split(',')
#         tmpline = [int(id) for id in line]
#         # print(tmpline)
#         tupleofuserandmovie2tags[str((tmpline[0],tmpline[1]))].append(tmpline[2])
#         # print(tupleofuserandmovie2tags)
#     json.dump(tupleofuserandmovie2tags,ofile)








# with open(datafile,'r') as f,open(outfile,'w') as ofile:
#     lines = f.readlines()[1:]
#     movie2tag = defaultdict(list)
#     for line in tqdm(lines):
#         line = line.strip().split(',')
#         for _ in range(len(line)):
#             line[_] = line[_].strip('["]')
#         tmpline = [int(id) for id in line]
#         movie2tag[tmpline[0]] = tmpline[1:]
#         # print(movie2tag)
#     json.dump(movie2tag,ofile)

# with open(outfile,'r') as f:
#     dic = json.load(f)
#     print(dic)

    # line = lines[0]
    # print(line)
    # line = line.strip().split(',')
    # for _ in range(len(line)):
    #     line[_] = line[_].strip('["]')
    # print(line)





