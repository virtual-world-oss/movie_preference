# movie_preference

#### 2022/10/1

​	主要做了一些数据集的处理和统计，新增文件如下：

- ​	**data/train/movie.json**：格式为{movie_id(str) : [tags_id] (list)}
- ​    **data/train/tupleofuserandmovie2tags_from_obstag.json**:格式为{”（user_id,movie_id）“ (str) : [tags_id] (list)},这个文件统计了一个用户-电影对中，用户再这个电影中喜欢的tags，其余tags也有可能喜欢。
- ​    **data/train/tupleofuserandmovie2tags_from_rcttag.json**:格式为{”user_id,movie_id）“ (str) : [tags_id] (list)“},这个文件统计了一个用户电影对中，用户在这个电影中喜欢的tags，这个电影中的其余tags一定不喜欢。
- ​    **data/train/user2movie2rating.json**:格式为{”user_id" (str) : { "movie_id" (str) : rating (int)}},这个文件统计了一个用户，对一个电影的评分。
- ​    **data/train/user2tagsdislike.json**:格式为{”user_id“(str) : [tags] (list)} 这个文件统计了某些用户一定不喜欢的tags
-    **data/train/user2tagslike.json**:格式为{”user_id“(str) : [tags] (list)} 这个文件统计了每个用户一定喜欢的tags



​	dataset.py和utils.py主要是一些临时代码，可以删除。

