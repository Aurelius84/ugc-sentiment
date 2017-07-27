#!/bash/bin

in_file='../docs/shenyang_20_cluster_detail.txt'
out_file='../docs/shenyang_20_topic.txt'

# 贩毒
awk '/贩毒|黑帮|安平/' $in_file > $out_file

# 吉涛
awk '/吉涛|拆迁/' $in_file >> $out_file

# 反腐
awk '/反腐/' $in_file >> $out_file

# 盗窃份子
awk '/盗窃份子/' $in_file >> $out_file

# 宗教
awk '/宗教/' $in_file >> $out_file

# 七年 信访
awk '/七年/' $in_file >> $out_file

#统计时间分布
