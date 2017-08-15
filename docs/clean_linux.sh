#!/bash/bin

in_file='../docs/sensitive_20.txt'

# rm <.*>
sed -i 's/<[^>@]*>//g' $in_file

# rm &nbsp;空格 正文
sed -i 's/&nbsp;//g' $in_file

# rm 空格
sed -i 's/ //g' $in_file

# rm 正文
sed -i 's/正文//g' $in_file

# rm http
sed -i 's/http:[a-zA-Z0-9\.\?\/]*//g' $in_file

# rm @丹东隋宝全:
sed -i 's/@[^@:\/\|]*://g' $in_file

# rm /
sed -i 's/\///g' $in_file

# rm @凤凰视频客户端@中国青年网@中央新闻
for i in {1..6}
do
  sed -i 's/@[^@\|\:\.\?]\{2,13\}@/@/g' $in_file
done

# rm @中国青年网|
# sed -i 's/@[^@\|\:\.\?]\{2,13\}\|/\|/g' $in_file

# 沈阳地区关键词
# 沈阳|盛京|奉天|大东|浑南|东陵|法库县|和平区|皇姑|康平区|辽中县|辽中区|沈北新区|沈河区|苏家屯|铁西|新民市|于洪区|昭陵|福陵|张氏帅府|怪坡|棋盘山|中共满洲省委旧址|周恩来少年读书旧址|辽滨塔|辽宁省博物馆|铸造博物馆|新乐遗址|九一八历史博物馆|彩电塔|中街|太原街|五爱市场|三好街|福陵叠翠|御苑松涛|浑河晚渡|塔湾夕照|柳塘春雨|道院秋风|神碑幻影|陡山霁雪|凤楼观塔|万泉垂钓|森林野生动物园|世博园|东北亚滑雪场|皇家极地海洋世界|方特欢乐世界|紫烟薰衣草庄园|爱琴谷|陨石山|刘老根大舞台|南关天主教堂|浑河西峡谷|永安桥|沈大高速|京沈高速|沈哈高速|沈吉高速|沈丹高速|沈彰高速|沈康高速|绕城高速|辽宁中部环线高速|长客总站|长客西站|辽宁省快速汽车客运站|北站长途客运站|南塔客运站|五爱客货联运总站|东北大学|辽宁大学|中国医科大学|辽宁中医药大学|中国刑事警察学院|鲁迅美术学院|辽宁何氏医学院|辽宁传媒学院|杏林学院|海华学院