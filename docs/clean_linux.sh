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
