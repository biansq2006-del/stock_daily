#!/bin/bash

# 记录开始时间
echo "Starting update at $(date)" >> /Users/saviour/Desktop/stock/update_log.txt

# 1. 进入文件夹
cd /Users/saviour/Desktop/stock/

# 2. 运行 Python (已修改为您新电脑的 Anaconda Python 路径)
/opt/anaconda3/bin/python3 stock_html.py >> /Users/saviour/Desktop/stock/update_log.txt 2>&1

# 3. Git 操作 (确保使用完整路径)
/usr/bin/git add index.html
/usr/bin/git commit -m "daily auto update" >> /Users/saviour/Desktop/stock/update_log.txt 2>&1
/usr/bin/git push >> /Users/saviour/Desktop/stock/update_log.txt 2>&1

echo "Finished update at $(date)" >> /Users/saviour/Desktop/stock/update_log.txt
echo "-----------------------------------" >> /Users/saviour/Desktop/stock/update_log.txt