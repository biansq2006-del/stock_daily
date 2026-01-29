#!/bin/bash

# 记录开始时间
echo "Starting update at $(date)" >> /Users/songqin.bian/Desktop/stock/update_log.txt

# 1. 进入文件夹
cd /Users/songqin.bian/Desktop/stock/

# 2. 运行 Python (确保使用完整路径)
/usr/local/bin/python3 stock_html.py >> /Users/songqin.bian/Desktop/stock/update_log.txt 2>&1

# 3. Git 操作 (确保使用完整路径)
/usr/bin/git add index.html
/usr/bin/git commit -m "daily auto update" >> /Users/songqin.bian/Desktop/stock/update_log.txt 2>&1
/usr/bin/git push >> /Users/songqin.bian/Desktop/stock/update_log.txt 2>&1

echo "Finished update at $(date)" >> /Users/songqin.bian/Desktop/stock/update_log.txt
echo "-----------------------------------" >> /Users/songqin.bian/Desktop/stock/update_log.txt
