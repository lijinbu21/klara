#!/bin/bash

# 设置 Git 仓库路径（替换为你的仓库路径）
REPO_PATH="/Users/klara/Documents/project/klara-notes"

# 日志文件路径
LOG_FILE="$REPO_PATH/logs/update.log"

# 当前日期和时间
CURRENT_TIME=$(date +'%Y-%m-%d %H:%M:%S')

# 进入仓库目录
cd "$REPO_PATH" || exit

# 检查是否有更改
if git diff-index --quiet HEAD --; then
  echo "$CURRENT_TIME: 没有检测到更改，不需要更新。" >> "$LOG_FILE"
else
  # 添加所有更改
  git add .

  # 提交更改，使用当前日期和时间作为提交信息
  git commit -m "Auto-update: $CURRENT_TIME"

  # 推送到远程仓库
  git push origin main

  echo "$CURRENT_TIME: 更改已推送到 GitHub (from mac)。" >> "$LOG_FILE"
fi

# 记录脚本运行结束的时间
echo "$CURRENT_TIME: 脚本执行结束。" >> "$LOG_FILE"
