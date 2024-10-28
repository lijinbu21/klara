[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 设置日志文件路径
$logFile = "D:\ALLINONE\klara-notes\auto_push_log.txt"

# 写入日志的函数
function Write-Log {
    param (
        [string]$message
    )
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "$timestamp - $message" | Out-File -FilePath $logFile -Append
}

# 启动日志记录
Write-Log "Start Excuting your task -- update notes :$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 设置 Git 仓库路径
$repoPath = "D:\ALLINONE\klara-notes"

# 进入仓库目录
Set-Location $repoPath

# 检查是否有更改
$changes = git status --porcelain

if ($changes) {
    # 添加所有更改
    git add .

    # 提交更改，使用当前日期和时间作为提交信息
    $commitMessage = "Auto-update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    git commit -m $commitMessage

    # 推送到远程仓库
    git push origin main

    Write-Log "OK, push to GitHub"
} else {
    Write-Log "Nothing new need to be pushed"
}

Write-Log "Fish your task"