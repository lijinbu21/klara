[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

    Write-Output "OK, push to GitHub"
} else {
    Write-Output "Nothing new need to be pushed"
}
