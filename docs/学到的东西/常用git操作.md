**忽略已经上传到git仓库的某个文件/文件夹**
1. 在.gitignore文件中添加路径
2. 将文件从git索引中移除（但是保留在本地文件夹中
	`git rm -r --cached logs/`
3. 提交更改 `git commit -m ""`
4. 推送更改 `git push`