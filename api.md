# API 文档

此文档列出了项目中的所有 API 接口。

## 基本信息

- Base URL: `https://api.example.com/`
- 所有请求都需要在 `Headers` 中提供 `Authorization` 字段。

## 目录

- [获取用户信息](#获取用户信息)
- [更新用户信息](#更新用户信息)
- [删除用户](#删除用户)

## 获取用户信息

- **GET** `/api/user`

获取指定用户的详细信息。

### 请求参数

| 参数 | 类型   | 必填 | 描述          |
| ---- | ------ | ---- | ------------- |
| `id` | string | 是   | 用户的唯一 ID |

### 示例请求

```bash
curl -X GET "https://api.example.com/api/user?id=123" -H "Authorization: Bearer your-token"
```

