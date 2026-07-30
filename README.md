# 织文

织文是一个 Web + Python 项目骨架，用于把文档转换为可协作、可追溯、可发布并能检查一致性的动态知识库。

## 环境要求

- Python 3.11+
- Docker 与 Docker Compose
- 本地可用的 PostgreSQL 端口 `5432`

## 安装依赖

```bash
python -m pip install -e ".[dev]"
```

## 启动 PostgreSQL

```bash
docker compose up -d
```

## 配置环境变量

复制示例文件并按本地环境修改：

```bash
copy .env.example .env
```

必须配置的环境变量：

- `APP_NAME`
- `APP_ENV`
- `DEBUG`
- `SECRET_KEY`
- `DATABASE_URL`
- `PORT`

## 执行 Alembic 迁移

当前项目已接入 Alembic 配置，后续创建模型后可直接生成迁移：

```bash
alembic revision --autogenerate -m "init"
alembic upgrade head
```

## 启动 FastAPI

优先使用：

```bash
uvicorn app.main:app --reload
```

## 运行测试

```bash
pytest
```

## 生产部署

### 1. 构建 Docker 镜像

```bash
docker build -t zhiwen .
```

### 2. 本地运行镜像

先准备 `.env`，再运行：

```bash
docker run --rm -p 8000:8000 --env-file .env zhiwen
```

容器启动后会先执行：

```bash
alembic upgrade head
```

然后再启动应用：

```bash
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```

### 3. 创建云端 PostgreSQL

- 在云平台创建独立的 PostgreSQL 实例
- 获取数据库连接串并写入 `DATABASE_URL`
- 生产库必须使用云端托管数据库，不要依赖容器本地文件

### 4. 配置全部环境变量

建议至少配置以下变量：

- `APP_NAME=zhiwen`
- `APP_ENV=production`
- `DEBUG=false`
- `SECRET_KEY=<高强度随机值>`
- `DATABASE_URL=<云端 PostgreSQL 连接串>`
- `PORT=8000`

说明：

- 生产环境下如果 `SECRET_KEY` 仍为默认弱值，应用会在启动时直接报错退出
- `PORT` 由平台注入时优先使用平台值，本地未注入时默认 `8000`

### 5. 部署应用服务

- 将仓库推送到代码托管平台
- 在云平台创建新的应用服务并关联仓库
- 使用仓库中的 `Dockerfile` 构建镜像
- 将上述环境变量写入平台配置
- 确保服务与云端 PostgreSQL 网络互通

### 6. 生成公网域名

- 在云平台为服务分配公网访问域名
- 如需自定义域名，再额外完成 DNS 解析与平台绑定
- 不要在应用代码中硬编码任何公网地址

### 7. 访问 `/health` 和 `/ready` 验收

- `GET /health`：仅检查应用进程是否存活
- `GET /ready`：执行数据库 `SELECT 1` 检查应用是否具备对外提供服务的条件

### Zeabur 部署示例

- 新建一个基于 Git 仓库的服务，并让平台以 Docker 方式构建当前项目
- 为项目单独创建 PostgreSQL 实例，复制连接串到 `DATABASE_URL`
- 在服务环境变量中填写 `APP_NAME`、`APP_ENV`、`DEBUG`、`SECRET_KEY`、`DATABASE_URL`、`PORT`
- 等待构建完成并观察启动日志，确认迁移成功后应用开始监听端口
- 通过平台分配的域名访问 `/health` 与 `/ready` 完成验收

### 存储说明

未来用户上传的原始文件与图片不能长期依赖容器本地磁盘，应接入持久化卷或对象存储；本轮不实现文件存储。

## 项目说明

- `app/main.py` 仅负责装配应用、挂载静态资源与注册路由
- `app/routers/` 提供 HTTP 入口
- `app/services/` 预留业务流程层
- `app/repositories/` 预留数据库读写层
- `app/models/` 放置 SQLAlchemy 模型基类与后续实体
- `app/schemas/` 放置 Pydantic 输入输出结构
- `app/core/` 放置配置、数据库与日志基础设施
- `app/templates/` 与 `app/static/` 承载服务端页面资源
