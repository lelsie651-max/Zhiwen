# 织文恢复开发 Runbook

## 1. 目的

本 Runbook 面向未来新成员或新对话上下文丢失后的恢复场景，目标是：

- 让接手者在不依赖历史聊天的情况下恢复开发
- 快速建立本地数据库、迁移、demo seed、API 与测试环境
- 明确第一批应该读什么、跑什么、先做什么

## 2. 环境准备

### 基本要求

- Python 3.13
- PostgreSQL 16
- Windows / WSL / macOS / Linux 均可，只要能提供 PostgreSQL
- 建议使用虚拟环境

### 依赖准备

在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果本机 PowerShell 执行策略限制脚本，可改用普通激活方式或 CMD。

## 3. 数据库创建

默认本地数据库配置参考 `.env.example`：

```env
DATABASE_URL=postgresql+asyncpg://zhiwen:zhiwen@localhost:5432/zhiwen
```

建议至少准备两个数据库：

1. 日常开发库：`zhiwen`
2. 百炼 demo 联调库：`zhiwen_bailian_demo`

## 4. migration

仓库当前 head 为：

```text
202608010600
```

执行迁移：

```powershell
alembic upgrade head
```

校验 head：

```powershell
alembic heads
```

## 5. demo seed

当前 demo seed CLI：

- 文件：`scripts/seed_bailian_demo.py`
- 安全阀：必须显式传 `--confirm-local-demo`

执行：

```powershell
python scripts/seed_bailian_demo.py --confirm-local-demo
```

当前 demo seed 的定位是：

- `single_revision_cross_batch_plugin_demo`

CLI 输出中会返回一组当前数据库真实 ID，可作为联调基准。

## 6. 启动 FastAPI

本地启动：

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

当前入口会自动：

- 装载 SessionMiddleware
- 在配置了 `FRONTEND_ORIGINS` 时装载 CORS
- 注册 HTML 路由、用户态 JSON API 和百炼工具 API

## 7. 跑定向和全量测试

### 全量测试

```powershell
python -m pytest -ra
```

当前已验证结果：

```text
1727 passed
```

### 推荐定向测试

```powershell
python -m pytest tests/test_frontend_api_router.py -q
python -m pytest tests/test_bailian_tools_router.py -q
python -m pytest tests/test_bailian_review_tools.py -q
python -m pytest tests/test_consistency_review.py -q
python -m pytest tests/test_project_version_service.py -q
python -m pytest tests/test_bailian_demo_seed.py -q
```

## 8. 验证 OpenAPI

生成并校验 OpenAPI：

```powershell
python -c "from app.main import app; app.openapi(); print('openapi-ok')"
```

浏览器查看：

```text
http://127.0.0.1:8000/openapi.json
```

当前重点检查项：

- 百炼工具 Bearer 安全定义 `BailianToolBearer`
- 前端用户态 API 的 operation_id
- HTML 表单路由与 JSON API 的区分

## 9. 验证用户态前端 API

### 9.1 当前用户

```text
GET /api/v1/me
```

要求：

- 已完成 session 登录
- 返回用户信息与 `csrf_token`

### 9.2 审阅列表

```text
GET /api/v1/projects/{project_id}/review-items
```

### 9.3 审阅详情

```text
GET /api/v1/projects/{project_id}/review-items/{fact_id}
```

### 9.4 提交人工决议

```text
POST /api/v1/projects/{project_id}/review-items/{fact_id}/decisions
Header: X-CSRF-Token
```

### 9.5 读取指定版本记录

```text
GET /api/v1/projects/{project_id}/versions/{project_version_id}/records/{subject_key}
```

注意：

- 所有用户态 API 都依赖 session 用户身份
- 所有项目内 API 都要求 Project membership
- 不支持 implicit latest

## 10. 常见错误排查

### 10.1 `authentication_required`

说明没有有效 session。  
先访问 `/setup` 完成用户创建，或用测试方式注入 session。

### 10.2 `project_access_denied`

说明当前用户不是该 `project_id` 的成员，或拿错项目 UUID。

### 10.3 `frontend_api_source_invalid`

说明只读投影认证、manifest 或 source identity 出现漂移；不要改成“放宽校验”，应先检查调用方是否传了错误的 project/schema/version/orchestration/application。

### 10.4 `frontend_review_decision_source_invalid`

说明 Decision 写入链上的来源认证失败，优先检查：

- `assessment_id`
- `consistency_check_application_id`
- 当前 fact 是否真在目标 review projection 中

### 10.5 `bailian_tool_unconfigured`

说明未配置 `BAILIAN_TOOL_TOKEN`。

### 10.6 `bailian_tool_unauthorized`

说明百炼接口没有带正确的 `Authorization: Bearer`。

### 10.7 CORS 不生效

先检查：

- `FRONTEND_ORIGINS` 是否配置
- 是否是显式 origin 白名单
- 是否错误地写成 `*`

## 11. 恢复开发时的第一批任务

建议接手后先做这批任务，而不是立刻扩算法：

1. 把项目/文档/revision/processing 主流程补成用户态 JSON API
2. 补 Knowledge View 用户态 router
3. 补 Dynamic Schema 当前定义读取 API
4. 补 ProjectVersion 列表/详情 API
5. 统一错误 DTO 和分页契约
6. 再决定是否继续做真正的前端工程

## 12. 新 Agent 接手时应先读取的文件顺序

建议严格按这个顺序建立上下文：

1. `docs/ZHIWEN_CURRENT_STATE.md`
2. `docs/ZHIWEN_PRODUCT_MANUAL.md`
3. `docs/ZHIWEN_TECHNICAL_HANDBOOK.md`
4. `docs/ZHIWEN_ARCHITECTURE_DECISIONS.md`
5. `docs/frontend_api_inventory.md`
6. `app/main.py`
7. `app/core/config.py`
8. `app/routers/frontend_api.py`
9. `app/routers/bailian_tools.py`
10. `app/services/frontend_api.py`
11. `app/services/consistency_review.py`
12. `app/services/dynamic_schema_review_projection.py`
13. `app/services/dynamic_schema_knowledge_view.py`
14. `app/services/project_version.py`
15. `app/services/bailian_demo_seed.py`
16. `tests/test_frontend_api_router.py`
17. `tests/test_bailian_tools_router.py`
18. `tests/test_bailian_review_tools.py`
19. `tests/test_consistency_review.py`
20. `tests/test_project_version_service.py`
