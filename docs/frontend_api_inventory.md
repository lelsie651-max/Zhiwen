# 前端 API 盘点与最小安全接入层

## 范围

- 本文盘点当前后端对 Web 前端可见的路由与本轮新增的最小用户态 JSON API。
- 本轮不创建前端代码，不改百炼工具兼容行为，不新增 migration。
- 结论优先级：
  1. 已可直接供前端调用的 JSON API
  2. 仅适合服务端渲染页面的 HTML 路由
  3. 底层 service 存在但尚未安全暴露的 blocker

## 核心结论

1. 变更前，真正的 JSON API 只有 `/health`、`/ready` 和 3 个百炼只读工具接口；项目、文档、处理流程都还是 HTML 页面。
2. 本轮新增了基于用户会话的最小安全接入层：
   - `GET /api/v1/me`
   - `GET /api/v1/projects/{project_id}/review-items`
   - `GET /api/v1/projects/{project_id}/review-items/{fact_id}`
   - `POST /api/v1/projects/{project_id}/review-items/{fact_id}/decisions`
   - `GET /api/v1/projects/{project_id}/versions/{project_version_id}/records/{subject_key}`
3. 新接口全部使用现有用户 Session 鉴权，不复用 `BAILIAN_TOOL_TOKEN`。
4. 新接口全部要求显式 `project_id`，并先校验 Project 成员关系；不做 implicit latest / max(version_no)。
5. 仍存在的主要 blocker：
   - 项目/文档/Revision/Processing Job 仍只有 HTML 路由，没有前端 JSON API。
   - Dynamic Schema 当前定义没有现成安全只读 service/router。
   - Knowledge View service 已存在，但本轮未新增用户态 router。
   - ProjectVersion 列表/详情缺少现成公共只读 service；本轮只开放精确 `version record`。
6. 当前 OpenAPI 的错误模型仍以 FastAPI 默认 `HTTPValidationError` + `{"detail": "<fixed-code>"}` 为主，没有统一的共享错误 DTO。

## 鉴权与授权

- 用户态 JSON API：
  - 鉴权：`SessionMiddleware` + `request.session["current_user_id"]`
  - 写接口额外要求 `X-CSRF-Token`
  - Project 授权：先用 `project_id + user_id` 查 `project_members`
- 百炼工具 API：
  - 鉴权：`Authorization: Bearer <BAILIAN_TOOL_TOKEN>`
  - 不可给浏览器直接使用

## CORS

- 新增 `FRONTEND_ORIGINS` 配置，逗号分隔白名单。
- 仅当配置非空时启用 `CORSMiddleware`。
- `allow_credentials=True` 时只允许显式白名单，禁止 `*`。
- 当前允许：
  - methods: `GET`, `POST`, `OPTIONS`
  - headers: `Content-Type`, `X-CSRF-Token`

## 1. 项目与文档

| Method | Path | operation_id | 请求/响应 DTO | service | 鉴权方式 | Project 权限 | R/W | 对应前端页面 | 可直接使用 | 缺口或风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET` | `/` | `home__get` | 无 / `HTMLResponse` | 模板渲染 | 无 | 否 | R | 首页 | 否 | 纯静态 HTML，不是 JSON API |
| `GET` | `/setup` | `setup_page_setup_get` | 无 / `HTMLResponse` | `identity_service` 间接使用 | Session 可选 | 否 | R | 首次身份设置 | 否 | 服务端表单页 |
| `POST` | `/setup` | `setup_submit_setup_post` | `Form` / `RedirectResponse` | `identity_service.register_user` | Session + CSRF | 否 | W | 首次身份设置 | 否 | 表单提交，返回跳转，不适合 SPA |
| `GET` | `/app` | `dashboard_app_get` | 无 / `HTMLResponse` | `project_service.list_projects_for_user` | Session | 否 | R | 工作台 | 否 | 返回 HTML，不是项目 JSON 列表 |
| `POST` | `/logout` | `logout_logout_post` | `Form` / `RedirectResponse` | Session 清理 | Session + CSRF | 否 | W | 退出登录 | 否 | 表单式 |
| `POST` | `/projects` | `create_project_projects_post` | `ProjectCreate`(经表单组装) / `RedirectResponse` | `project_service.create_project_for_owner` | Session + CSRF | 否 | W | 新建项目 | 否 | 没有 JSON 项目创建 API |
| `GET` | `/projects/{slug}` | `project_detail_projects__slug__get` | 路径参数 / `HTMLResponse` | `document_upload_service.get_project_workspace_context` | Session | 是，按 `slug` | R | 项目详情 | 否 | 仅 HTML；前端若用 JSON 需新 router |
| `GET` | `/projects/{slug}/documents/upload` | `document_upload_page_projects__slug__documents_upload_get` | 路径参数 / `HTMLResponse` | `document_upload_service.get_project_upload_context` | Session | 是，按 `slug` | R | 上传页 | 否 | 仅 HTML；上传上下文没有 JSON API |
| `POST` | `/projects/{slug}/documents/upload` | `document_upload_submit_projects__slug__documents_upload_post` | `DocumentUploadSubmit`(表单+文件) / `RedirectResponse` | `document_upload_service.upload_document_for_project` | Session + CSRF | 是，按 `slug` | W | 文件上传 | 否 | 仅 HTML multipart；没有用户态 JSON 上传 API |
| `GET` | `/projects/{slug}/documents/{document_id}/revisions/{revision_id}` | `revision_detail_projects__slug__documents__document_id__revisions__revision_id__get` | 路径参数 / `HTMLResponse` | `document_upload_service.get_revision_detail_for_user` | Session | 是，按 `slug` | R | Revision 详情 | 否 | 可复用底层 service，但当前只有 HTML |
| `POST` | `/projects/{slug}/documents/{document_id}/revisions/{revision_id}/admission` | `revision_admission_submit_projects__slug__documents__document_id__revisions__revision_id__admission_post` | `RevisionAdmissionDecisionInput`(表单组装) / `RedirectResponse` | `revision_admission_service` | Session + CSRF | 是，按 `slug` | W | Revision 准入 | 否 | 仅 HTML；未开放 JSON admission API |
| `POST` | `/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/start` | `revision_processing_start_projects__slug__documents__document_id__revisions__revision_id__processing_start_post` | 表单 / `RedirectResponse` | `processing_job_service.enqueue_revision_extraction_job` | Session + CSRF | 是，按 `slug` | W | 解析启动 | 否 | 仅 HTML；没有 JSON Job 启动/状态 API |
| `POST` | `/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/retry` | `revision_processing_retry_projects__slug__documents__document_id__revisions__revision_id__processing_retry_post` | 表单 / `RedirectResponse` | `processing_job_service.retry_failed_revision_extraction` | Session + CSRF | 是，按 `slug` | W | 解析重试 | 否 | 仅 HTML |
| `POST` | `/projects/{slug}/documents/{document_id}/revisions/{revision_id}/processing/recover` | `revision_processing_recover_projects__slug__documents__document_id__revisions__revision_id__processing_recover_post` | 表单 / `RedirectResponse` | `processing_job_service.recover_stale_revision_extraction` | Session + CSRF | 是，按 `slug` | W | 解析恢复 | 否 | 仅 HTML |

**结论**：项目/文档主流程的底层 service 已存在，但当前 Web 前端若想用 JSON，需要新增独立 router。该部分本轮未补齐，属于后续 blocker。

## 2. 一致性审阅

| Method | Path | operation_id | 请求/响应 DTO | service | 鉴权方式 | Project 权限 | R/W | 对应前端页面 | 可直接使用 | 缺口或风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET` | `/api/v1/integrations/bailian/projects/{project_id}/review-items` | `bailianListReviewItems` | Query / `BailianReviewItemsResponse` | `bailian_review_tools.list_review_items` | Bailian Bearer | 仅显式 `project_id` 来源认证，不是用户权限 | R | 评审列表 | 否 | 共享 Token，不能给浏览器 |
| `GET` | `/api/v1/integrations/bailian/projects/{project_id}/review-items/{fact_id}` | `bailianGetReviewItemDetail` | Query / `BailianReviewItemDetailResponse` | `bailian_review_tools.get_review_item_detail` | Bailian Bearer | 同上 | R | 事实详情 | 否 | 共享 Token，不能给浏览器 |
| `GET` | `/api/v1/projects/{project_id}/review-items` | `frontendListReviewItems` | Query / `FrontendReviewItemsResponse` | `frontend_api.list_review_items` -> `review_query.list_review_items` | Session | 是，先校验 `project_members` | R | 评审列表 | 是 | 复用了平台中立只读查询与认证，保留显式 source identity |
| `GET` | `/api/v1/projects/{project_id}/review-items/{fact_id}` | `frontendGetReviewItemDetail` | Query / `FrontendReviewItemDetailResponse` | `frontend_api.get_review_item_detail` -> `review_query.get_review_item_detail` | Session | 是 | R | 事实详情 | 是 | 完整保留 `value_groups` / `FactValue` / `Evidence` / 当前人工状态 |
| `POST` | `/api/v1/projects/{project_id}/review-items/{fact_id}/decisions` | `frontendAppendReviewDecision` | `FrontendReviewDecisionWriteRequest` / `FrontendReviewDecisionWriteResponse` | `frontend_api.write_review_decision` -> `dynamic_schema_review_projection` + `consistency_review.append_consistency_review_decision` + `review_query.get_review_item_detail` | Session + `X-CSRF-Token` | 是 | W | 人工 Decision 提交 | 是 | Router 不复制 Decision 语义；先校验 fact/assessment 绑定，再复用 11C 写路径 |

**结论**：审阅最小闭环已具备浏览器可用的安全入口。

## 3. 知识与版本

| Method | Path | operation_id | 请求/响应 DTO | service | 鉴权方式 | Project 权限 | R/W | 对应前端页面 | 可直接使用 | 缺口或风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET` | `/api/v1/integrations/bailian/projects/{project_id}/versions/{project_version_id}/records/{subject_key}` | `bailianGetVersionRecord` | 路径参数 / `BailianVersionRecordResponse` | `bailian_review_tools.get_version_record` | Bailian Bearer | 非用户权限 | R | 指定 subject record | 否 | 共享 Token，不能给浏览器 |
| `GET` | `/api/v1/projects/{project_id}/versions/{project_version_id}/records/{subject_key}` | `frontendGetVersionRecord` | 路径参数 / `FrontendVersionRecordResponse` | `frontend_api.get_version_record` -> `review_query.get_version_record` | Session | 是 | R | 指定 subject record | 是 | 精确 `(project_id, project_version_id, subject_key)`；不支持 implicit latest |
| - | Dynamic Schema 当前定义 | - | 现有 `DynamicSchemaRead / DynamicSchemaVersionRead / DynamicSchemaFieldRead` 仅 DTO，无公共只读 service/router | - | - | - | - | Schema 配置页 | 否 | blocker：当前只有写 service，缺少安全只读入口 |
| - | Knowledge View | - | 现有 `DynamicSchemaKnowledgeView` dataclass + `build_dynamic_schema_knowledge_view(...)` service | `dynamic_schema_knowledge_view.build_dynamic_schema_knowledge_view` | - | 可通过新 router 校验 | R | 知识视图 | 否 | blocker：底层 service 已有，本轮未新增用户态 router |
| - | ProjectVersion 列表/详情 | - | 现有 `ProjectVersionSnapshot` 仅支持精确单条读取 | `project_version.get_project_version_snapshot` | - | 可通过新 router 校验 | R | 版本列表/详情 | 否 | blocker：缺少公共 list/detail service，不能临时伪造 |

**结论**：本轮只开放了最安全、最明确的 `Version Record` 精确读取；其余知识/版本页面仍有 blocker。

## 4. 系统体验

| Method | Path | operation_id | 请求/响应 DTO | service | 鉴权方式 | Project 权限 | R/W | 对应前端页面 | 可直接使用 | 缺口或风险 |
|---|---|---|---|---|---|---|---|---|---|---|
| `GET` | `/health` | `health_check_health_get` | 无 / JSON | 内联 | 无 | 否 | R | 健康检查 | 是 | 只返回基础状态 |
| `GET` | `/ready` | `ready_check_ready_get` | 无 / JSON | `health_service.is_database_ready` | 无 | 否 | R | 环境就绪 | 是 | 只返回 readiness，不含业务上下文 |
| `GET` | `/api/v1/me` | `getCurrentUser` | 无 / `FrontendCurrentUserResponse` | `require_current_user` + `ensure_csrf_token` | Session | 否 | R | 当前用户 / 前端初始化 | 是 | 返回用户信息和 CSRF token，供写接口使用 |

## 当前错误模型与 OpenAPI

- 旧 HTML 路由：
  - 多数 `operation_id` 仍为 FastAPI 自动生成值。
  - 错误主要是：
    - `HTTPValidationError`
    - `HTTPException(detail="<message>")`
    - HTML 表单内联错误
- 百炼工具路由：
  - 3 个 `operation_id` 已显式固定：
    - `bailianListReviewItems`
    - `bailianGetReviewItemDetail`
    - `bailianGetVersionRecord`
  - 鉴权失败固定：
    - `401 bailian_tool_unauthorized`
    - `503 bailian_tool_unconfigured`
- 新前端 JSON 路由：
  - 新增显式 `operation_id`：
    - `getCurrentUser`
    - `frontendListReviewItems`
    - `frontendGetReviewItemDetail`
    - `frontendAppendReviewDecision`
    - `frontendGetVersionRecord`
  - 继续使用固定 `detail` 字符串，不回显异常、SQL、栈或密钥。
  - 对 invariant/source drift 类错误统一脱敏为：
    - `frontend_api_source_invalid`
    - `frontend_review_decision_source_invalid`

## 建议的下一批接口

按 MVP 缺口优先级，后续建议顺序：

1. 项目列表/详情 JSON API
2. 文档上传与 Revision/Processing 状态 JSON API
3. Knowledge View 用户态 router
4. Dynamic Schema 当前定义只读 service + router
5. ProjectVersion 列表/详情公共只读 service + router
