# 织文技术手册

## 1. 总体架构的文字版

织文当前是一个模块化单体后端，核心结构如下：

1. `app/models/`
   - SQLAlchemy 数据模型，定义账本、不变量、外键和约束。
2. `app/repositories/`
   - 面向数据库的查询与持久化访问层。
3. `app/services/`
   - 业务规则、状态推进、哈希认证、投影构造、幂等与事务边界。
4. `app/routers/`
   - FastAPI 路由层，当前同时承载 HTML 页面、用户态 JSON API 和百炼只读工具 API。
5. `app/schemas/`
   - Pydantic DTO 与 dataclass 投影对象。
6. `app/agents/`
   - 抽取与一致性检查用的 LLM 交互层，但不是系统可信状态的唯一来源。
7. `alembic/`
   - 数据库迁移链。
8. `tests/`
   - 单元、服务、集成与 router 回归测试。

## 2. 目录与模块职责

### 2.1 入口与配置

- `app/main.py`
  - 创建 FastAPI app
  - 装载 `SessionMiddleware`
  - 按配置装载 `CORSMiddleware`
  - 注册 `web / health / app / projects / frontend_api / bailian_tools`
- `app/core/config.py`
  - 统一读取 `.env`
  - 提供 `database_url_sync`、`is_production`、`frontend_origins`
- `app/core/database.py`
  - 创建 async engine / session factory

### 2.2 用户与项目

- `app/services/identity.py`
  - 用户注册与读取
- `app/services/project.py`
  - 项目创建、项目列表
- `app/repositories/project.py`
  - project / membership 查询

### 2.3 文档导入与处理

- `app/services/document_upload.py`
  - 上传上下文、revision 详情、project 权限校验
- `app/services/file_ingestion.py`
  - 文件落盘与上传处理
- `app/services/revision_admission.py`
  - 文档 revision 准入
- `app/services/processing_job.py`
  - 解析与抽取任务入队、重试、恢复
- `app/services/document_content.py`
  - 解析结果落库，形成 `ExtractionRun / DocumentBlock / SourceEvidence`

### 2.4 抽取与事实层

- `app/services/fact_extraction_orchestration.py`
  - 抽取 orchestration
- `app/services/fact_extraction_persistence.py`
  - 抽取结果持久化
- `app/services/fact.py`
  - Fact / FactValue / Evidence 写入与更新
- `app/services/effective_fact_value.py`
  - 有效事实值推导

### 2.5 一致性检查与审阅

- `app/services/fact_value_duplicate_grouping.py`
  - 候选值分组与确定性哈希
- `app/services/consistency_check*.py`
  - consistency application / assessment / workflow / persistence
- `app/services/consistency_review.py`
  - 人工决议不可变链写入

### 2.6 Schema、投影与知识发布

- `app/services/dynamic_schema.py`
  - schema draft / proposal / activate
- `app/services/dynamic_schema_ufl_projection.py`
  - 从事实账本构造 schema 对齐的 UFL 投影
- `app/services/dynamic_schema_review_projection.py`
  - 在 UFL 投影上叠加审阅状态
- `app/services/dynamic_schema_knowledge_view.py`
  - 构造知识视图
- `app/services/project_version.py`
  - 冻结版本、认证 snapshot、读取精确版本

### 2.7 外部接口层

- `app/routers/projects.py`
  - 当前主要是 HTML 工作台
- `app/routers/frontend_api.py`
  - 用户态 JSON API
- `app/routers/bailian_tools.py`
  - 百炼 3 个只读工具
- `app/services/bailian_review_tools.py`
  - 复用 review/detail/version record 只读服务与认证逻辑

## 3. 核心数据模型及关系

### 3.1 身份与协作

- `User`
- `Project`
- `ProjectMember`

`ProjectMember` 决定项目内角色；当前重要权限角色是 `owner` 与 `editor`。

### 3.2 文档与解析

- `Document`
- `DocumentRevision`
- `ExtractionRun`
- `DocumentBlock`
- `SourceEvidence`
- `ProcessingJob`

这层负责从原始文件走到结构化 block 与 evidence。

### 3.3 事实账本

- `Fact`
- `FactValue`
- `FactEvidenceLink`
- `Entity`

这里表达“事实身份”“候选值版本”“值与证据的绑定”。

### 3.4 一致性与审阅

- `FactValueConsistencyCandidate*`
- `ConsistencyCheckApplication`
- `ConsistencyAssessmentLedger`
- `ConsistencyReviewDecision`
- `ConsistencyReviewDecisionSelection`

这里表达冲突分组、检查结果和人工决策链。

### 3.5 Schema 与发布

- `DynamicSchema`
- `DynamicSchemaVersion`
- `DynamicSchemaField`
- `ProjectVersion`

这里表达知识界面定义与已冻结发布版本。

## 4. 从导入到版本发布的完整数据流

1. 用户创建项目并上传文档 revision。
2. revision 通过 admission 后进入 processing job。
3. 文档解析生成 `ExtractionRun / DocumentBlock / SourceEvidence`。
4. 抽取结果写入 `Fact / FactValue / FactEvidenceLink`。
5. duplicate grouping 生成候选组。
6. consistency check 生成 application 与 assessment ledger。
7. review projection 把事实、候选、审阅状态、人工决策组织成可读结构。
8. Dynamic Schema 把 review projection 映射为字段化界面。
9. knowledge view 在 schema 基础上构造对外知识记录。
10. `ProjectVersion` 把这一时刻的知识视图与来源清单冻结成认证版本。

## 5. Manifest 与哈希认证

当前系统广泛依赖确定性哈希：

- `identity_hash`
- `value_hash`
- `decision_manifest_hash`
- `reviewed_projection_manifest_hash`
- `knowledge_view_manifest_hash`
- `snapshot_json_hash`
- `version_manifest_hash`
- 百炼工具 `payload_hash`

设计目的：

- 认证来源链
- 防止“看起来像同一个结果，其实底层来源已漂移”
- 支持幂等写入与只读接口结果认证

## 6. fail-closed 边界

织文的关键服务倾向于 fail-closed：

- UUID / bool / int / enum 类型不接受模糊输入
- manifest/hash 不匹配直接报错
- project/version/source identity 不匹配直接拒绝
- 读取不到精确来源时不做“猜测式恢复”
- 发现 projection 或 immutable ledger 漂移时返回 invariant error

## 7. 用户认证与 Project 授权

### 用户认证

- 浏览器用户侧使用 `SessionMiddleware`
- session 中保存 `current_user_id`
- `app/dependencies/auth.py` 提供：
  - `get_optional_current_user`
  - `require_current_user`
  - `verify_api_csrf_token`

### Project 授权

- HTML 路由大多通过 `slug` 找 membership
- 新用户态 JSON API 通过 `project_id + user_id` 查 membership
- 无 membership 直接 `403 project_access_denied`

### 百炼接口鉴权

- 百炼接口不使用用户 session
- 只接受 `Authorization: Bearer <BAILIAN_TOOL_TOKEN>`

## 8. Decision 不可变链

`app/services/consistency_review.py` 的关键职责不是“设置当前决议”，而是：

1. 认证 authoritative application
2. 校验 actor 项目权限
3. 校验 assessment、candidate member、selected values 的一致性
4. 计算 `decision_manifest_hash`
5. 以 `decision_no + supersedes_decision_id` 追加一条新链节点
6. 在冲突约束触发时按 manifest 做幂等恢复

因此外部调用者不应直接修改当前状态，而应总是通过 append path 写入。

## 9. Dynamic Schema 投影

Dynamic Schema 相关能力分为三层：

1. `dynamic_schema_ufl_projection`
   - 把事实账本映射成 schema 对齐的原始投影
2. `dynamic_schema_review_projection`
   - 在 UFL 投影上叠加 consistency 与人工审阅状态
3. `dynamic_schema_knowledge_view`
   - 把 review projection 组织成知识视图

这一层使得“底层事实账本”和“上层知识页面”解耦。

## 10. ProjectVersion 认证

`ProjectVersion` 的读取必须基于显式 `(project_id, project_version_id)`。  
读取时会：

- 找到目标项目和目标版本
- 重建并认证 snapshot
- 校验 snapshot 的哈希、布尔、计数、UUID 和来源字段

版本记录读取进一步要求显式 `subject_key`，不做 implicit latest。

## 11. API 分类

### 11.1 HTML 工作台

- `/setup`
- `/app`
- `/projects/...`

这部分适合当前内置模板页面，不是稳定的前端 JSON 契约。

### 11.2 用户态 JSON API

- `GET /api/v1/me`
- `GET /api/v1/projects/{project_id}/review-items`
- `GET /api/v1/projects/{project_id}/review-items/{fact_id}`
- `POST /api/v1/projects/{project_id}/review-items/{fact_id}/decisions`
- `GET /api/v1/projects/{project_id}/versions/{project_version_id}/records/{subject_key}`

### 11.3 百炼只读工具 API

- `bailianListReviewItems`
- `bailianGetReviewItemDetail`
- `bailianGetVersionRecord`

这些接口与浏览器用户接口逻辑可复用，但鉴权边界必须分离。

## 12. 事务与幂等边界

当前值得重点保护的边界：

- 上传/解析/抽取写入在 service 层内自行 `commit/rollback`
- Decision 写入在 service 层处理幂等与 `IntegrityError` 恢复
- ProjectVersion 创建在锁定 project 后完成版本号递增与 snapshot 认证
- fail-closed 情况下优先回滚，不返回半成品状态

## 13. 禁止破坏的系统不变量

以下规则是恢复开发时不能随手破坏的：

1. Evidence 与 FactValue 的来源链必须可追溯。
2. Fact 身份与当前值是两个层次，不能把它们重新混成一个字段。
3. 人工决议只能追加，不应覆盖旧决议。
4. ProjectVersion 必须绑定显式来源与显式版本。
5. 浏览器用户接口不能复用百炼共享 Token。
6. 不允许依赖 implicit latest / max(version_no) 作为正式业务契约。
7. 认证类错误必须保持脱敏，不回显 SQL、堆栈、密钥或原始敏感 sentinel。
8. 只读工具接口的认证与 payload hash 行为不能被无关改动破坏。
