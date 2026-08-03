# 织文当前状态

## 1. 仓库状态

- 当前分支：`main`
- 当前基线 commit：`1d21911fc1f00599d0429c5d950b8dedff49a7bc`
- 本文档生成时工作树状态：
  - 基线代码来自上述 commit
  - 当前存在未提交的文档变更，用于补齐长期冻结包

## 2. Alembic 与数据库

- 当前 `alembic heads`：`202608010600`
- 最新迁移：`alembic/versions/202608010600_project_versions.py`
- 当前仓库不需要新增 migration 才能启动与运行现有能力

## 3. 测试状态

- 当前全量测试：`1727 passed`
- 执行命令：`python -m pytest -ra`
- 当前警告重点：
  - `starlette.testclient` 关于 `httpx` 的弃用提醒
  - Alembic `path_separator` 配置弃用提醒

## 4. 已完成阶段

按当前代码可确认，已经完成的主要阶段包括：

1. 用户、项目、文档上传与 revision 基础模型
2. 文档解析、block 化与 evidence 保存
3. Fact / FactValue / Evidence 账本
4. 抽取 orchestration 与持久化
5. duplicate grouping
6. consistency check application / assessment
7. consistency review 决议不可变链
8. Dynamic Schema draft/proposal/activate
9. review projection 与 knowledge view
10. ProjectVersion 冻结版本
11. 百炼 3 个只读工具接口
12. 面向浏览器前端的最小用户态 JSON API

## 5. 当前前端用户态 API

当前真正可供浏览器前端调用的 JSON API 为：

- `GET /api/v1/me`
- `GET /api/v1/projects/{project_id}/review-items`
- `GET /api/v1/projects/{project_id}/review-items/{fact_id}`
- `POST /api/v1/projects/{project_id}/review-items/{fact_id}/decisions`
- `GET /api/v1/projects/{project_id}/versions/{project_version_id}/records/{subject_key}`

这些接口特点：

- 使用用户 Session 鉴权
- 写接口要求 `X-CSRF-Token`
- 先校验 Project 成员关系
- 复用既有 review / decision / project version 认证服务

## 6. 尚未补齐的 JSON API

当前仍缺的主要用户态 JSON API：

- 项目列表 / 项目详情
- 文档上传 JSON API
- Revision / Block / 解析状态 JSON API
- Processing Job 状态 JSON API
- Dynamic Schema 当前定义读取 API
- Knowledge View 用户态读取 API
- ProjectVersion 列表 / 详情 API

这些能力里，有些底层 service 已存在但尚未暴露，有些则还缺公共只读 service。

## 7. 已知 blocker

### 7.1 前端主流程仍偏 HTML

`/projects/...` 相关路由目前仍是 HTML/表单工作台风格，前端若切到 SPA，需要继续补 JSON 契约。

### 7.2 Dynamic Schema 当前定义读取缺口

当前 `app/services/dynamic_schema.py` 主要是写路径（create/activate），没有现成用户态只读入口。

### 7.3 Knowledge View 尚未有用户态 router

`build_dynamic_schema_knowledge_view(...)` 已存在，但还没有浏览器侧正式 JSON API。

### 7.4 ProjectVersion 仍缺列表/详情公共 service

当前只有精确单版本读取和指定 record 读取，无法安全地临时拼出“latest”或列表接口。

## 8. 百炼集成已证明能力

按当前代码与既有集成回归，可确认已经证明的能力是：

- 通过 `Authorization: Bearer` 的百炼只读工具鉴权
- 审阅事项列表读取
- 单条审阅事项详情读取
- 指定 `project_version_id + subject_key` 的版本记录读取
- 只读工具输出包含来源 manifest 与 payload hash

## 9. 百炼稳定性与延迟问题

百炼集成当前是“已联调、可替换、非核心业务地基”的状态。  
从此前联调记录看，主要问题不在织文内部算法，而更多在外部运行时层：

- 工具名/工具实例在插件侧可能被运行时重命名
- 多工具插件的一致调用稳定性不足
- 有时 `tool_call` 会退化成文本输出而非真实执行
- 首轮调用真实执行率并不稳定
- T2/T3 场景存在高延迟

因此当前结论应是：

- 百炼只保留为可替换集成能力
- 不应把百炼 Agent 当成织文核心主链路

## 10. 当前 demo 数据库边界

当前 demo seed 的真实定位是：

- `single_revision_cross_batch_plugin_demo`

它已经证明：

- 单个 revision 内跨 batch 的事实冲突
- 待审 / 已解决 / 观察值三种状态
- 百炼 3 个只读工具的联调场景
- 精确 subject record 读取

它尚未证明：

- 跨独立 document 的一致性候选
- 更复杂的多 orchestration 发布链

## 11. 下一阶段建议

恢复开发后，建议按这个顺序推进：

1. 把项目/文档/revision/processing 主流程补成用户态 JSON API
2. 增加 Knowledge View 用户态 router
3. 增加 Dynamic Schema 当前定义只读 API
4. 增加 ProjectVersion 列表 / 详情 API
5. 根据前端真实页面需要，整理统一错误 DTO 与分页契约
6. 继续把 HTML 工作台与正式 JSON API 的职责边界拉清

## 12. 相关文档

- [产品手册](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/ZHIWEN_PRODUCT_MANUAL.md)
- [技术手册](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/ZHIWEN_TECHNICAL_HANDBOOK.md)
- [恢复 Runbook](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/ZHIWEN_RESTART_RUNBOOK.md)
- [架构决策](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/ZHIWEN_ARCHITECTURE_DECISIONS.md)
- [百炼复盘](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/ZHIWEN_BAILIAN_POSTMORTEM.md)
- [前端 API 盘点](file:///c:/Users/1/Documents/GitHub/Zhiwen/docs/frontend_api_inventory.md)
