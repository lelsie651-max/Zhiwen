# 织文架构决策记录

## ADR-001：为什么 Evidence 必须不可变

### 决策

Evidence 一旦进入事实链路，就应视为来源记录，不应被随意原地改写。

### 原因

- Evidence 是 FactValue 成立与冲突的依据
- 后续 review、knowledge view、version record 都依赖它
- 一旦允许覆盖，就无法回答“当时为什么会做这个判断”

### 结果

- 系统更偏向追加新 extraction/run/evidence，而不是覆盖旧记录
- 认证与回放能力得到保留

## ADR-002：为什么 Fact 不应该成为“当前值容器”

### 决策

Fact 负责表达事实身份，FactValue 负责表达候选值与版本。

### 原因

- 同一事实可能长期存在多个候选值
- 决议可能保留多个值，也可能暂缓决策
- 如果把 Fact 简化成“当前唯一值”，系统就失去冲突表达能力

### 结果

- 上层读取必须基于投影与决议，而不是偷懒直接读单一字段

## ADR-003：为什么 Decision 采用追加链，而不是覆盖写

### 决策

人工决议通过 `ConsistencyReviewDecision` 追加到不可变链中，使用 `supersedes_decision_id` 线性推进。

### 原因

- 决议本身也是审计对象
- 需要支持幂等重试与冲突恢复
- 需要知道“现在的当前决议”是从哪条旧决议演化而来

### 结果

- router 不能直接“改当前状态”
- service 需要维护 manifest hash、decision_no 与链一致性

## ADR-004：为什么禁止 implicit latest

### 决策

正式接口不得把“latest”“max(version_no)”当成默认业务契约。

### 原因

- latest 在并发和版本切换期间不稳定
- 外部系统难以证明自己读到的是哪一版
- 这会削弱 ProjectVersion 的存在意义

### 结果

- 用户态版本记录接口要求显式 `project_version_id + subject_key`
- 百炼工具接口同样要求显式来源身份

## ADR-005：为什么 ProjectVersion 必须做来源认证

### 决策

`ProjectVersion` 必须绑定并认证：

- project
- schema / schema_version
- orchestration
- consistency_check_application
- knowledge view manifest
- snapshot hash

### 原因

- 否则 version 只是“某次查询结果的缓存”
- 一旦来源漂移，就无法证明 record 仍然可信

### 结果

- 版本读取时要做严格 snapshot 认证
- 版本创建不能绕过来源链直接写一份 JSON

## ADR-006：为什么 Agent 不是核心业务地基

### 决策

Agent/LLM 是编排或抽取执行层，不是织文可信业务状态的基座。

### 原因

- Agent 调用本身有外部运行时不稳定性
- 工具调用可能失败、超时、被重命名或退化成文本输出
- 业务可信状态必须回到数据库账本、投影与认证链上

### 结果

- 即使没有 Agent，系统依然应该能恢复、审阅、发布和对外只读
- 百炼只保留为可替换集成，不上升为产品根基

## ADR-007：为什么浏览器用户接口与百炼工具接口必须分离

### 决策

浏览器用户接口和百炼工具接口复用底层 service，但不共用鉴权与路由边界。

### 原因

- 浏览器侧依赖用户身份、Project membership 和 CSRF
- 百炼工具侧是共享 Bearer Token 的服务级只读访问
- 两者的风险模型、错误边界、调用场景都不同

### 结果

- `app/routers/frontend_api.py` 与 `app/routers/bailian_tools.py` 分开维护
- 前者走 session，后者走 `Authorization: Bearer`

## ADR-008：为什么系统偏向 fail-closed

### 决策

对 source drift、hash 不匹配、类型不合法、来源不精确的情况，优先拒绝而不是兜底猜测。

### 原因

- 织文的目标是高可信事实状态，而不是“尽量返回点东西”
- 在知识发布、只读工具和版本读取上，错误成功比明确失败更危险

### 结果

- 代码中存在大量 `StateError` / `InvariantError`
- 外层接口通常把 invariant 类错误映射成脱敏固定码
