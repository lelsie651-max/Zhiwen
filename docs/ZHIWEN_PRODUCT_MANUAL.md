# 织文产品手册

## 1. 产品定位

织文是一个以 Evidence 为先的事实抽取、冲突审阅与知识发布平台。它的目标不是“生成一段看起来合理的总结”，而是把来自文档的结构化事实、事实候选、证据链、人工决议和最终发布版本拆开保存，并允许后续逐步认证与回放。

当前仓库中的产品形态更接近一个“事实账本 + 审阅投影 + 已认证知识快照”的后端内核，配套：

- 服务端渲染的项目/文档工作台
- 面向百炼的 3 个只读工具接口
- 面向浏览器前端的最小用户态 JSON API

## 2. 目标用户

- 需要把长文档、规章、产品资料、企业知识资料转为结构化事实的团队
- 需要对模型抽取结果进行人工确认、留痕和回溯的运营/审核人员
- 需要把“当前可信知识状态”发布为稳定版本的产品或知识工程团队
- 需要外部 Agent/插件只读访问审阅结果或版本记录的集成方

## 3. 解决的问题

织文解决的不是“文本能不能抽出来”，而是下面几类更难的问题：

1. 同一事实会出现多个候选值，且候选之间可能冲突。
2. 候选值必须带上来源证据，不能只留下模型结论。
3. 人工决议不能覆盖历史，否则后续无法审计“为什么现在是这个结果”。
4. 面向外部系统暴露的知识状态，必须可以认证来源、版本和哈希，而不是一份随手拼的 JSON。
5. 同一个项目会不断上传新文档、产生新提取、形成新版本，系统必须能在不破坏旧状态的前提下演进。

## 4. 核心用户旅程

当前代码支持的核心旅程可以概括为：

1. 创建用户并进入工作台。
2. 创建 Project。
3. 上传 Document 与 Revision。
4. 通过解析与抽取流程生成 ExtractionRun、Fact、FactValue 和 Evidence。
5. 对候选值做 duplicate grouping 与 consistency check。
6. 在审阅投影中查看待审事实、详情、候选值和证据。
7. 人工提交 Decision，系统按不可变链追加决议。
8. 以 Dynamic Schema 组织知识视图。
9. 将当前知识视图冻结为 ProjectVersion。
10. 通过只读 API 或外部工具读取审阅结果与版本记录。

## 5. Evidence-first 原则

织文的产品核心是 Evidence-first：

- 事实值必须能追溯到证据块。
- 事实冲突的判断不能脱离证据。
- 人工决议是对候选关系的判断，不是“直接改最终值”。
- 发布版本不是“当前数据库状态快照”，而是基于已认证来源投影构造出来的版本对象。

这套原则决定了：织文更像一条可审计的事实生产线，而不是一个普通 CMS。

## 6. Fact / FactValue / Evidence 语义

### Fact

`Fact` 表示一个事实身份，由以下维度决定：

- `subject_kind`
- `subject_key`
- `predicate_key`
- `scope_key`
- `identity_hash`

它描述“这是哪一条事实”，而不是“当前值是什么”。

### FactValue

`FactValue` 是某个 `Fact` 的一个候选值版本，包含：

- `value_type`
- `value_json`
- `normalized_value_text`
- `value_hash`
- 来源类型与来源 run
- 状态（如 `proposed` / `accepted` / `rejected` / `superseded`）

多个 `FactValue` 可以同时存在，用来表达冲突、重复候选、观察值和人工保留结果。

### Evidence

证据链来自 `DocumentBlock` 与 `SourceEvidence`，再通过 `FactEvidenceLink` 关联到 `FactValue`。在产品语义上，Evidence 不是说明性备注，而是事实值成立或冲突的依据。

## 7. 一致性审阅与人工决议

系统会对候选值做分组和一致性评估，形成：

- consistency candidate
- assessment
- review projection

人工决议支持当前已实现的 4 类契约：

- `select_one`
- `keep_multiple`
- `confirm_compatible`
- `defer`

这些决议不会覆盖旧决议，而是以 `ConsistencyReviewDecision` 追加到不可变链上，并通过 supersede 关系线性推进。

## 8. Dynamic Schema

Dynamic Schema 是织文把“事实账本”投影成“知识界面”的组织层。它定义：

- 字段 key / label / group
- 对应的 `predicate_key` / `scope_key`
- 值类型与基数
- 展示顺序和布局配置

当前代码已实现：

- Human draft 创建
- AI proposal 创建
- 激活某个 schema version
- 基于 UFL 与 review projection 构造 knowledge view / review view

## 9. ProjectVersion

`ProjectVersion` 是织文的发布冻结层。它不是临时查询结果，而是：

- 绑定 project/schema/schema_version/orchestration/application
- 保存多层 manifest hash
- 保存 snapshot json 与 snapshot hash
- 记录 record / section / field 统计
- 维护项目当前版本指针

它的价值在于：以后任何外部读取都可以基于明确版本和 subject record 进行，不需要猜“最新状态”。

## 10. 典型使用场景

### 企业知识库整理

把企业产品资料、操作规范、FAQ 等整理为结构化 subject record，并发布为可查询版本。

### 政策/规章梳理

从规章文档中抽取主体、时间、范围、条件、限制等事实，对冲突信息进行人工审阅。

### 高可信 Agent 只读接入

让外部 Agent 读取“待人工审阅的事项”或“指定版本中的指定 record”，但不允许它绕过内部账本直接改值。

## 11. 商业化方向

基于当前产品内核，较自然的商业化方向包括：

- 面向企业知识整理与发布的 SaaS/私有化部署
- 面向审计、法规、法务、投研、政策情报的事实核验工作台
- 面向外部 Agent/插件的高可信知识只读层
- 面向高价值文档流的“证据可追溯结构化抽取”能力输出

## 12. 明确不做的事情

按当前架构与代码边界，织文明确不以这些方向为核心：

- 不把 Agent 当作核心业务地基
- 不把“latest/max(version_no)”当作正式外部契约
- 不把没有证据的抽取结果直接作为最终知识
- 不把人工决议设计成覆盖式更新
- 不把浏览器用户接口和百炼工具接口混为一层
- 不把前端缓存或 UI 聚合作为可信数据来源
- 不把百炼集成描述为稳定生产主链路
