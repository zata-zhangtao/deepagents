# DeepAgents 业务人员 Agent 平台 — 系统设计文档

> **版本**: v1.0  
> **日期**: 2026-05-10  
> **目标读者**: 技术负责人、后端开发、前端开发、产品经理  
> **状态**: 设计草案

---

## 1. 项目概述

### 1.1 背景与目标

为企业业务人员提供一套**无代码/低代码 Agent 创建平台**。业务人员无需编写代码，即可通过表单配置创建专属 AI Agent，用于数据分析、内容生成、客户服务等场景。

**核心价值主张**:
- 业务人员 5 分钟创建专属 Agent
- 开发者只需维护一套平台代码
- Agent 行为通过文件系统配置驱动，版本可控、可复用

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **代码层锁死基础设施** | model、backend、tools、permissions 由平台固定，业务人员不可修改 |
| **文件层开放业务逻辑** | AGENTS.md、skills/、subagents.yaml 由业务人员配置 |
| **表单层降低门槛** | 提供可视化表单，自动生成底层配置文件 |
| **渐进式开放** | 初级用户用表单，高级用户直接编辑文件 |
| **安全前置** | 权限边界在代码层固定，业务配置无法突破 |

### 1.3 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| Agent 配置包 | Agent Config Package | 一个文件夹，包含 AGENTS.md + skills/ + 可选 subagents.yaml |
| 平台核心 | Platform Core | 开发者维护的 Python 代码，负责加载配置包并创建 Agent |
| 技能模板库 | Skill Template Library | 平台预置的 SKILL.md 模板集合 |
| 预置工具池 | Prebuilt Tool Pool | 平台封装好的工具（查数据库、搜索、发邮件等） |
| 创建向导 | Creation Wizard | 前端表单界面，引导业务人员创建 Agent |

---

## 2. 整体架构

### 2.1 系统分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户层 (User Layer)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Web 管理后台 │  │  创建向导     │  │  Agent 对话页 │  │  技能市场     │    │
│  │ (Agent 列表)  │  │ (表单填空)    │  │ (聊天界面)   │  │ (浏览/下载)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              应用层 (Application Layer)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Agent 工厂 (Agent Factory)                      │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │   │
│  │  │ 配置加载器 │  │ 模板渲染器 │  │ 权限校验器 │  │ Agent 实例化器     │   │   │
│  │  │ (Loader)  │  │ (Renderer)│  │ (Validator)│  │ (Instantiator)   │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      会话管理器 (Session Manager)                    │   │
│  │  - 线程生命周期管理  - 状态持久化  - 多轮对话上下文                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              框架层 (Framework Layer)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     DeepAgents SDK (libs/deepagents)                 │   │
│  │  create_deep_agent() · 中间件栈 · 工具调度 · 子Agent管理             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LangGraph Runtime                                │   │
│  │  图执行 · Checkpoint · 状态管理 · 流式输出                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              基础设施层 (Infrastructure Layer)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   对象存储    │  │   数据库      │  │   缓存       │  │   沙盒服务    │    │
│  │  (S3/OSS)    │  │ (Postgres)   │  │  (Redis)     │  │(LangSmith/  │    │
│  │  存配置包     │  │  存元数据    │  │  存会话状态  │  │ Daytona等)  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent 配置包结构

每个 Agent 对应一个**配置包**，是平台与业务人员的唯一交互界面。

```
agent-configs/{org_id}/{agent_id}/           # 组织隔离
├── AGENTS.md                                # Agent 身份与业务规则 (必需)
├── skills/                                  # 技能目录 (可选)
│   ├── data-query/
│   │   └── SKILL.md
│   └── report-generation/
│       └── SKILL.md
│       └── scripts/                         # 技能配套脚本 (可选)
│           └── generate_chart.py
├── subagents.yaml                           # 子 Agent 配置 (可选)
├── mcp.json                                 # MCP 服务配置 (可选)
└── .metadata.json                           # 平台生成的元数据
```

### 2.3 请求流转时序

```
用户 (业务人员)
    │
    │ 1. 在创建向导填写表单
    ▼
前端服务
    │
    │ 2. POST /api/v1/agents
    │    {name, role, terms, selected_skills, rules}
    ▼
Agent 工厂服务
    │
    │ 3. 生成 AGENTS.md
    │ 4. 从模板库复制 skills/
    │ 5. 写入对象存储
    │ 6. 记录元数据到 Postgres
    ▼
对象存储 (S3)
    │
    │ 7. 配置包持久化
    ▼
[稍后] 用户打开 Agent 对话页
    │
    │ 8. GET /api/v1/agents/{id}/chat
    ▼
Agent 工厂
    │
    │ 9. 从 S3 拉取配置包
    │ 10. 解析 AGENTS.md + skills/
    │ 11. 调用 create_deep_agent()
    ▼
DeepAgents → LangGraph
    │
    │ 12. 执行对话循环
    ▼
用户收到回复
```

---

## 3. 核心模块详细设计

### 3.1 平台核心 — Agent 工厂 (Agent Factory)

**职责**: 唯一创建 Agent 实例的入口。封装所有基础设施细节，暴露简单的配置驱动接口。

**位置**: `platform/core/factory.py`

```python
class AgentFactory:
    """
    Agent 工厂 — 根据配置包路径创建 DeepAgents 实例。
    
    开发者维护此类，业务人员通过配置包与之交互。
    """
    
    def __init__(
        self,
        model: BaseChatModel,                    # 固定模型
        tool_pool: ToolPool,                     # 预置工具池
        backend: BackendProtocol,                # 固定后端
        checkpointer: Checkpointer,              # 固定持久化
        skill_template_library: SkillLibrary,    # 技能模板库
        default_permissions: list[FilesystemPermission],
    ):
        ...
    
    def create_from_package(
        self,
        package_path: str,                       # 配置包路径
        org_id: str,                             # 组织 ID (多租户)
        user_id: str,                            # 用户 ID (权限隔离)
    ) -> CompiledStateGraph:
        """
        从配置包创建 Agent 实例。
        
        流程:
        1. 加载 AGENTS.md → memory
        2. 扫描 skills/ 目录 → skills
        3. 解析 subagents.yaml → subagents (如存在)
        4. 根据 org_id + user_id 注入上下文变量
        5. 组装 create_deep_agent() 参数
        6. 返回编译后的图
        """
        ...
    
    def validate_package(self, package_path: str) -> ValidationResult:
        """
        校验配置包是否合法。
        
        检查项:
        - AGENTS.md 是否存在且非空
        - skills/ 下的 SKILL.md 格式是否正确
        - 引用的工具名是否在预置池内
        - 文件大小限制 (AGENTS.md < 100KB, 单个 skill < 10MB)
        """
        ...
```

#### 3.1.1 预置工具池 (Tool Pool)

平台封装所有业务工具，业务人员**通过 AGENTS.md 描述来间接使用**，而非直接注册工具。

```python
# platform/tools/__init__.py

PREBUILT_TOOLS = [
    # 数据查询类
    QueryDatabaseTool(),           # 安全 SQL 查询 (只读)
    QueryWarehouseTool(),          # 数仓查询
    DescribeTableTool(),           # 表结构查询
    
    # 内容生成类
    GenerateImageTool(),           # 文生图
    GenerateDocumentTool(),        # 生成 Word/PDF
    
    # 通信协作类
    SendEmailTool(),               # 发邮件
    SendDingTalkTool(),            # 发钉钉
    CreateCalendarEventTool(),     # 创建日程
    
    # 外部搜索类
    WebSearchTool(),               # 网页搜索
    SearchInternalDocsTool(),      # 搜索内部文档
    
    # 文件操作类 (由 DeepAgents 内置提供)
    # write_todos, read_file, write_file, edit_file, ls, glob, grep, execute
]
```

**工具安全封装原则**:
- 所有数据库工具使用**只读连接**
- SQL 执行前经过**关键词拦截** (DROP/DELETE/UPDATE 等)
- 自动追加 `LIMIT` 防止大数据量拖垮
- 敏感字段（手机号、身份证）**自动脱敏**

#### 3.1.2 技能模板库 (Skill Library)

```
platform/templates/skills/
├── _manifest.json                    # 模板清单 (名称/描述/分类/标签)
├── data-analysis/
│   ├── SKILL.md                      # 技能定义
│   └── icon.svg
├── report-generation/
│   ├── SKILL.md
│   └── sample_output.md
├── customer-service/
│   ├── SKILL.md
│   └── faq_integration.py            # 技能配套脚本
└── code-review/
    ├── SKILL.md
    └── scripts/
        └── lint_check.py
```

**模板元数据示例** (`_manifest.json`):

```json
{
  "skills": [
    {
      "id": "data-analysis",
      "name": "数据分析",
      "description": "执行 SQL 查询、统计分析、生成图表",
      "category": "数据",
      "tags": ["sql", "chart", "statistics"],
      "required_tools": ["query_database", "describe_table"],
      "complexity": "medium",
      "icon": "data-analysis/icon.svg"
    },
    {
      "id": "report-generation",
      "name": "报告生成",
      "description": "将分析结果整理为结构化报告",
      "category": "内容",
      "tags": ["writing", "markdown"],
      "required_tools": ["write_file", "generate_image"],
      "complexity": "low"
    }
  ]
}
```

### 3.2 配置生成器 (Config Generator)

**职责**: 将前端表单数据转换为标准配置包。

**位置**: `platform/core/generator.py`

```python
class ConfigGenerator:
    """将业务人员表单输入转换为 AGENTS.md + skills/ + subagents.yaml"""
    
    def generate_agents_md(self, form_data: AgentForm) -> str:
        """
        根据表单生成 AGENTS.md。
        
        输入:
        {
            "name": "销售数据分析助手",
            "role": "你是公司的销售数据分析师...",
            "scope": ["销售数据", "客户数据"],
            "terms": {"销售额": "gmv", "华东区": "region='east'"},
            "safety_rules": ["只读", "敏感信息脱敏"],
            "output_format": "表格+结论+图表"
        }
        
        输出: 标准 AGENTS.md Markdown 字符串
        """
        ...
    
    def generate_subagents_yaml(
        self,
        selected_subagents: list[str]
    ) -> str | None:
        """
        根据用户勾选的子 Agent 类型生成 subagents.yaml。
        
        平台预定义子 Agent 模板:
        - researcher: 研究员 (擅长搜索和信息收集)
        - sql-expert: SQL 专家 (擅长复杂查询优化)
        - reviewer: 审稿员 (擅长检查逻辑和格式)
        """
        ...
```

#### 3.2.1 AGENTS.md 模板结构

```markdown
# {name}

## 你的角色
{role_description}

## 业务范围
你可以协助以下业务场景：
{scope_list}

## 你可以访问的数据
{accessible_data}

## 术语映射
业务人员使用的术语与数据库字段对应关系：
{terms_table}

## 安全与约束规则
{security_rules}

## 输出规范
{output_specification}

## 工作流指引
{workflow_guidance}
```

### 3.3 配置包管理器 (Package Manager)

**职责**: 管理配置包的 CRUD、版本、发布。

**位置**: `platform/core/package_manager.py`

```python
class PackageManager:
    """管理 Agent 配置包的全生命周期。"""
    
    def create(
        self,
        org_id: str,
        creator_id: str,
        form_data: AgentForm,
    ) -> PackageMetadata:
        """
        创建新配置包。
        流程:
        1. 生成配置包内容
        2. 写入对象存储 (路径: s3://agents/{org_id}/{agent_id}/)
        3. 写入元数据库
        4. 返回包元数据
        """
        ...
    
    def update(
        self,
        agent_id: str,
        org_id: str,
        form_data: AgentForm,
    ) -> PackageMetadata:
        """更新配置包，自动创建版本历史。"""
        ...
    
    def get(self, agent_id: str, org_id: str) -> PackageContent:
        """读取配置包完整内容。"""
        ...
    
    def list_skills(self, agent_id: str) -> list[SkillInfo]:
        """列出某 Agent 已安装的技能。"""
        ...
    
    def add_skill(self, agent_id: str, skill_id: str) -> None:
        """为 Agent 添加技能 (从模板库复制)。"""
        ...
    
    def remove_skill(self, agent_id: str, skill_id: str) -> None:
        """移除 Agent 的某个技能。"""
        ...
```

### 3.4 会话管理器 (Session Manager)

**职责**: 管理用户与 Agent 的对话会话，处理多轮对话、上下文、并发。

**位置**: `platform/core/session.py`

```python
class SessionManager:
    """
    会话管理器。
    
    关键设计:
    - 每个会话 = LangGraph 的一个 thread
    - 使用 checkpointer 持久化对话历史
    - 支持流式输出 (SSE/WebSocket)
    - 支持会话恢复 (断线重连)
    """
    
    def create_session(
        self,
        agent_id: str,
        user_id: str,
        org_id: str,
        metadata: dict | None = None,
    ) -> Session:
        """创建新会话。"""
        ...
    
    async def stream_chat(
        self,
        session_id: str,
        message: str,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式对话。
        
        输出事件类型:
        - "thinking"     : Agent 正在思考
        - "tool_call"    : 调用了某个工具
        - "tool_result"  : 工具返回结果
        - "message"      : Agent 文本回复
        - "error"        : 执行错误
        - "done"         : 本轮完成
        """
        ...
    
    def get_history(self, session_id: str) -> list[Message]:
        """获取会话历史。"""
        ...
    
    def delete_session(self, session_id: str) -> None:
        """删除会话 (软删除，保留元数据)。"""
        ...
```

---

## 4. 数据模型

### 4.1 实体关系图 (ER Diagram)

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   Organization│       │    Agent     │       │   Session    │
│   (组织)      │1    N │  (Agent配置)  │1    N │   (会话)     │
├─────────────┤       ├─────────────┤       ├─────────────┤
│ id (PK)     │◄──────│ id (PK)     │◄──────│ id (PK)     │
│ name        │       │ org_id (FK) │       │ agent_id(FK) │
│ settings    │       │ name        │       │ user_id     │
└─────────────┘       │ description │       │ thread_id   │
                      │ package_path│       │ status      │
                      │ creator_id  │       │ created_at  │
                      │ status      │       │ updated_at  │
                      │ version     │       └─────────────┘
                      │ created_at  │
                      │ updated_at  │
                      └─────────────┘
                            │
                            │1    N
                            ▼
                      ┌─────────────┐
                      │ AgentVersion│
                      │ (版本历史)   │
                      ├─────────────┤
                      │ id (PK)     │
                      │ agent_id(FK)│
                      │ version_no  │
                      │ package_path│
                      │ change_log  │
                      │ created_by  │
                      │ created_at  │
                      └─────────────┘
                            │
                            │N    M
                            ▼
                      ┌─────────────┐
                      │   Skill     │
                      │  (技能关联)  │
                      ├─────────────┤
                      │ id (PK)     │
                      │ agent_id(FK)│
                      │ skill_id    │
                      │ skill_path  │
                      │ added_at    │
                      └─────────────┘
```

### 4.2 核心表结构

#### 4.2.1 `agents` 表

```sql
CREATE TABLE agents (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id          UUID NOT NULL REFERENCES organizations(id),
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    
    -- 配置包存储位置
    package_path    VARCHAR(512) NOT NULL,
    
    -- 当前版本
    current_version INTEGER NOT NULL DEFAULT 1,
    
    -- 状态: draft / active / archived / error
    status          VARCHAR(32) NOT NULL DEFAULT 'draft',
    
    -- 创建/更新信息
    creator_id      UUID NOT NULL,
    updater_id      UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- 索引
    CONSTRAINT idx_agents_org UNIQUE(org_id, name)
);
```

#### 4.2.2 `agent_versions` 表

```sql
CREATE TABLE agent_versions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    version_no      INTEGER NOT NULL,
    package_path    VARCHAR(512) NOT NULL,
    change_log      TEXT,
    created_by      UUID NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT idx_agent_versions_unique UNIQUE(agent_id, version_no)
);
```

#### 4.2.3 `sessions` 表

```sql
CREATE TABLE sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL REFERENCES agents(id),
    user_id         UUID NOT NULL,
    
    -- LangGraph thread_id (用于恢复对话)
    thread_id       VARCHAR(128) NOT NULL UNIQUE,
    
    -- 会话标题 (由首条消息自动生成摘要)
    title           VARCHAR(256),
    
    -- 状态: active / paused / closed
    status          VARCHAR(32) NOT NULL DEFAULT 'active',
    
    -- 消息计数 (缓存，避免频繁查询)
    message_count   INTEGER NOT NULL DEFAULT 0,
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    CONSTRAINT idx_sessions_agent_user UNIQUE(agent_id, user_id, thread_id)
);
```

#### 4.2.4 `skill_templates` 表 (平台技能模板库)

```sql
CREATE TABLE skill_templates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_id        VARCHAR(64) NOT NULL UNIQUE,   -- 如 "data-analysis"
    name            VARCHAR(128) NOT NULL,
    description     TEXT NOT NULL,
    category        VARCHAR(64) NOT NULL,         -- 数据/内容/协作/...
    tags            VARCHAR(64)[],
    
    -- 模板文件在对象存储中的路径
    template_path   VARCHAR(512) NOT NULL,
    
    -- 依赖的预置工具
    required_tools  VARCHAR(64)[],
    
    complexity      VARCHAR(16) NOT NULL DEFAULT 'medium',  -- low/medium/high
    is_builtin      BOOLEAN NOT NULL DEFAULT true,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

---

## 5. API 接口设计

### 5.1 RESTful API 概览

```
POST   /api/v1/agents                    # 创建 Agent
GET    /api/v1/agents                    # 列表查询
GET    /api/v1/agents/{id}               # 获取详情
PUT    /api/v1/agents/{id}               # 更新配置
DELETE /api/v1/agents/{id}               # 删除 Agent
POST   /api/v1/agents/{id}/fork          #  fork 一个新版本

GET    /api/v1/agents/{id}/versions      # 获取版本历史
POST   /api/v1/agents/{id}/versions/{v}/restore  # 回滚到某版本

GET    /api/v1/skills                    # 获取技能模板列表
POST   /api/v1/agents/{id}/skills/{skill_id}     # 给 Agent 添加技能
DELETE /api/v1/agents/{id}/skills/{skill_id}     # 移除技能

POST   /api/v1/agents/{id}/chat          # 发起对话 (SSE 流式)
GET    /api/v1/agents/{id}/sessions      # 获取会话列表
GET    /api/v1/sessions/{id}/messages    # 获取会话消息历史
DELETE /api/v1/sessions/{id}             # 删除会话
```

### 5.2 关键接口详细定义

#### 5.2.1 创建 Agent

```http
POST /api/v1/agents
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "name": "销售数据分析助手",
  "description": "帮助销售团队快速查询和分析销售数据",
  
  "role_config": {
    "role_description": "你是公司的销售数据分析师，精通 SQL 和数据分析",
    "business_scope": ["销售数据查询", "客户分析", "趋势预测"],
    "forbidden_scope": ["财务数据", "人事数据", "薪资信息"]
  },
  
  "data_access": {
    "allowed_datasets": ["dws_sales_daily", "dim_customer", "dim_product"],
    "default_time_range": "最近90天"
  },
  
  "terms": [
    {"business_term": "销售额", "technical_field": "dws_sales_daily.gmv"},
    {"business_term": "订单量", "technical_field": "dws_sales_daily.order_cnt"},
    {"business_term": "华东区", "technical_field": "region_name IN ('上海','江苏','浙江','安徽')"}
  ],
  
  "safety_rules": {
    "read_only": true,
    "sensitive_data_masking": true,
    "max_rows_per_query": 1000,
    "require_approval_for": ["export"]  // 导出操作需要审批
  },
  
  "output_format": {
    "default_style": "表格+结论+建议",
    "include_charts": true,
    "chart_types": ["line", "bar", "pie"]
  },
  
  "skills": ["data-analysis", "report-generation"],
  
  "subagents": ["sql-expert"],
  
  "advanced": {
    "system_prompt_extra": "额外补充指令...",
    "temperature": 0.2
  }
}
```

**响应**:

```json
{
  "id": "agent-uuid",
  "name": "销售数据分析助手",
  "status": "active",
  "package_path": "s3://agents/org-uuid/agent-uuid/v1/",
  "current_version": 1,
  "created_at": "2026-05-10T10:00:00Z",
  "preview_url": "/api/v1/agents/agent-uuid/preview"
}
```

#### 5.2.2 流式对话 (SSE)

```http
POST /api/v1/agents/{agent_id}/chat
Content-Type: application/json
Authorization: Bearer {jwt_token}

{
  "message": "上季度华东区各品类销售额趋势如何？",
  "session_id": "optional-existing-session-id",
  "stream": true
}
```

**SSE 响应流**:

```
event: thinking
data: {"type": "thinking", "content": "正在分析您的需求..."}

event: tool_call
data: {"type": "tool_call", "tool": "query_database", "args": {"sql": "SELECT..."}}

event: tool_result
data: {"type": "tool_result", "tool": "query_database", "status": "success", "rows": 12}

event: message
data: {"type": "message", "content": "根据查询结果，上季度华东区销售额...", "delta": "根据查询"}

event: message
data: {"type": "message", "content": "根据查询结果，上季度华东区销售额...", "delta": "结果，上季度"}

event: done
data: {"type": "done", "session_id": "sess-uuid", "message_count": 5}
```

---

## 6. 安全设计

### 6.1 多层安全模型

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: 平台层安全 (开发者控制，不可突破)                        │
│ ───────────────────────────────────────────────────────────────│
│ • 数据库连接使用只读账号                                          │
│ • SQL 关键词拦截 (DROP/DELETE/UPDATE/INSERT/ALTER/CREATE/TRUNCATE)│
│ • 自动追加 LIMIT，防止全表扫描                                    │
│ • 敏感字段自动脱敏 (手机号/身份证/银行卡)                          │
│ • 文件系统权限边界 (只能读写 /workspace 目录)                      │
│ • 命令执行沙盒化 (LangSmith/Daytona/Modal)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Agent 层安全 (AGENTS.md 声明，提示词约束)                │
│ ───────────────────────────────────────────────────────────────│
│ • 业务范围声明 (只能查哪些数据)                                    │
│ • 安全规则声明 (只读、脱敏、审批)                                  │
│ • 输出格式约束 (禁止返回原始敏感数据)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: 组织层安全 (多租户隔离)                                   │
│ ───────────────────────────────────────────────────────────────│
│ • 组织间配置包物理隔离 (S3 路径前缀)                               │
│ • JWT Token 携带 org_id，API 层校验                                │
│ • 数据访问范围按组织过滤                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 权限模型

```python
# 平台固定的文件系统权限
DEFAULT_PERMISSIONS = [
    FilesystemPermission(
        operations=["read", "write", "edit", "ls", "glob", "grep"],
        paths=["/workspace/*"],
        mode="allow",
    ),
    FilesystemPermission(
        operations=["read", "write", "edit", "execute"],
        paths=["/etc/*", "/root/*", "/home/*", "/var/*"],
        mode="deny",
    ),
    FilesystemPermission(
        operations=["write", "edit"],
        paths=["/workspace/AGENTS.md"],  # AGENTS.md 只读，防止自我篡改
        mode="deny",
    ),
]
```

### 6.3 数据脱敏规则

```python
SENSITIVE_FIELD_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "email": r"[\w.-]+@[\w.-]+\.\w+",
    "bank_card": r"\d{16,19}",
}

AUTO_MASKING_RULES = {
    "phone": lambda x: x[:3] + "****" + x[-4:],
    "id_card": lambda x: x[:6] + "********" + x[-4:],
    "email": lambda x: x.split("@")[0][:2] + "***@" + x.split("@")[1],
    "bank_card": lambda x: x[:4] + " **** **** " + x[-4:],
}
```

---

## 7. 部署架构

### 7.1 生产环境部署图

```
                              ┌─────────────┐
                              │   CDN/WAF   │
                              └──────┬──────┘
                                     │
                              ┌──────▼──────┐
                              │  Nginx/ALB  │
                              │  (负载均衡)  │
                              └──────┬──────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
     ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
     │  Web 服务   │          │  Web 服务   │          │  Web 服务   │
     │  (FastAPI)  │          │  (FastAPI)  │          │  (FastAPI)  │
     │             │          │             │          │             │
     │ Agent工厂   │          │ Agent工厂   │          │ Agent工厂   │
     │ 会话管理    │          │ 会话管理    │          │ 会话管理    │
     └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
     ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
     │   Redis     │          │   Postgres  │          │  对象存储   │
     │  (会话缓存)  │          │  (元数据)   │          │ (配置包)   │
     └─────────────┘          └─────────────┘          └─────────────┘
                                     │
                              ┌──────▼──────┐
                              │ LangSmith   │
                              │  Cloud/Self │
                              │  Hosted     │
                              │             │
                              │ • 沙盒服务   │
                              │ • Trace     │
                              │ • Deploy    │
                              └─────────────┘
```

### 7.2 服务拆分建议

| 服务 | 职责 | 技术栈 | 部署方式 |
|------|------|--------|---------|
| **API Gateway** | 路由、认证、限流 | Nginx / Kong / AWS ALB | 容器/K8s |
| **Web Service** | HTTP API、Agent 工厂、会话管理 | FastAPI + uvicorn | 容器/K8s |
| **Worker** | 异步任务 (配置包生成、版本备份) | Celery / RQ | 容器/K8s |
| **LangGraph Server** | Agent 图执行 (可选独立部署) | LangGraph Cloud / 自托管 | 托管/容器 |
| **Postgres** | 元数据、会话索引 | PostgreSQL 15+ | RDS/托管 |
| **Redis** | 会话缓存、限频计数 | Redis 7+ | ElastiCache/托管 |
| **对象存储** | 配置包、附件 | S3 / OSS / MinIO | 托管/自托管 |

### 7.3 环境变量配置

```bash
# 模型配置
MODEL_PROVIDER=anthropic
MODEL_NAME=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...

# 数据库
DATABASE_URL=postgresql://user:pass@host:5432/agent_platform
REDIS_URL=redis://host:6379/0

# 对象存储
S3_BUCKET=agent-configs
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=...
S3_SECRET_KEY=...

# LangSmith (可选，用于沙盒和追踪)
LANGSMITH_API_KEY=ls-...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# 安全
JWT_SECRET=...
ENCRYPTION_KEY=...
MAX_AGENTS_PER_ORG=50
MAX_SESSIONS_PER_AGENT=100

# 功能开关
ENABLE_SANDBOX=true
ENABLE_CODE_EXECUTION=false
ENABLE_IMAGE_GENERATION=true
```

---

## 8. 前端设计要点

### 8.1 创建向导页面流程

```
Step 1: 基础信息
├─ Agent 名称 * (文本输入)
├─ 描述 (文本域)
└─ 头像/图标 (选择器)

Step 2: 角色定义
├─ 角色描述 * (文本域，带模板建议)
├─ 业务范围 (多选：数据查询/报告生成/客户分析/...)
└─ 禁止事项 (文本域)

Step 3: 数据访问
├─ 可用数据源 (多选：销售库/客户库/产品库/...)
├─ 默认时间范围 (下拉：7天/30天/90天/自定义)
└─ 术语映射表 (Key-Value 表单，可动态添加)

Step 4: 安全与约束
├─ [✓] 只读模式 (固定勾选，不可取消)
├─ [✓] 敏感信息脱敏 (默认勾选)
├─ 单次最大返回行数 (数字输入，默认1000)
└─ 需要审批的操作 (多选：导出/跨库查询/...)

Step 5: 技能装配
├─ 技能市场 (卡片网格，带分类筛选)
│   ├─ [✓] 数据分析
│   ├─ [✓] 报告生成
│   └─ [ ] 竞品调研 (灰色，无权限)
└─ 已选技能预览 (可拖拽排序)

Step 6: 预览与创建
├─ 自动生成的 AGENTS.md 预览 (只读，可展开)
├─ 技能列表确认
└─ [创建 Agent] [保存草稿]
```

### 8.2 对话页面设计

```
┌────────────────────────────────────────────────────────────────────┐
│ 销售数据分析助手                                    [设置] [历史]  │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 🤖 你好！我是销售数据分析助手。我可以帮你：                   │   │
│  │ • 查询销售数据、客户数据                                     │   │
│  │ • 生成周报、月报、趋势分析                                   │   │
│  │ • 请直接描述你的需求                                         │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 👤 上季度华东区各品类销售额趋势如何？                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │ 🤖 正在分析...                                               │   │
│  │                                                              │   │
│  │ 🔍 执行查询: dws_sales_daily (12行结果)                      │   │
│  │ 📊 生成图表: trend_chart.png                                 │   │
│  │                                                              │   │
│  │ 根据查询结果，上季度华东区销售额呈现以下趋势：                │   │
│  │ ...                                                          │   │
│  │                                                              │   │
│  │ [查看图表] [下载报告] [追问]                                  │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│ [快捷提问: 今日销售额 |  top10客户 |  库存预警 ]                    │
│ [______________________________________________] [发送]             │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. 扩展性设计

### 9.1 新增数据源

当业务需要连接新的数据库时：

1. **开发者**: 在 `platform/tools/` 新增 `QueryNewDBTool`
2. **开发者**: 在数据库配置中心注册新数据源连接串
3. **平台**: 创建向导的"可用数据源"下拉框自动出现新选项
4. **业务人员**: 创建 Agent 时勾选即可使用

**不需要修改 Agent 工厂代码** — 工具池是动态加载的。

### 9.2 新增技能模板

1. **开发者/高级业务人员**: 编写 SKILL.md
2. **上传至**: `platform/templates/skills/{new-skill}/`
3. **更新**: `_manifest.json`
4. **平台**: 技能市场自动展示

### 9.3 多模型支持

```python
# platform/core/model_registry.py

MODEL_REGISTRY = {
    "standard": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "max_tokens": 8192,
    },
    "advanced": {
        "provider": "anthropic",
        "model": "claude-opus-4",
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
        "max_tokens": 8192,
    },
    "economy": {
        "provider": "openai",
        "model": "gpt-5.4-nano",
        "cost_per_1k_input": 0.0001,
        "cost_per_1k_output": 0.0004,
        "max_tokens": 4096,
    },
}
```

业务人员创建 Agent 时选择"标准/高级/经济"档位，平台自动映射到具体模型。

### 9.4 插件化工具系统

未来支持第三方开发者贡献工具：

```python
# 工具包标准接口
class ToolPackage:
    @property
    def name(self) -> str: ...
    
    @property
    def tools(self) -> list[BaseTool]: ...
    
    @property
    def required_env_vars(self) -> list[str]: ...
    
    def validate_config(self, config: dict) -> ValidationResult: ...
```

通过 `mcp.json` 或平台插件市场安装。

---

## 10. 监控与运维

### 10.1 关键指标 (SLI/SLO)

| 指标 | 目标 | 测量方式 |
|------|------|---------|
| Agent 创建成功率 | > 99% | 创建 API 成功率 |
| 首次响应延迟 (TTFT) | < 2s | 从发送消息到首 token 返回 |
| 完整响应延迟 | < 30s | 简单查询 < 10s，复杂分析 < 60s |
| 会话可用性 | > 99.5% | 健康检查 + 错误率 |
| SQL 查询成功率 | > 98% | Agent 生成并执行 SQL 的成功率 |
| 成本/会话 | 可控 | 按 token 用量和模型档位计费 |

### 10.2 日志与追踪

```python
# 每个会话的追踪结构
{
    "trace_id": "trace-uuid",
    "session_id": "sess-uuid",
    "agent_id": "agent-uuid",
    "org_id": "org-uuid",
    "user_id": "user-uuid",
    
    "events": [
        {"type": "user_message", "timestamp": "...", "content_length": 50},
        {"type": "llm_first_token", "timestamp": "...", "latency_ms": 1200},
        {"type": "tool_call", "tool": "query_database", "args_hash": "..."},
        {"type": "tool_result", "duration_ms": 500, "rows": 12},
        {"type": "llm_complete", "timestamp": "...", "total_tokens": 2048},
    ],
    
    "cost": {
        "input_tokens": 1024,
        "output_tokens": 1024,
        "model": "claude-sonnet-4-5",
        "estimated_usd": 0.018
    }
}
```

### 10.3 告警规则

- **错误率突增**: 5分钟内 SQL 查询失败率 > 10%
- **延迟异常**: P95 响应延迟 > 60s 持续 3 分钟
- **成本异常**: 单个会话 token 消耗 > 50万 (可能进入死循环)
- **安全事件**: Agent 尝试执行被拦截的 SQL 关键词

---

## 11. 开发路线图 (Roadmap)

### Phase 1: MVP (4-6 周)

- [x] Agent 工厂核心 (`create_deep_agent` 封装)
- [x] 配置包管理系统 (AGENTS.md + skills/ 的 CRUD)
- [x] 基础创建向导 (表单 → 配置包)
- [x] SSE 流式对话
- [x] 预置工具池: SQL 查询、文件操作、网页搜索
- [x] 多租户基础 (org_id 隔离)

### Phase 2: 增强 (4-6 周)

- [ ] 技能模板市场 (浏览、搜索、一键安装)
- [ ] 版本管理 (历史版本、回滚、对比)
- [ ] 会话管理 (历史列表、标题自动生成、会话导出)
- [ ] 高级安全 (敏感数据脱敏、操作审批流)
- [ ] 沙盒执行 (Python 数据分析、图表生成)

### Phase 3: 企业级 (6-8 周)

- [ ] SSO / LDAP 集成
- [ ] 审计日志 (完整操作记录)
- [ ] 成本中心 (按组织/用户计费、用量配额)
- [ ] MCP 插件市场 (第三方工具接入)
- [ ] A/B 测试 (不同 AGENTS.md 配置的效果对比)

---

## 12. 附录

### 12.1 参考示例

| 示例项目 | 位置 | 参考价值 |
|---------|------|---------|
| Text-to-SQL Agent | `examples/text-to-sql-agent/` | SQL 工具 + AGENTS.md + Skills 的完整示范 |
| Content Builder | `examples/content-builder-agent/` | 文件驱动配置 + 子 Agent YAML 外部化 |
| Deploy Coding Agent | `examples/deploy-coding-agent/` | `deepagents deploy` + LangSmith 沙盒 |

### 12.2 相关文档

- DeepAgents SDK 文档: https://docs.langchain.com/oss/python/deepagents/overview
- LangGraph 文档: https://docs.langchain.com/oss/python/langgraph/overview
- Agent Protocol: https://github.com/langchain-ai/agent-protocol

---

> **本设计文档是活文档 (Living Document)**。随着开发推进，各模块的详细设计应在对应子目录的 `DESIGN.md` 中维护，本文件保持架构层面的总览。
