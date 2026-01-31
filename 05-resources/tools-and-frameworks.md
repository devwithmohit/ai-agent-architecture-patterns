# Tools & Frameworks for AI Agents

> Curated list of production-ready tools, libraries, and platforms

## 🦜 Agent Frameworks

### LangChain

**Language:** Python, JavaScript
**GitHub:** [langchain-ai/langchain](https://github.com/langchain-ai/langchain) - 85K+ ⭐
**Best for:** Rapid prototyping, extensive tool ecosystem

**Key Features:**

- 100+ pre-built tools and integrations
- Built-in memory management
- LangSmith for observability
- LangServe for deployment

**Installation:**

```bash
pip install langchain langchain-openai
```

**When to use:** Standard agent patterns, quick MVP, multi-provider support

**When to avoid:** Performance-critical apps (high abstraction overhead)

---

### LlamaIndex

**Language:** Python, TypeScript
**GitHub:** [run-llama/llama_index](https://github.com/run-llama/llama_index) - 32K+ ⭐
**Best for:** RAG-heavy applications, document processing

**Key Features:**

- Purpose-built for retrieval
- 50+ data loaders
- Advanced indexing strategies
- Query engines

**Installation:**

```bash
pip install llama-index
```

**When to use:** Document-heavy workflows, knowledge bases, RAG at scale

**When to avoid:** Non-RAG agent patterns (limited compared to LangChain)

---

### LangGraph

**Language:** Python
**GitHub:** [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)
**Best for:** Stateful, cyclical agent workflows

**Key Features:**

- Graph-based agent orchestration
- Built-in persistence
- Human-in-the-loop support
- Streaming

**Installation:**

```bash
pip install langgraph
```

**When to use:** Complex state management, multi-step workflows with branches

---

### AutoGen (Microsoft)

**Language:** Python
**GitHub:** [microsoft/autogen](https://github.com/microsoft/autogen) - 26K+ ⭐
**Best for:** Multi-agent conversations

**Key Features:**

- Conversational agent framework
- Built-in code execution
- Group chat for agents
- Human proxy agent

**Installation:**

```bash
pip install pyautogen
```

**When to use:** Multiple agents collaborating, conversational workflows

---

### CrewAI

**Language:** Python
**GitHub:** [joaomdmoura/crewAI](https://github.com/joaomdmoura/crewAI) - 15K+ ⭐
**Best for:** Role-based multi-agent systems, hierarchical task delegation

**Key Features:**

- Role assignment for agents
- Task delegation
- Sequential and parallel execution
- Process management

**Installation:**

```bash
pip install crewai
```

**When to use:** Team-like agent collaboration, clear role separation

---

### Vercel AI SDK

**Language:** TypeScript/JavaScript
**GitHub:** [vercel/ai](https://github.com/vercel/ai) - 8K+ ⭐
**Best for:** Edge-deployed streaming agents, React/Next.js integration

**Key Features:**

- React Server Components integration
- Edge runtime optimization
- Streaming responses with UI state
- Built-in function calling
- OpenAI, Anthropic, Mistral support

**Installation:**

```bash
npm install ai
```

**When to use:** Frontend-first applications, Next.js/React apps, edge deployment

**When to avoid:** Backend-heavy orchestration, Python-based teams

---

### Semantic Kernel (Microsoft)

**Language:** C#, Python, Java
**GitHub:** [microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) - 19K+ ⭐
**Best for:** Enterprise .NET environments

**Key Features:**

- Native .NET support
- Plugin-based architecture
- Planning capabilities
- Memory connectors

**Installation:**

```bash
pip install semantic-kernel
```

**When to use:** .NET shops, enterprise integration, Azure ecosystem

---

## 🔧 LLM Providers

### OpenAI

**Models:** GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
**Function Calling:** ✅ Native
**Pricing:** $0.01-0.03/1K tokens
**Best for:** General-purpose agents, function calling

**SDK:**

```bash
pip install openai
```

**Key Feature:** Most reliable function calling, extensive tooling

---

### Anthropic

**Models:** Claude 3 (Opus, Sonnet, Haiku)
**Function Calling:** ✅ Tool use
**Pricing:** $0.015-0.075/1K tokens
**Best for:** Long context (200K), reasoning-heavy tasks

**SDK:**

```bash
pip install anthropic
```

**Key Feature:** Best reasoning, largest context window

---

### Google Gemini

**Models:** Gemini 1.5 Pro, Flash
**Function Calling:** ✅ Native
**Pricing:** $0.00125-0.005/1K tokens (cheapest)
**Best for:** Cost-sensitive applications, multi-modal

**SDK:**

```bash
pip install google-generativeai
```

**Key Feature:** 1M+ context window, lowest cost

**Additional Resources:**

- **Docs:** [ai.google.dev](https://ai.google.dev/)
- **Agent support:** Native function calling, multi-modal agents
- **Best for:** Gemini-first applications, massive context needs, multi-modal workflows

---

### Cohere

**Models:** Command R, Command R+
**Function Calling:** ✅ Tools
**Pricing:** $0.001-0.003/1K tokens
**Best for:** Enterprise RAG, multilingual

**SDK:**

```bash
pip install cohere
```

**Key Feature:** Enterprise features, RAG optimized

---

### Local Models (Ollama)

**Models:** Llama 3, Mistral, Mixtral
**Function Calling:** ⚠️ Limited
**Pricing:** Free (self-hosted)
**Best for:** Privacy-critical, air-gapped environments

**Installation:**

```bash
curl https://ollama.ai/install.sh | sh
ollama run llama3
```

**Key Feature:** No API costs, full data privacy

---

## 💾 Vector Databases

### Pinecone

**Type:** Managed cloud
**Pricing:** $70/mo (Starter)
**Best for:** Production RAG, low-latency

**Features:**

- Managed service
- Automatic scaling
- Hybrid search
- Metadata filtering

**When to use:** Don't want to manage infrastructure

---

### Weaviate

**Type:** Open-source + managed
**Pricing:** Free (self-hosted), $25+/mo (cloud)
**Best for:** Complex search, multi-tenancy

**Features:**

- GraphQL API
- Hybrid search
- Multi-modal support
- Self-hosted option

**When to use:** Need flexibility, cost control

---

### Qdrant

**Type:** Open-source + managed
**Pricing:** Free (self-hosted), $0.20/GB (cloud)
**Best for:** High-performance, filtering-heavy

**Features:**

- Fast filtering
- Payload-based search
- Quantization
- Distributed mode

**When to use:** Complex metadata filtering required

---

### ChromaDB

**Type:** Open-source (embedded)
**Pricing:** Free
**Best for:** Development, small datasets (<1M vectors)

**Installation:**

```bash
pip install chromadb
```

**When to use:** Prototyping, local development, embedded use

---

### pgvector

**Type:** PostgreSQL extension
**Pricing:** PostgreSQL hosting costs
**Best for:** Existing PostgreSQL users, small-medium scale

**Installation:**

```sql
CREATE EXTENSION vector;
```

**When to use:** Already using PostgreSQL, want simple setup

---

## 🛠️ Development Tools

### LangSmith

**Purpose:** Agent observability & debugging
**Provider:** LangChain
**Pricing:** Free tier, $39+/mo

**Features:**

- Trace LLM calls
- Prompt management
- Evaluation datasets
- Production monitoring

**When to use:** Debugging complex agent flows

---

### Weights & Biases (W&B)

**Purpose:** Experiment tracking
**Pricing:** Free (individuals), $50+/mo (teams)

**Features:**

- LLM call tracking
- Cost monitoring
- A/B testing
- Team collaboration

**When to use:** Optimizing prompts, tracking costs

---

### PromptLayer

**Purpose:** Prompt versioning & analytics
**Pricing:** $49+/mo

**Features:**

- Prompt registry
- Version control
- Analytics
- Team sharing

**When to use:** Managing prompt library across team

---

### Helicone

**Purpose:** LLM observability
**Pricing:** Free (1K req/mo), $20+/mo

**Features:**

- Request logging
- Cost tracking
- Caching layer
- Rate limiting

**When to use:** Production monitoring, cost control

---

### Langfuse

**Purpose:** Open-source LLM observability & analytics
**Pricing:** Open-source (self-host) or cloud ($49+/mo)
**GitHub:** [langfuse/langfuse](https://github.com/langfuse/langfuse)

**Features:**

- Framework-agnostic tracing (LangChain, LlamaIndex, custom)
- Detailed cost analytics per user/session/prompt
- Prompt versioning and management
- A/B testing for prompts
- LLM playground for testing
- Evaluation datasets
- Production monitoring dashboards
- Self-hosting option for data privacy

**Installation:**

```bash
pip install langfuse
```

**Integration:**

```python
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-..."
)

# Manual tracing
trace = langfuse.trace(name="agent-query", user_id="user123")
span = trace.span(name="llm-call")

response = llm.generate(prompt)

span.end(output=response, metadata={"cost": 0.0045})
trace.end()
```

**When to use:**

- Framework-agnostic observability needed
- Detailed cost tracking per user/session required
- Want self-hosting option for data privacy
- Need prompt versioning and A/B testing
- Team collaboration on prompt optimization

**Comparison to LangSmith:** More flexible (works with any framework), open-source option, stronger cost analytics

---

### Arize AI

**Purpose:** Production ML monitoring for LLMs
**Pricing:** Custom (enterprise)
**Website:** [arize.com](https://arize.com/)

**Features:**

- LLM performance monitoring
- Embedding drift detection
- Model comparison across deployments
- Hallucination detection
- Prompt template monitoring
- Automated alerting on quality degradation
- Integration with existing ML monitoring
- Root cause analysis

**Integration:**

```python
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes

arize_client = Client(
    api_key="...",
    space_key="..."
)

# Log predictions
response = arize_client.log(
    model_id="llm-agent",
    model_version="v1",
    model_type=ModelTypes.GENERATIVE_LLM,
    prediction_label=llm_output,
    actual_label=ground_truth,  # if available
    embedding_features={"prompt_embedding": embedding}
)
```

**When to use:**

- Enterprise ML operations
- Need comprehensive production monitoring
- Detecting embedding/prompt drift over time
- Combining LLM + traditional ML monitoring
- Large-scale deployments (>100K requests/day)

**Comparison to other tools:** Enterprise-focused, stronger drift detection, higher cost

---

## 📊 Evaluation Tools

### OpenAI Evals

**Purpose:** LLM evaluation framework
**Type:** Open-source
**GitHub:** [openai/evals](https://github.com/openai/evals)

**Use cases:**

- Benchmark agent performance
- Regression testing
- Model comparison

---

### Promptfoo

**Purpose:** LLM testing & evaluation
**Type:** Open-source + SaaS
**GitHub:** [promptfoo/promptfoo](https://github.com/promptfoo/promptfoo)

**Features:**

- Test prompts across models
- Automated red teaming
- Performance comparison

---

### LangChain Evaluators

**Purpose:** Built-in evaluation
**Type:** Library

**Evaluators:**

- QA correctness
- Criteria-based
- Pairwise comparison
- LLM-as-judge

---

## 🔐 Security Tools

### Prompt Injection Detection

**Tools:**

- [Rebuff](https://github.com/woop/rebuff) - Prompt injection detector
- [LLM Guard](https://github.com/protectai/llm-guard) - Security toolkit
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - Safety rails

---

### PII Detection

**Tools:**

- [Presidio](https://github.com/microsoft/presidio) - PII detection & anonymization (Microsoft)
- [spaCy](https://spacy.io/) - NER for PII
- [AWS Comprehend](https://aws.amazon.com/comprehend/) - PII detection service

---

## 🚀 Deployment Platforms

### Modal

**Purpose:** Serverless LLM apps
**Pricing:** Pay-per-use

**Features:**

- GPU access
- Container deployment
- Scheduled jobs
- Webhooks

**When to use:** Python apps, need GPUs, serverless

---

### Vercel

**Purpose:** Next.js AI apps
**Pricing:** Free tier, $20+/mo

**Features:**

- Edge functions
- Streaming support
- Built-in observability

**When to use:** JavaScript/TypeScript, web apps

---

### Steamship

**Purpose:** Agent hosting platform
**Pricing:** $0.03/min runtime

**Features:**

- Managed agent hosting
- Built-in memory
- Tool library
- Multi-tenancy

**When to use:** Don't want to manage infrastructure

---

## 📚 Learning Resources

### Interactive Platforms

**Courses:**

- [DeepLearning.AI LangChain Course](https://www.deeplearning.ai/short-courses/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Academy](https://academy.langchain.com/)

**Playgrounds:**

- [OpenAI Playground](https://platform.openai.com/playground)
- [Anthropic Console](https://console.anthropic.com/)
- [Google AI Studio](https://aistudio.google.com/)

---

## 🎯 Decision Matrix

| **Need**           | **Recommended Tool**                       |
| ------------------ | ------------------------------------------ |
| Quick prototype    | LangChain                                  |
| RAG-heavy          | LlamaIndex                                 |
| Multi-agent        | AutoGen or CrewAI                          |
| Stateful workflows | LangGraph                                  |
| .NET environment   | Semantic Kernel                            |
| Vector search      | Pinecone (managed) or Qdrant (self-hosted) |
| Observability      | LangSmith or Helicone                      |
| Security           | LLM Guard + Presidio                       |
| Evaluation         | Promptfoo + OpenAI Evals                   |

---

## 🆕 Emerging Tools (2025)

**Watch these:**

- **Marvin** - AI engineering framework
- **Instructor** - Structured LLM outputs
- **DSPy** - Programming (not prompting) LLMs
- **Guidance** - Structured generation

---

## ⚠️ Deprecated Technologies

> Historical approaches documented for teams migrating away from them

### OpenAI Assistants API

**Status:** 🚫 **Deprecated (Sunset: July 2026)**
**Replacement:** Custom agents with Chat Completions API + LangGraph

**Why deprecated:**

- Limited customization and control
- Opaque pricing (hidden token multipliers)
- Vendor lock-in to OpenAI ecosystem
- Poor observability and debugging

**Migration path:**

1. Audit current Assistants API usage and costs
2. Implement equivalent logic with Chat Completions + function calling
3. Use LangGraph for stateful workflows (replaces threads)
4. Migrate file storage to your own system (S3, GCS, etc.)
5. Implement custom memory management

**Resources:**

- [Migration guide](../03-comparisons/openai-assistants-vs-custom-agents.md)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)

---

### GPT-3.5-turbo (legacy versions)

**Status:** ⚠️ **Legacy (use GPT-4o-mini instead)**
**Replacement:** GPT-4o-mini (faster, cheaper, smarter)

**Why legacy:**

- GPT-4o-mini outperforms at lower cost
- Better function calling reliability
- Faster response times
- More recent training data

**Migration:** Simple drop-in replacement

```python
# BEFORE
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Legacy
    messages=messages
)

# AFTER
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Recommended
    messages=messages
)
```

---

### LangChain Agents (Pre-0.1.0)

**Status:** ⚠️ **Legacy API (use LangGraph instead)**
**Replacement:** LangGraph for stateful agents

**Why deprecated:**

- Limited state management
- Difficult debugging
- Less control over execution flow
- No built-in persistence

**Migration path:**

- **Simple agents:** Migrate to LangGraph's ReAct pattern
- **Complex workflows:** Use LangGraph's graph-based orchestration
- **Stateful agents:** LangGraph provides built-in checkpointing

**Resources:**

- [LangGraph migration guide](https://python.langchain.com/docs/langgraph)

---

### Pinecone Starter Plan (Legacy)

**Status:** ⚠️ **Legacy tier**
**Replacement:** Pinecone Serverless (2024+)

**Why legacy:**

- Pod-based architecture more expensive
- Manual scaling required
- Higher latency
- No auto-scaling

**Migration:** Pinecone provides automatic migration tools

---

### OpenAI Fine-tuning for Agents

**Status:** ⚠️ **Not recommended for most use cases**
**Replacement:** Prompt engineering + RAG + few-shot examples

**Why not recommended:**

- High cost ($10-100+ per fine-tune)
- Maintenance burden (model drift)
- Prompt engineering often sufficient
- GPT-4 class models rarely need fine-tuning

**When fine-tuning still makes sense:**

- Highly specialized domain language
- Extremely cost-sensitive (many millions of calls)
- Consistent output format requirements

**Better alternatives:**

1. Prompt engineering with examples
2. RAG for knowledge augmentation
3. Structured output with Pydantic/JSON mode
4. Chain-of-thought prompting

---

### Langchain.agents.AgentExecutor (Legacy)

**Status:** ⚠️ **Maintenance mode**
**Replacement:** LangGraph or custom orchestration

**Why maintenance mode:**

- Poor error handling
- Limited observability
- Difficult to debug
- No built-in state management

**Migration example:**

```python
# LEGACY (AgentExecutor)
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": query})

# RECOMMENDED (LangGraph)
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [("user", query)]})
```

---

### Vector DB Self-Hosting for Small Projects

**Status:** ⚠️ **Overkill for most MVPs**
**Replacement:** Managed services (Pinecone, Weaviate Cloud)

**Why not recommended for small teams:**

- Operational complexity
- Scaling challenges
- Monitoring and maintenance
- Security patching

**When self-hosting makes sense:**

- > 10M vectors
- Strict data residency requirements
- High query volume (>1000 QPS)
- Team has dedicated infrastructure expertise

**For MVPs:** Use managed services, migrate to self-hosted later if needed

---

## 📊 Technology Lifecycle

| Technology                  | Status             | Timeline         | Action                 |
| --------------------------- | ------------------ | ---------------- | ---------------------- |
| **OpenAI Assistants API**   | 🚫 Deprecated      | Sunset July 2026 | Migrate ASAP           |
| **GPT-3.5-turbo**           | ⚠️ Legacy          | Still supported  | Consider upgrade       |
| **LangChain AgentExecutor** | ⚠️ Maintenance     | Ongoing support  | Plan migration         |
| **Pinecone Pods**           | ⚠️ Legacy tier     | Still supported  | Migrate to Serverless  |
| **Fine-tuning for agents**  | ⚠️ Not recommended | N/A              | Use prompt engineering |

---

## Migration Support

**Resources:**

- [OpenAI Assistants → Custom migration](../03-comparisons/openai-assistants-vs-custom-agents.md)
- [LangChain → LangGraph migration](https://python.langchain.com/docs/langgraph)
- [Framework comparison](../03-comparisons/langchain-vs-llamaindex-vs-custom.md)

**Community:**

- Discord: Ask in `#migrations` channels
- GitHub Discussions: Search for migration issues
- Professional services: Most framework vendors offer migration help

---

## Contributing

Suggest a tool? Requirements:

- Active maintenance (commit in last 3 months)
- 500+ GitHub stars OR commercial backing
- Production-ready (not alpha)
- Clear differentiation from existing tools

---

## Next Steps

- **Research papers?** → See [Papers](./papers.md)
- **Join communities?** → See [Communities](./communities.md)
- **Start building?** → See [Core Patterns](../01-patterns/)
