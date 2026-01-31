# Observability for AI Agents

> If you can't see what your agent is doing, you can't debug, optimize, or trust it

## Why Observability Matters

**Production agents are black boxes without observability:**

- Can't debug why agent made a decision
- Can't track costs per user/task
- Can't identify slow operations
- Can't detect degraded performance
- Can't reproduce issues

**Good observability enables:**

- Root cause analysis when things break
- Performance optimization
- Cost attribution and optimization
- Compliance and audit trails
- User experience improvements

---

## Three Pillars of Observability

### 1. Logging

**What:** Discrete events with context
**When:** Always (at appropriate levels)
**Use for:** Debugging, audit trails

### 2. Metrics

**What:** Numeric measurements over time
**When:** Continuous aggregation
**Use for:** Dashboards, alerts, trends

### 3. Traces

**What:** Request flows through distributed systems
**When:** Per-request
**Use for:** Latency analysis, dependency mapping

---

## Logging Best Practices

### Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, level, message, **context):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **context
        }
        self.logger.log(getattr(logging, level), json.dumps(log_entry))

# Usage
logger = StructuredLogger("agent")

logger.log("INFO", "Agent started",
           user_id="user123",
           session_id="sess456",
           agent_type="customer_support")

logger.log("INFO", "Tool called",
           tool_name="search_web",
           arguments={"query": "AI agents"},
           execution_time_ms=245)
```

**Output:**

```json
{
  "timestamp": "2026-01-27T10:30:00",
  "level": "INFO",
  "message": "Tool called",
  "tool_name": "search_web",
  "arguments": { "query": "AI agents" },
  "execution_time_ms": 245
}
```

---

### What to Log

**✅ Always log:**

- Agent initialization
- LLM calls (prompt, model, tokens)
- Tool executions (name, args, result, duration)
- Errors and exceptions
- User interactions
- Performance metrics

**❌ Never log:**

- Passwords, API keys, secrets
- Full user data (PII)
- Credit card numbers
- Unredacted sensitive content

**Example comprehensive logging:**

```python
def execute_agent_task(user_id, task):
    task_id = generate_id()

    logger.log("INFO", "Task started",
               task_id=task_id,
               user_id=user_id,
               task_type=task["type"])

    start_time = time.time()

    try:
        # Log LLM call
        prompt = build_prompt(task)
        logger.log("DEBUG", "LLM call",
                   task_id=task_id,
                   model="gpt-4",
                   prompt_tokens=len(prompt) // 4)

        response = llm.generate(prompt)

        logger.log("DEBUG", "LLM response",
                   task_id=task_id,
                   completion_tokens=len(response) // 4,
                   duration_ms=(time.time() - start_time) * 1000)

        # Log tool calls
        for tool_call in extract_tool_calls(response):
            tool_start = time.time()

            logger.log("INFO", "Tool execution started",
                       task_id=task_id,
                       tool_name=tool_call["name"])

            result = execute_tool(tool_call)

            logger.log("INFO", "Tool execution completed",
                       task_id=task_id,
                       tool_name=tool_call["name"],
                       duration_ms=(time.time() - tool_start) * 1000,
                       success=True)

        total_duration = time.time() - start_time

        logger.log("INFO", "Task completed",
                   task_id=task_id,
                   total_duration_ms=total_duration * 1000,
                   success=True)

        return result

    except Exception as e:
        logger.log("ERROR", "Task failed",
                   task_id=task_id,
                   error_type=type(e).__name__,
                   error_message=str(e),
                   duration_ms=(time.time() - start_time) * 1000,
                   exc_info=True)
        raise
```

---

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# Setup tracer
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)

def traced_agent_execution(task):
    with tracer.start_as_current_span("agent_execution") as span:
        span.set_attribute("task.type", task["type"])
        span.set_attribute("user.id", task["user_id"])

        # Nested span for LLM call
        with tracer.start_as_current_span("llm_call") as llm_span:
            llm_span.set_attribute("model", "gpt-4")
            response = llm.generate(task["prompt"])
            llm_span.set_attribute("tokens.total", len(response) // 4)

        # Nested span for tool execution
        with tracer.start_as_current_span("tool_execution") as tool_span:
            tool_span.set_attribute("tool.name", "search_web")
            result = execute_tool("search_web", {"query": task["query"]})

        return result
```

**Benefits:**

- See full request flow
- Identify bottlenecks
- Track cross-service dependencies
- Debug distributed systems

---

## Metrics Collection

### Key Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters (monotonically increasing)
llm_calls_total = Counter('llm_calls_total', 'Total LLM API calls', ['model', 'status'])
tool_calls_total = Counter('tool_calls_total', 'Total tool executions', ['tool_name', 'status'])

# Histograms (distribution of values)
llm_latency = Histogram('llm_latency_seconds', 'LLM response time', ['model'])
tool_latency = Histogram('tool_latency_seconds', 'Tool execution time', ['tool_name'])

# Gauges (current value)
active_agents = Gauge('active_agents', 'Currently running agents')
token_usage = Gauge('token_usage_total', 'Cumulative token usage')

# Usage
def call_llm(prompt, model="gpt-4"):
    llm_calls_total.labels(model=model, status="started").inc()

    start_time = time.time()

    try:
        response = llm.generate(prompt, model=model)

        duration = time.time() - start_time
        llm_latency.labels(model=model).observe(duration)
        llm_calls_total.labels(model=model, status="success").inc()

        return response
    except Exception as e:
        llm_calls_total.labels(model=model, status="error").inc()
        raise
```

---

## Debugging Tools

### Conversation Replay

```python
class ConversationRecorder:
    def __init__(self):
        self.conversations = {}

    def record_turn(self, session_id, role, content, metadata=None):
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        self.conversations[session_id].append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        })

    def replay(self, session_id):
        """Replay conversation for debugging"""
        conversation = self.conversations.get(session_id, [])

        for turn in conversation:
            print(f"[{turn['timestamp']}] {turn['role']}: {turn['content'][:100]}...")
            if turn['metadata']:
                print(f"  Metadata: {turn['metadata']}")

    def export(self, session_id, format="json"):
        """Export for analysis"""
        conversation = self.conversations.get(session_id, [])

        if format == "json":
            return json.dumps(conversation, indent=2)
        elif format == "markdown":
            md = f"# Conversation {session_id}\n\n"
            for turn in conversation:
                md += f"## {turn['role']} ({turn['timestamp']})\n\n"
                md += f"{turn['content']}\n\n"
            return md

# Usage
recorder = ConversationRecorder()

recorder.record_turn("sess123", "user", "What's the weather?")
recorder.record_turn("sess123", "assistant", "Let me check that for you.",
                    metadata={"tool_called": "get_weather"})

# Later, debug issue
recorder.replay("sess123")
```

---

### LLM Call Inspector

```python
class LLMCallInspector:
    def __init__(self):
        self.calls = []

    def log_call(self, prompt, response, model, tokens, duration_ms):
        self.calls.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "model": model,
            "tokens": tokens,
            "duration_ms": duration_ms,
            "cost": self.calculate_cost(model, tokens)
        })

    def calculate_cost(self, model, tokens):
        rates = {
            "gpt-4": 0.03 / 1000,
            "gpt-3.5-turbo": 0.001 / 1000
        }
        return tokens * rates.get(model, 0)

    def analyze(self):
        """Analyze LLM usage patterns"""
        return {
            "total_calls": len(self.calls),
            "total_tokens": sum(c["tokens"] for c in self.calls),
            "total_cost": sum(c["cost"] for c in self.calls),
            "avg_latency_ms": sum(c["duration_ms"] for c in self.calls) / len(self.calls),
            "by_model": self._group_by_model()
        }

    def _group_by_model(self):
        from collections import defaultdict
        stats = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0})

        for call in self.calls:
            model = call["model"]
            stats[model]["calls"] += 1
            stats[model]["tokens"] += call["tokens"]
            stats[model]["cost"] += call["cost"]

        return dict(stats)

# Usage
inspector = LLMCallInspector()

# After each LLM call
inspector.log_call(
    prompt="What's the capital of France?",
    response="The capital of France is Paris.",
    model="gpt-4",
    tokens=150,
    duration_ms=1200
)

# Analyze usage
print(inspector.analyze())
```

---

## Cost Tracking

### Per-User Cost Attribution

```python
class CostTracker:
    def __init__(self):
        self.costs = defaultdict(lambda: {
            "llm_calls": 0,
            "llm_cost": 0,
            "tool_calls": 0,
            "tool_cost": 0
        })

    def track_llm_call(self, user_id, model, tokens):
        cost = self.calculate_llm_cost(model, tokens)
        self.costs[user_id]["llm_calls"] += 1
        self.costs[user_id]["llm_cost"] += cost

    def track_tool_call(self, user_id, tool_name):
        cost = self.get_tool_cost(tool_name)
        self.costs[user_id]["tool_calls"] += 1
        self.costs[user_id]["tool_cost"] += cost

    def get_user_cost(self, user_id):
        user_costs = self.costs[user_id]
        return {
            **user_costs,
            "total_cost": user_costs["llm_cost"] + user_costs["tool_cost"]
        }

    def get_top_users_by_cost(self, n=10):
        sorted_users = sorted(
            self.costs.items(),
            key=lambda x: x[1]["llm_cost"] + x[1]["tool_cost"],
            reverse=True
        )
        return sorted_users[:n]

# Alert on high costs
def check_user_cost_limits(user_id):
    cost = tracker.get_user_cost(user_id)

    if cost["total_cost"] > USER_DAILY_LIMIT:
        alert(f"User {user_id} exceeded daily limit: ${cost['total_cost']:.2f}")
        return False

    return True
```

---

## Dashboard Visualization

### Example Metrics Dashboard (Pseudo-code for Grafana/DataDog)

```
# Panel 1: Request Rate
Query: rate(agent_requests_total[5m])
Visualization: Time series graph

# Panel 2: Error Rate
Query: rate(agent_errors_total[5m]) / rate(agent_requests_total[5m])
Visualization: Time series graph with threshold alert at 5%

# Panel 3: Latency Percentiles
Query:
  - p50: histogram_quantile(0.5, llm_latency_seconds)
  - p95: histogram_quantile(0.95, llm_latency_seconds)
  - p99: histogram_quantile(0.99, llm_latency_seconds)

# Panel 4: Cost Over Time
Query: increase(token_cost_total[1h])
Visualization: Area chart

# Panel 5: Top Tools Used
Query: topk(5, tool_calls_total)
Visualization: Bar chart
```

---

## Common Pitfalls

### 1. Logging Everything

**Problem:** TB of logs, can't find signal

```python
# ❌ Too much logging
for token in response:
    logger.debug(f"Token: {token}")  # 1000s of logs per response
```

**Solution:** Use appropriate log levels

### 2. Not Correlating Logs

**Problem:** Can't trace request across services

```python
# ❌ No correlation
logger.info("Request started")
logger.info("Tool called")  # Which request?
```

**Solution:** Use correlation IDs

### 3. Ignoring Sensitive Data

**Problem:** Logging passwords, API keys

```python
# ❌ Logging secrets
logger.info(f"API call with key: {api_key}")
```

**Solution:** Redact sensitive data

### 4. No Sampling

**Problem:** 100% trace sampling overwhelming

**Solution:** Sample high-volume paths (e.g., 1%)

---

## References

- **OpenTelemetry:** [Docs](https://opentelemetry.io/)
- **Prometheus:** [Best Practices](https://prometheus.io/docs/practices/)
- **LangSmith:** [LangChain Observability](https://smith.langchain.com/)
- **Weights & Biases:** [LLM Tracing](https://wandb.ai/site/solutions/llm)

---

## Next Steps

- **Need error handling?** → See [Error Handling](./error-handling.md)
- **Cost optimization?** → See [Cost Optimization](./cost-optimization.md)
- **Testing?** → See [Testing Strategies](./testing-strategies.md)
