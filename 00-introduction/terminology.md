# Terminology & Concepts

> Standard vocabulary for AI agent systems - speak the same language

## Core Concepts

### Agent

A system that uses an LLM to autonomously make decisions and take actions to accomplish a goal.

**Not to be confused with:**

- **Chatbot** - Conversational interface (may or may not be an agent)
- **Assistant** - Often marketing term; may be simple RAG + chat
- **Co-pilot** - Usually assists humans rather than acting autonomously

---

### Orchestrator

The component that coordinates agent behavior - decides what to do next.

**Also called:**

- Controller
- Planner
- Reasoner

**Responsibilities:**

- Parse LLM output
- Decide next action
- Manage execution flow
- Handle errors and retries

---

### Tool

An external function or API that an agent can invoke.

**Examples:**

- Web search API
- Database query
- Calculator function
- File system access
- Email sender

**Key properties:**

- **Name** - Identifier the LLM uses
- **Description** - What the tool does (critical for LLM understanding)
- **Parameters** - Input schema (usually JSON)
- **Return type** - Output format

---

### Tool Calling (Function Calling)

The mechanism by which an LLM requests to execute a tool.

**Flow:**

1. LLM receives prompt + tool definitions
2. LLM outputs structured tool request (JSON)
3. Orchestrator parses and executes tool
4. Result returned to LLM for next step

**Formats:**

- **OpenAI function calling** - Native support in API
- **Anthropic tool use** - Similar structured format
- **ReAct format** - Text-based with Thought/Action/Observation
- **Custom JSON** - Roll your own schema

---

### Memory

The ability to store and retrieve information across interactions.

**Types:**

#### Short-term Memory (Working Memory)

- Current conversation or task context
- Stored in prompt/context window
- Cleared after task completes

#### Long-term Memory (Persistent)

- Facts, preferences, history stored externally
- Survives across sessions
- Requires retrieval mechanism (vector DB, key-value store)

#### Semantic Memory

- General knowledge and learned patterns
- Often implemented via RAG

#### Episodic Memory

- Specific past interactions or events
- "Remember when we discussed X?"

---

### Context Window

The amount of text (tokens) an LLM can process at once.

**Implications:**

- Limits how much conversation history you can include
- Affects cost (longer context = more tokens)
- Trade-off: More context vs. speed/cost

**Typical sizes (as of 2026):**

- Small models: 4K-8K tokens
- Standard: 32K-128K tokens
- Long-context: 200K-1M+ tokens

---

### Prompt

The input text sent to the LLM, including instructions, context, and user query.

**Components:**

1. **System message** - Role and behavior instructions
2. **Few-shot examples** - Example inputs/outputs
3. **Context** - Relevant background information
4. **User query** - The actual task/question
5. **Tool definitions** - Available functions
6. **History** - Previous conversation turns

---

### Reasoning

The LLM's internal process of deciding what to do.

**Patterns:**

#### ReAct (Reason + Act)

```
Thought: I need to find the weather
Action: search_weather(location="Seattle")
Observation: 65°F, partly cloudy
Thought: Now I can answer the user
Final Answer: It's 65°F in Seattle
```

#### Chain-of-Thought (CoT)

```
Let's think step by step:
1. First, I need the user's location
2. Then I'll check the weather API
3. Finally, I'll format a friendly response
```

#### Plan-and-Execute

```
Plan:
- Step 1: Get location
- Step 2: Fetch weather
- Step 3: Respond

Execution: [execute plan steps]
```

---

### Observation

The result returned after an action (tool call) is executed.

**Example flow:**

- **Action:** `calculate(2 + 2)`
- **Observation:** `4`
- **Next thought:** "The answer is 4"

---

### Router

A component or agent that dispatches tasks to specialized handlers.

**Types:**

- **LLM-based** - Uses reasoning to choose route
- **Rule-based** - Keyword matching or classifiers
- **Hybrid** - Combines both approaches

**Example:**

```
User: "I need a refund"
Router: [classifies as billing issue]
→ Routes to billing_agent
```

---

### Sub-agent (Worker Agent)

A specialized agent that handles a specific domain or task type.

**Example hierarchy:**

- **Manager agent** - Coordinates overall task
  - **Research agent** - Gathers information
  - **Writing agent** - Generates content
  - **Editing agent** - Proofreads output

---

### Workflow vs. Agent

| Workflow         | Agent               |
| ---------------- | ------------------- |
| Predefined steps | Dynamic decisions   |
| Deterministic    | Non-deterministic   |
| If-then logic    | LLM reasoning       |
| State machine    | Autonomous executor |

**When terms blur:** Some systems are "agentic workflows" - workflows with LLM decision points.

---

## Execution Patterns

### Sequential Execution

Tasks run one after another; output of step N feeds into step N+1.

```
Step 1 → Step 2 → Step 3 → Result
```

---

### Parallel Execution

Multiple tasks run simultaneously; results combined afterward.

```
     ┌─ Task A ─┐
Start├─ Task B ─┤→ Combine → Result
     └─ Task C ─┘
```

---

### Conditional Execution

Next step depends on a decision or result.

```
Start → Check condition
         ├─ If true → Path A
         └─ If false → Path B
```

---

### Iterative Execution (Loop)

Repeat actions until goal achieved or limit reached.

```
Start → Action → Check goal
         ↑           |
         └───────────┘ (loop if not done)
```

---

## Error Handling Concepts

### Retry Strategy

Automatically re-attempt failed actions.

**Approaches:**

- **Simple retry** - Try again N times
- **Exponential backoff** - Wait longer between retries
- **Conditional retry** - Retry only for specific errors

---

### Fallback

Alternative action when primary fails.

**Example:**

- Try API call → If fails → Use cached data → If no cache → Return error message

---

### Validation

Check LLM output before executing.

**Types:**

- **Schema validation** - Is JSON formatted correctly?
- **Constraint validation** - Are parameters within bounds?
- **Semantic validation** - Does action make sense in context?

---

### Circuit Breaker

Stop calling a failing service after repeated errors.

**States:**

- **Closed** - Normal operation
- **Open** - Reject requests immediately (service is down)
- **Half-open** - Test if service recovered

---

## Performance Concepts

### Latency

Time from request to response.

**Sources:**

- LLM inference time
- Tool execution time
- Network delays
- Orchestration overhead

**Optimization:**

- Parallel execution
- Caching
- Streaming responses
- Smaller models

---

### Throughput

Number of requests handled per unit time.

**Bottlenecks:**

- Rate limits (API quotas)
- Model concurrency limits
- Database connections
- Cost budgets

---

### Token

The basic unit of text for LLMs.

**Rough conversion:**

- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

**Why it matters:**

- Pricing is per token
- Context window measured in tokens
- Latency increases with token count

---

### Token Budget

The maximum tokens you're willing to spend per task.

**Components:**

- Input tokens (prompt + context)
- Output tokens (LLM response)
- Tool call overhead

**Example:**

- Max budget: 10K tokens
- Prompt: 2K tokens
- LLM response: 500 tokens
- Tool results: 1K tokens
- Remaining: 6.5K tokens for additional steps

---

## Observability Concepts

### Trace

A record of all steps taken by an agent to complete a task.

**Contains:**

- LLM calls with prompts and responses
- Tool executions with inputs and outputs
- Timestamps and durations
- Errors and retries

---

### Span

A single operation within a trace.

**Example span types:**

- LLM inference
- Tool call
- Database query
- External API request

---

### Logging

Recording events for debugging and monitoring.

**Levels:**

- **Debug** - Detailed internal state
- **Info** - Key milestones
- **Warn** - Recoverable issues
- **Error** - Failures requiring attention

---

## Framework-Specific Terms

### LangChain

- **Chain** - Sequence of components
- **Agent executor** - Runs agent loop
- **Callback** - Hook for observability

### LlamaIndex

- **Query engine** - Executes data retrieval
- **Agent runner** - Orchestration layer
- **Index** - Data structure for retrieval

### Semantic Kernel

- **Kernel** - Core orchestration engine
- **Plugin** - Reusable skill or tool
- **Planner** - Generates execution plans

---

## Quick Reference

| Term           | One-Line Definition                                      |
| -------------- | -------------------------------------------------------- |
| Agent          | Autonomous system using LLM to take actions toward goals |
| Tool           | External function agent can invoke                       |
| Memory         | Ability to store/retrieve info across interactions       |
| Orchestrator   | Coordinates agent behavior and execution flow            |
| Router         | Dispatches tasks to specialized handlers                 |
| ReAct          | Reason → Act → Observe loop pattern                      |
| Token          | Basic unit of LLM text (≈0.75 words)                     |
| Trace          | Record of all agent steps in a task                      |
| Context window | Max tokens LLM can process at once                       |
| Fallback       | Alternative when primary action fails                    |

---

## Next Steps

- **Understand the terms?** → Use the [Decision Tree](./decision-tree.md) to pick your pattern
- **Ready to implement?** → Jump to [Core Patterns](../patterns/)
- **Need production concepts?** → Check [Production Engineering](../production/)

---

## Further Reading

- **Prompting:** [Prompt Engineering Guide](https://www.promptingguide.ai/)
- **RAG:** [Retrieval-Augmented Generation explained](https://arxiv.org/abs/2005.11401)
- **ReAct:** [ReAct paper](https://arxiv.org/abs/2210.03629)
