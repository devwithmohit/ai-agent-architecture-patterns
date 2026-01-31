# What is an AI Agent?

> Clear definitions to avoid confusion and set expectations

## The Core Definition

An **AI agent** is a system that uses a language model to autonomously make decisions about which actions to take to accomplish a goal.

**Key characteristics:**

1. **Autonomy** - Makes decisions without human intervention at each step
2. **Goal-oriented** - Works toward an objective, not just responding to prompts
3. **Action-taking** - Interacts with external systems (APIs, databases, tools)
4. **Iterative** - Can observe results and adjust its approach

## What Agents Are NOT

### Agent vs. Chatbot

| Chatbot                     | Agent                      |
| --------------------------- | -------------------------- |
| Responds to user messages   | Pursues goals autonomously |
| Stateless or simple context | Maintains working memory   |
| No external actions         | Calls tools and APIs       |
| Single-turn or conversation | Multi-step reasoning       |

**Example:**

- **Chatbot:** "What's the weather?" → "It's sunny"
- **Agent:** "Plan my outdoor event" → Checks weather, suggests dates, books venue, sends invites

### Agent vs. Workflow

| Workflow            | Agent                         |
| ------------------- | ----------------------------- |
| Predefined sequence | Dynamic decision-making       |
| Deterministic       | Non-deterministic (LLM-based) |
| If-then logic       | Reasoning and planning        |
| Cheaper, faster     | More flexible, expensive      |

**Example:**

- **Workflow:** Order → Payment → Shipping (fixed steps)
- **Agent:** "Resolve customer issue" → Determines if refund, replacement, or escalation is needed

### Agent vs. RAG System

| RAG System            | Agent                           |
| --------------------- | ------------------------------- |
| Retrieve → Generate   | Reason → Act → Observe → Repeat |
| Single retrieval step | Multiple tool calls             |
| Document-focused      | Action-focused                  |
| Stateless enhancement | Stateful execution              |

**Example:**

- **RAG:** "Explain this policy" → Retrieves docs → Generates answer
- **Agent:** "File expense report" → Retrieves policy → Validates receipt → Submits form → Notifies manager

## Agent Spectrum

Not all agents are equal. Think of it as a spectrum:

```
Simple                                                      Complex
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
  Tool      ReAct    Chain of   Router    Hierarchical  Multi-Agent
  Caller    Loop     Thought    Agent     Agent         Swarms
```

### 1. Tool Caller (Simplest)

- LLM decides which function to call
- Single-turn: Prompt → Tool selection → Result
- **Example:** "What's 2+2?" → Calls calculator → Returns 4

### 2. ReAct Agent (Common)

- Reason → Act → Observe loop
- Multi-turn with self-reflection
- **Example:** Research assistant that searches, reads, synthesizes

### 3. Chain-of-Thought Agent

- Explicit reasoning steps before acting
- Better for complex problem-solving
- **Example:** Math tutor that shows work step-by-step

### 4. Router Agent

- Distributes tasks to specialized sub-agents
- Centralized coordinator
- **Example:** Customer service routing to billing/technical/sales

### 5. Hierarchical Agent

- Manager agents supervise worker agents
- Multi-level delegation
- **Example:** Project manager → Dev lead → Individual devs

### 6. Multi-Agent Swarms (Most Complex)

- Autonomous agents collaborate/compete
- Emergent behavior
- **Example:** Simulation systems, distributed problem-solving

## When to Use Agents

### ✅ Good Use Cases

- **Unpredictable inputs** - Can't script all scenarios
- **External data needed** - API calls, database lookups
- **Multi-step reasoning** - Requires planning and adaptation
- **Contextual decisions** - Needs to interpret nuance
- **Exploration tasks** - Open-ended research or discovery

### ❌ Bad Use Cases

- **Deterministic workflows** - Use automation instead
- **Latency-critical** - Agents are slower than hardcoded logic
- **Strict accuracy requirements** - LLMs hallucinate
- **Simple lookups** - RAG or database query is cheaper
- **High-volume, low-value tasks** - Cost prohibitive

## Decision Framework

Ask yourself:

1. **Is the path predictable?** → If yes, use a workflow
2. **Is it a single question?** → If yes, use RAG or direct LLM call
3. **Does it need external actions?** → If yes, consider agent
4. **Is the cost justified?** → If no, simplify the approach

**Rule of thumb:** Use the simplest solution that works. Agents are powerful but expensive and complex.

## Key Takeaways

- **Agents ≠ Chatbots** - Agents take autonomous actions
- **Agents ≠ Workflows** - Agents make dynamic decisions
- **Not all problems need agents** - Use the right tool for the job
- **Start simple** - Tool-calling agents before multi-agent systems

## Next Steps

- **Understand the spectrum?** → Read [Terminology](./terminology.md)
- **Ready to pick a pattern?** → Use the [Decision Tree](./decision-tree.md)
- **Want implementation details?** → Jump to [Core Patterns](../patterns/)
