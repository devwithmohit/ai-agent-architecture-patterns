# Sequential Chain Pattern

> Linear pipeline where output of each step becomes input to the next - like an assembly line

## When to Use

**Perfect for:**

- Predictable multi-step workflows with clear dependencies
- Data transformation pipelines (extract → transform → load)
- Content generation workflows (research → outline → draft → edit)
- Document processing (parse → analyze → summarize → format)
- Order-of-operations tasks where step N+1 needs output from step N

**Ideal scenarios:**

- "Research topic → Create outline → Write article → Proofread"
- "Extract invoice data → Validate → Calculate totals → Generate report"
- "Analyze code → Find bugs → Suggest fixes → Format output"
- "Scrape website → Clean data → Analyze sentiment → Visualize"

## When NOT to Use

**❌ Avoid when:**

- **Steps can run in parallel** - Use parallel execution instead (faster)
- **Path is unpredictable** - Use ReAct or router pattern
- **Single step sufficient** - Direct tool calling is simpler
- **Need branching logic** - Conditional workflows or router better fit
- **Early steps might fail often** - Wastes cost on later steps that never run
- **Real-time interaction needed** - User can't provide input mid-chain

**Cost trap:** If early steps fail, you've wasted all the cost of that failed run. No partial credit.

## Architecture Diagram

```mermaid
flowchart LR
    Input[User Input] --> Step1[Step 1: Research]
    Step1 --> Output1[Output 1]
    Output1 --> Step2[Step 2: Outline]
    Step2 --> Output2[Output 2]
    Output2 --> Step3[Step 3: Draft]
    Step3 --> Output3[Output 3]
    Output3 --> Step4[Step 4: Proofread]
    Step4 --> Final[Final Output]

    style Step1 fill:#ffebcc
    style Step2 fill:#ffebcc
    style Step3 fill:#ffebcc
    style Step4 fill:#ffebcc
```

## Flow Breakdown

### Basic Chain Structure

```python
def sequential_chain(input_data):
    # Step 1
    result_1 = step_1(input_data)

    # Step 2 uses output from Step 1
    result_2 = step_2(result_1)

    # Step 3 uses output from Step 2
    result_3 = step_3(result_2)

    # Final output
    return result_3
```

### Example: Article Generation

**Input:** "Write an article about AI safety"

**Step 1: Research**

```
Input: "AI safety"
Action: Search web for recent articles
Output: ["Article 1 summary", "Article 2 summary", "Article 3 summary"]
```

**Step 2: Create Outline**

```
Input: [Article summaries from Step 1]
Action: Generate structured outline
Output:
  I. Introduction to AI Safety
  II. Current Challenges
  III. Proposed Solutions
  IV. Future Outlook
```

**Step 3: Write Draft**

```
Input: [Outline from Step 2] + [Research from Step 1]
Action: Generate full article text
Output: [3000-word draft article]
```

**Step 4: Proofread and Edit**

```
Input: [Draft from Step 3]
Action: Check grammar, improve clarity, fact-check
Output: [Final polished article]
```

**Total time:** 30-60 seconds (sequential)
**Total cost:** ~8,000 tokens = $0.08

## Tradeoffs Table

| Aspect             | Pro                                    | Con                                          |
| ------------------ | -------------------------------------- | -------------------------------------------- |
| **Predictability** | Deterministic flow, easy to understand | Can't adapt to unexpected situations         |
| **Cost**           | Pay once per step                      | Early failure wastes entire run              |
| **Latency**        | Optimized for total time               | Must wait for each step to complete          |
| **Debugging**      | Clear which step failed                | Can't inspect intermediate state mid-run     |
| **Complexity**     | Simple to implement                    | Rigid structure                              |
| **Reliability**    | Each step validated before next        | Failure propagates (garbage in, garbage out) |
| **Scalability**    | Easy to parallelize multiple chains    | Individual chain is sequential               |

## Implementation Approaches

### Approach 1: Simple Function Composition

```python
def step_1(data):
    return f"Processed: {data}"

def step_2(data):
    return f"Enhanced: {data}"

def step_3(data):
    return f"Finalized: {data}"

def run_chain(input_data):
    return step_3(step_2(step_1(input_data)))
```

**Pros:** Dead simple, minimal code
**Cons:** Hard to debug, no error handling, no observability

---

### Approach 2: Explicit Chain with Logging

```python
class Chain:
    def __init__(self):
        self.steps = []
        self.history = []

    def add_step(self, name, function):
        self.steps.append({"name": name, "fn": function})

    def run(self, input_data):
        data = input_data

        for step in self.steps:
            print(f"Running: {step['name']}")
            try:
                data = step['fn'](data)
                self.history.append({
                    "step": step['name'],
                    "output": data,
                    "success": True
                })
            except Exception as e:
                self.history.append({
                    "step": step['name'],
                    "error": str(e),
                    "success": False
                })
                raise

        return data

# Usage
chain = Chain()
chain.add_step("Research", research_step)
chain.add_step("Outline", outline_step)
chain.add_step("Draft", draft_step)
chain.add_step("Edit", edit_step)

result = chain.run("AI safety")
```

**Pros:** Observable, debuggable, error tracking
**Cons:** More boilerplate

---

### Approach 3: LangChain Sequential Chain

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Define individual chains
research_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="Research this topic: {topic}"
    ),
    output_key="research"
)

outline_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["research"],
        template="Create outline from: {research}"
    ),
    output_key="outline"
)

# Combine into sequential chain
full_chain = SequentialChain(
    chains=[research_chain, outline_chain, draft_chain, edit_chain],
    input_variables=["topic"],
    output_variables=["final_article"]
)

result = full_chain({"topic": "AI safety"})
```

**Pros:** Framework support, built-in features
**Cons:** Framework dependency, less control

---

### Approach 4: Async Sequential (for I/O-bound steps)

```python
import asyncio

async def step_1_async(data):
    # Simulate API call
    await asyncio.sleep(1)
    return f"Step 1: {data}"

async def step_2_async(data):
    await asyncio.sleep(1)
    return f"Step 2: {data}"

async def run_chain_async(input_data):
    data = input_data
    data = await step_1_async(data)
    data = await step_2_async(data)
    return data

# Run
result = asyncio.run(run_chain_async("input"))
```

**Pros:** Non-blocking I/O, efficient for API calls
**Cons:** Doesn't reduce total time (still sequential)

## Real-world Example: Customer Onboarding

**Task:** Onboard new customer with personalized experience

**Step 1: Validate Input**

```python
def validate_customer(customer_data):
    """
    Input: {name, email, company, plan}
    Output: Validated data or error
    """
    if not email_is_valid(customer_data['email']):
        raise ValueError("Invalid email")
    return customer_data
```

**Step 2: Create Account**

```python
def create_account(validated_data):
    """
    Input: Validated customer data
    Output: {customer_id, account_created_at}
    """
    customer_id = database.create_customer(validated_data)
    return {**validated_data, "customer_id": customer_id}
```

**Step 3: Generate Welcome Email**

```python
def generate_welcome_email(account_data):
    """
    Input: Account data with customer_id
    Output: Personalized email content
    """
    prompt = f"Generate welcome email for {account_data['name']}, plan: {account_data['plan']}"
    email_content = llm.generate(prompt)
    return {**account_data, "email_content": email_content}
```

**Step 4: Send Email and Create Tasks**

```python
def send_and_setup(email_data):
    """
    Input: Account data + email content
    Output: Confirmation
    """
    email_service.send(email_data['email'], email_data['email_content'])
    task_manager.create_onboarding_tasks(email_data['customer_id'])
    return {"status": "complete", "customer_id": email_data['customer_id']}
```

**Run chain:**

```python
chain = Chain()
chain.add_step("Validate", validate_customer)
chain.add_step("Create Account", create_account)
chain.add_step("Generate Email", generate_welcome_email)
chain.add_step("Send & Setup", send_and_setup)

result = chain.run({
    "name": "Jane Doe",
    "email": "jane@example.com",
    "company": "Acme Inc",
    "plan": "Enterprise"
})
```

**Output:** Customer onboarded, email sent, tasks created
**Time:** ~5 seconds total
**Cost:** ~500 tokens ($0.005)

## Cost Analysis

**Token breakdown per step:**

| Step      | Input Tokens | Output Tokens | Cost (@$0.01/1K) |
| --------- | ------------ | ------------- | ---------------- |
| Research  | 100          | 800           | $0.009           |
| Outline   | 900          | 300           | $0.012           |
| Draft     | 1200         | 2000          | $0.032           |
| Edit      | 3200         | 2000          | $0.052           |
| **Total** | **5,400**    | **5,100**     | **$0.105**       |

**Cost optimization:**

- Use cheaper model for early steps (research, outline)
- Use expensive model for final step (editing quality matters)
- Cache research results if topic repeats
- Summarize large intermediate outputs before passing to next step

**Failure cost:**

- If Step 1 fails: $0.009 wasted
- If Step 3 fails: $0.053 wasted (all prior steps)
- **Solution:** Validate early, fail fast

## Common Pitfalls

### 1. Output Format Mismatch

**Problem:** Step 2 expects JSON, Step 1 returns text

```python
# Step 1
def research(topic):
    return "Here are some facts about X..."  # Plain text

# Step 2
def outline(research_data):
    # Expects: {"facts": [...]}
    facts = research_data["facts"]  # ❌ Error: str has no key 'facts'
```

**Solution:** Enforce schemas between steps

```python
from pydantic import BaseModel

class ResearchOutput(BaseModel):
    facts: list[str]
    sources: list[str]

def research(topic) -> ResearchOutput:
    return ResearchOutput(facts=[...], sources=[...])
```

### 2. Context Loss Across Steps

**Problem:** Later steps need info from early steps but only get immediate prior output

```python
# Step 1: User provides preferences
preferences = {"tone": "casual", "length": "short"}

# Step 2: Research (loses preferences)
research_data = research(topic)

# Step 3: Wants to use preferences
draft = write_draft(research_data)  # ❌ No access to preferences!
```

**Solution:** Pass full context or use shared state

```python
class ChainContext:
    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

# Steps now receive context
def write_draft(context):
    research = context.get("research")
    preferences = context.get("preferences")
    # Use both
```

### 3. No Partial Results on Failure

**Problem:** Chain fails at step 3, lose all progress

```python
result_1 = expensive_step_1()  # $0.05
result_2 = expensive_step_2(result_1)  # $0.05
result_3 = buggy_step_3(result_2)  # ❌ Fails - $0.10 wasted
```

**Solution:** Checkpoint intermediate results

```python
def run_chain_with_checkpoints(input_data, cache_key):
    if cache := get_cache(f"{cache_key}_step1"):
        result_1 = cache
    else:
        result_1 = step_1(input_data)
        set_cache(f"{cache_key}_step1", result_1)

    # Repeat for each step
```

### 4. Bloated Intermediate Data

**Problem:** Step 1 returns 10K tokens, passed through all steps

```python
research = search_web(topic)  # Returns 50 full articles (10K tokens)
outline = create_outline(research)  # Only uses 5% of it
draft = write_draft(research, outline)  # 10K tokens in prompt again
```

**Solution:** Summarize between steps

```python
research = search_web(topic)
research_summary = summarize(research, max_tokens=500)
outline = create_outline(research_summary)
```

### 5. Rigid Error Handling

**Problem:** One step fails, entire chain aborts

```python
try:
    result = run_chain(input)
except Exception:
    return "Chain failed"  # No detail on which step
```

**Solution:** Graceful degradation

```python
def run_chain_with_fallbacks(input_data):
    try:
        result_1 = step_1(input_data)
    except Exception as e:
        result_1 = fallback_step_1(input_data)

    try:
        result_2 = step_2(result_1)
    except Exception as e:
        result_2 = fallback_step_2(result_1)

    # Continue with best available data
```

### 6. Ignoring Step Dependencies

**Problem:** Steps assumed independent, but aren't

```python
# Assumes steps are independent - WRONG
result_2 = step_2(input)  # Skipping step_1
result_3 = step_3(result_2)  # ❌ Fails - needs data from step_1
```

**Solution:** Explicit dependency tracking

```python
steps = [
    {"name": "research", "fn": research, "depends_on": []},
    {"name": "outline", "fn": outline, "depends_on": ["research"]},
    {"name": "draft", "fn": draft, "depends_on": ["research", "outline"]}
]
```

## Advanced Patterns

### Branching Sequential Chains

Execute different sequences based on conditions:

```python
def conditional_chain(input_data):
    result_1 = step_1(input_data)

    if result_1["type"] == "A":
        # Path A: steps 2a → 3a
        result_2 = step_2a(result_1)
        result_3 = step_3a(result_2)
    else:
        # Path B: steps 2b → 3b
        result_2 = step_2b(result_1)
        result_3 = step_3b(result_2)

    return result_3
```

### Nested Chains

Chain of chains:

```python
def sub_chain_1(data):
    return step_c(step_b(step_a(data)))

def sub_chain_2(data):
    return step_f(step_e(step_d(data)))

def main_chain(data):
    intermediate = sub_chain_1(data)
    final = sub_chain_2(intermediate)
    return final
```

### Retry Chain

Retry failed steps with modified input:

```python
def run_chain_with_retry(input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return run_chain(input_data)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # Modify input based on error
            input_data = adjust_input_for_retry(input_data, e)
```

## Testing & Validation

### Unit Test Each Step

```python
def test_step_1():
    input_data = {"topic": "AI"}
    output = step_1(input_data)
    assert "research" in output
    assert len(output["research"]) > 0

def test_step_2():
    mock_research = {"research": ["fact 1", "fact 2"]}
    output = step_2(mock_research)
    assert "outline" in output
```

### Integration Test Full Chain

```python
def test_full_chain():
    input_data = {"topic": "test topic"}
    result = run_chain(input_data)
    assert result["status"] == "complete"
    assert "final_output" in result
```

### Test Failure Modes

```python
def test_step_failure():
    with pytest.raises(ValueError):
        run_chain({"invalid": "data"})

def test_partial_failure():
    # Mock step 2 to fail
    with patch('step_2', side_effect=Exception("Fail")):
        result = run_chain_with_fallbacks({"topic": "AI"})
        assert result["step_2_fallback_used"] == True
```

## Production Considerations

### Observability

Log each step:

```json
{
  "chain_id": "abc123",
  "input": { "topic": "AI safety" },
  "steps": [
    {
      "name": "research",
      "start_time": "2026-01-27T10:00:00Z",
      "end_time": "2026-01-27T10:00:05Z",
      "duration_ms": 5000,
      "tokens_used": 900,
      "cost": 0.009,
      "success": true
    },
    {
      "name": "outline",
      "start_time": "2026-01-27T10:00:05Z",
      "end_time": "2026-01-27T10:00:08Z",
      "duration_ms": 3000,
      "tokens_used": 1200,
      "cost": 0.012,
      "success": true
    }
  ],
  "total_duration_ms": 35000,
  "total_cost": 0.105,
  "final_status": "complete"
}
```

### Circuit Breakers

Stop chain if step is consistently failing:

```python
from collections import defaultdict

failure_counts = defaultdict(int)

def run_with_circuit_breaker(step_name, step_fn, data):
    if failure_counts[step_name] > 10:
        raise Exception(f"Circuit open for {step_name}")

    try:
        result = step_fn(data)
        failure_counts[step_name] = 0  # Reset on success
        return result
    except Exception as e:
        failure_counts[step_name] += 1
        raise
```

### Progressive Disclosure

Return partial results as they complete:

```python
async def run_chain_streaming(input_data, callback):
    result_1 = await step_1(input_data)
    callback({"step": 1, "complete": True, "data": result_1})

    result_2 = await step_2(result_1)
    callback({"step": 2, "complete": True, "data": result_2})

    # User sees progress in real-time
```

## Comparison: Sequential vs Other Patterns

| Pattern        | Use When                    | Latency        | Cost     | Flexibility |
| -------------- | --------------------------- | -------------- | -------- | ----------- |
| **Sequential** | Predictable dependent steps | N × step_time  | Linear   | Low         |
| **Parallel**   | Independent tasks           | max(step_time) | Same     | Low         |
| **ReAct**      | Unpredictable exploration   | Variable       | Variable | High        |
| **Router**     | Need specialized handling   | 1 × step_time  | Low      | Medium      |

**When to combine:**

- **Sequential + Parallel** - Some steps run in parallel within sequence
- **Sequential + Router** - Route to different sequential chains
- **Sequential + ReAct** - One step in chain is a ReAct loop

## References

- **LangChain Sequential Chain:** [Docs](https://python.langchain.com/docs/modules/chains/foundational/sequential_chains)
- **Function Composition:** [Wiki](https://en.wikipedia.org/wiki/Function_composition)
- **Pipeline Pattern:** [Martin Fowler](https://martinfowler.com/articles/collection-pipeline/)

## Next Steps

- **Need parallel execution?** → See [Parallel Execution](./parallel-execution.md)
- **Need dynamic decisions?** → See [Router Agent](./router-agent.md)
- **Need exploration?** → See [ReAct Pattern](./react-reasoning-acting.md)
- **Production deployment?** → See [Error Handling](../production/error-handling.md)
