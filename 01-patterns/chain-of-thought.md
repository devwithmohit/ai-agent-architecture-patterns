# Chain-of-Thought Pattern

> Explicit step-by-step reasoning before taking action - "thinking out loud"

## When to Use

**Perfect for:**

- Complex problem-solving requiring logical reasoning
- Math and calculation-heavy tasks
- Multi-step logical deductions
- Tasks where showing work/reasoning is valuable
- Debugging issues that need systematic diagnosis
- Educational contexts (tutoring, explanations)

**Ideal scenarios:**

- "Solve this word problem: If train A leaves..."
- "Debug: Why is my React component not re-rendering?"
- "Explain: What's the best investment strategy given X, Y, Z?"
- "Analyze: What's the root cause of this performance issue?"

## When NOT to Use

**❌ Avoid when:**

- **Simple lookups** - "What's the weather?" doesn't need reasoning chain
- **Time-sensitive tasks** - CoT adds latency (extra tokens to generate)
- **Cost-constrained** - Reasoning output increases token usage 2-3x
- **User doesn't care about reasoning** - Just want the answer, not the work
- **Reasoning is obvious** - "What's 2+2?" doesn't need step-by-step
- **Non-reasoning tasks** - Image generation, simple classification

**Cost trap:** CoT makes LLM output much longer. If users skip the reasoning, you're paying for tokens no one reads.

## Architecture Diagram

```mermaid
flowchart LR
    Query[User Query] --> Prompt[Prompt with CoT Instructions]
    Prompt --> LLM[LLM]
    LLM --> Step1[Step 1: Identify variables]
    Step1 --> Step2[Step 2: Apply formula]
    Step2 --> Step3[Step 3: Calculate result]
    Step3 --> Step4[Step 4: Verify answer]
    Step4 --> Answer[Final Answer]

    style Step1 fill:#e1f5ff
    style Step2 fill:#e1f5ff
    style Step3 fill:#e1f5ff
    style Step4 fill:#e1f5ff
```

## Flow Breakdown

### Basic CoT Prompt Structure

```
System: Let's think step by step.

User: {question}
```

**LLM Response:**

```
Step 1: [First logical step]
Step 2: [Second logical step]
Step 3: [Third logical step]
Therefore: [Final answer]
```

### Example: Math Problem

**Question:** "If Sarah has 3 apples and buys 2 more packs with 5 apples each, how many apples does she have?"

**CoT Response:**

```
Step 1: Sarah starts with 3 apples
Step 2: She buys 2 packs, each with 5 apples
Step 3: Calculate apples from packs: 2 × 5 = 10 apples
Step 4: Add to original: 3 + 10 = 13 apples
Therefore: Sarah has 13 apples total.
```

### Advanced: Zero-Shot CoT

**Prompt:** Just add "Let's think step by step" to any question

```python
query = "What's the square root of 144?"
prompt = f"{query}\n\nLet's think step by step:"
```

**Response:**

```
Let's think step by step:
1. We need to find a number that, when multiplied by itself, equals 144
2. I'll try some values:
   - 10 × 10 = 100 (too small)
   - 12 × 12 = 144 (correct!)
3. Therefore, √144 = 12
```

### Few-Shot CoT

**Provide examples in prompt:**

```
Q: Roger has 5 tennis balls. He buys 2 more cans with 3 balls each. How many balls does he have?
A: Let's think step by step:
- Roger starts with 5 balls
- He buys 2 cans with 3 balls each: 2 × 3 = 6 balls
- Total: 5 + 6 = 11 balls

Q: A restaurant has 23 tables. 12 are occupied. Then 5 more groups arrive and 3 leave. How many tables are occupied?
A: Let's think step by step:
- Start with 12 occupied tables
- 5 groups arrive: 12 + 5 = 17 occupied
- 3 groups leave: 17 - 3 = 14 occupied
- Answer: 14 tables are occupied

Q: {user_question}
A: Let's think step by step:
```

## Tradeoffs Table

| Aspect            | Pro                                 | Con                             |
| ----------------- | ----------------------------------- | ------------------------------- |
| **Accuracy**      | 30-50% improvement on complex tasks | Minimal gain on simple tasks    |
| **Cost**          | Only LLM tokens, no tools           | 2-3x more output tokens         |
| **Latency**       | Single LLM call                     | Longer generation time          |
| **Transparency**  | Full reasoning visible              | Users may not read it           |
| **Debuggability** | Easy to spot logical errors         | Can't fix without regenerating  |
| **Reliability**   | Reduces hallucination               | Can still make logical mistakes |
| **Flexibility**   | Works with any LLM                  | Quality varies by model         |

## Implementation Approaches

### Approach 1: Simple Instruction

```python
def chain_of_thought(query):
    prompt = f"""
{query}

Let's approach this step by step:
"""
    return llm.generate(prompt)
```

**Pros:** Minimal code, works immediately
**Cons:** No control over steps, format varies

---

### Approach 2: Structured Steps

```python
def structured_cot(query):
    prompt = f"""
Solve this problem using these steps:

Step 1: Understand the problem (identify what's given and what we need to find)
Step 2: Break down the problem (what are the sub-problems?)
Step 3: Solve each part (show your work)
Step 4: Combine results (get final answer)
Step 5: Verify (does the answer make sense?)

Problem: {query}
"""
    return llm.generate(prompt)
```

**Pros:** Consistent format, easier to parse
**Cons:** May constrain LLM too much for some problems

---

### Approach 3: Self-Consistency

Run CoT multiple times, pick most common answer:

```python
def self_consistent_cot(query, n=5):
    answers = []

    for _ in range(n):
        response = chain_of_thought(query)
        answer = extract_final_answer(response)
        answers.append(answer)

    # Return most common answer
    return max(set(answers), key=answers.count)
```

**Pros:** Higher accuracy (reduces errors)
**Cons:** 5x cost and latency

---

### Approach 4: CoT + Tool Calling

Combine reasoning with tools:

```python
def cot_with_tools(query):
    prompt = f"""
Solve this problem step by step. When you need to calculate, use the calculator tool.

Tools available:
- calculate(expression): Evaluate math expressions

Problem: {query}

Let's think step by step:
"""

    response = llm.generate(prompt)

    # If tool call detected, execute and continue
    if "calculate(" in response:
        result = execute_calculation(response)
        response += f"\nCalculation result: {result}\n"
        response += llm.generate("Continue reasoning with this result:")

    return response
```

**Pros:** Combines reasoning with accurate computation
**Cons:** More complex orchestration

## Real-world Example: Technical Debugging

**Query:** "My React component isn't updating when state changes. Here's the code: [code snippet]"

**CoT Response:**

```
Let's debug this step by step:

Step 1: Check state initialization
- The state is initialized with useState() ✓
- Initial value looks correct ✓

Step 2: Examine state update
- setState is called in handleClick ✓
- But... the update uses the old state value directly ✗
- This is a closure issue!

Step 3: Identify the problem
- The handleClick function captures the state value at render time
- When called, it uses the stale value
- React doesn't re-render because the value doesn't actually change

Step 4: Solution
- Use functional state update: setState(prev => prev + 1)
- Or add state to dependency array if in useEffect

Step 5: Verify
- With functional update, each click will use the current state
- React will detect the change and re-render ✓

Root cause: Stale closure capturing old state value
Fix: Use setState(prev => prev + 1) instead of setState(state + 1)
```

**Value:** User sees the diagnostic process, learns debugging methodology.

## Cost Analysis

**Standard response:**

```
Question: What's 15% tip on $47.83? (15 tokens)
Answer: $7.17 (10 tokens)
Total: 25 tokens = $0.00025
```

**Chain-of-Thought response:**

```
Question: What's 15% tip on $47.83? (15 tokens)

CoT Answer:
Let's calculate step by step:
Step 1: Convert 15% to decimal: 15 / 100 = 0.15
Step 2: Multiply by bill amount: $47.83 × 0.15
Step 3: Calculate: 47.83 × 0.15 = 7.1745
Step 4: Round to cents: $7.17
Therefore, a 15% tip on $47.83 is $7.17

(120 tokens)

Total: 135 tokens = $0.00135
```

**Cost multiplier: 5.4x** for CoT vs direct answer

**When it's worth it:**

- User values the reasoning (educational, auditing)
- Problem is complex (accuracy boost justifies cost)
- Error cost is high (medical, financial, legal)

**When to skip:**

- High-volume, simple queries
- Users only want final answer
- Cost is primary constraint

## Common Pitfalls

### 1. Overly Verbose Reasoning

**Problem:** LLM generates 20 steps for a simple problem

```
Step 1: Understand the question
Step 2: Identify this is a math problem
Step 3: Note that we need addition
... [17 more obvious steps]
```

**Solution:** Prompt "Be concise. Only include non-obvious steps."

### 2. Incorrect But Confident Reasoning

**Problem:** Each step sounds logical but compounds errors

```
Step 1: 2 + 2 = 5 (incorrect)
Step 2: 5 × 3 = 15 (correct math, wrong input)
Step 3: Therefore 15 is the answer (confidently wrong)
```

**Solution:**

- Self-consistency (multiple attempts)
- Tool calling for calculations
- Verification step at end

### 3. Breaking Out of Format

**Problem:** LLM stops following step structure

```
Step 1: First I'll...
Step 2: Then I need...
Oh wait, I should actually just...
[abandons format]
```

**Solution:** Structured prompts with clear format enforcement

### 4. Circular Reasoning

**Problem:** Steps don't progress toward answer

```
Step 1: I need to find X
Step 2: To find X, I need Y
Step 3: To find Y, I need X  ← Loop
```

**Solution:** Include "sanity check" instruction

### 5. Can't Extract Final Answer

**Problem:** Reasoning is clear but no explicit answer

```
Step 1: ...
Step 2: ...
Step 3: So we can see that...
[ends without stating final answer]
```

**Solution:** Always prompt "State final answer clearly at end"

## Advanced Patterns

### Least-to-Most Prompting

Break problem into progressively complex sub-problems:

```
Problem: Calculate compound interest on $1000 at 5% for 3 years

Approach:
1. First, what's simple interest for 1 year? (easiest)
2. Now, what's the amount after 1 year with compound interest?
3. Repeat for year 2
4. Repeat for year 3
5. What's the total interest earned? (final answer)
```

### Self-Ask Prompting

LLM generates and answers its own sub-questions:

```
Q: What's the population of the capital of France?

Self-Ask:
Sub-Q1: What is the capital of France?
A1: Paris

Sub-Q2: What is the population of Paris?
A2: Approximately 2.1 million

Final Answer: 2.1 million
```

### Tree-of-Thought

Explore multiple reasoning paths:

```
Problem: Find best route from A to B

Branch 1: Highway route
- Step 1a: Check traffic
- Step 1b: 45 min expected
- Evaluation: Fast but toll cost

Branch 2: City streets
- Step 2a: Check stoplights
- Step 2b: 60 min expected
- Evaluation: Free but slower

Branch 3: Combined route
- Step 3a: Highway then local
- Step 3b: 50 min expected
- Evaluation: Balanced

Best: Branch 1 (highway) if time-sensitive
```

## Testing & Validation

### Benchmarks

Test CoT on standard datasets:

- **GSM8K** - Grade school math problems
- **MMLU** - Multi-task reasoning
- **BBH** - Big-Bench Hard reasoning tasks

### Success Metrics

- **Accuracy** - % of correct final answers
- **Step validity** - Are intermediate steps logical?
- **Completeness** - Does reasoning cover all aspects?
- **Conciseness** - Not overly verbose?

### Evaluation Criteria

```python
def evaluate_cot_response(response, expected_answer):
    return {
        "correct_answer": extract_answer(response) == expected_answer,
        "num_steps": count_steps(response),
        "has_verification": "verify" in response.lower(),
        "token_count": len(tokenize(response)),
        "clarity_score": rate_clarity(response)  # Human eval
    }
```

## Production Considerations

### Caching Reasoning Paths

For common questions, cache CoT responses:

```python
cot_cache = {
    "What's 15% tip on $50": "Step 1: 50 × 0.15...",
    "Square root of 144": "Step 1: Find number that..."
}

def cot_with_cache(query):
    if query in cot_cache:
        return cot_cache[query]

    response = chain_of_thought(query)
    cot_cache[query] = response
    return response
```

### Progressive Disclosure

Show answer first, reasoning on demand:

```python
response = chain_of_thought(query)
answer = extract_final_answer(response)
reasoning = extract_steps(response)

return {
    "answer": answer,  # Show immediately
    "reasoning": reasoning,  # Expandable section
    "show_work": False  # User can toggle
}
```

### Adaptive CoT

Use CoT only when needed:

```python
def adaptive_cot(query):
    # Classify query complexity
    complexity = classify_complexity(query)

    if complexity == "simple":
        return llm.generate(query)  # Direct answer
    elif complexity == "medium":
        return chain_of_thought(query)  # CoT
    else:  # complex
        return self_consistent_cot(query)  # Multiple attempts
```

### Monitoring

Track reasoning quality:

```json
{
  "query": "Calculate compound interest...",
  "answer": "$1157.63",
  "num_steps": 4,
  "tokens": 180,
  "cost": 0.0018,
  "correct": true,
  "user_rating": 5,
  "reasoning_helped": true
}
```

## Comparison: CoT vs Other Patterns

| Pattern              | Reasoning Visible | Tool Use | Iterations | Best For          |
| -------------------- | ----------------- | -------- | ---------- | ----------------- |
| **Direct Answer**    | No                | No       | 1          | Simple queries    |
| **Chain-of-Thought** | Yes               | Optional | 1          | Complex reasoning |
| **ReAct**            | Yes               | Yes      | Multiple   | Exploration tasks |
| **Tool Calling**     | No                | Yes      | 1          | Data retrieval    |

**When to combine:**

- **CoT + Tool Calling** - Reasoning about when/how to use tools
- **CoT + ReAct** - Each ReAct iteration uses CoT for better decisions
- **CoT + Router** - Reason about which route to take

## References

- **Chain-of-Thought Paper:** [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) - Original research
- **Zero-Shot CoT:** [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) - "Let's think step by step"
- **Self-Consistency:** [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) - Multiple paths
- **Tree-of-Thought:** [arXiv:2305.10601](https://arxiv.org/abs/2305.10601) - Branching reasoning

## Next Steps

- **Need external data?** → Combine with [Tool Calling](./tool-calling.md)
- **Need exploration?** → See [ReAct Pattern](./react-reasoning-acting.md)
- **Predefined workflow?** → See [Sequential Chain](./sequential-chain.md)
- **Production deployment?** → See [Cost Optimization](../production/cost-optimization.md)
