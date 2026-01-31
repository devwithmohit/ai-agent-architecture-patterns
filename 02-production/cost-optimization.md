# Cost Optimization for AI Agents

> Making agents affordable at scale - from prototype to production economics

## Why Cost Optimization Matters

**AI agent costs can spiral quickly:**

- GPT-4: $0.03/1K input tokens, $0.06/1K output
- Average conversation: 5K-20K tokens = $0.15-$1.20
- 10K users × 10 conversations/day = $15K-$120K/month

**Without optimization:**

- Monthly bills that exceed revenue
- Unsustainable economics
- Can't scale to more users

---

## Cost Breakdown

### Typical Agent Cost Components

| Component      | % of Total | Optimization Potential          |
| -------------- | ---------- | ------------------------------- |
| LLM API calls  | 70-85%     | High (model selection, caching) |
| Embeddings     | 5-10%      | Medium (batch processing)       |
| Tool API calls | 5-15%      | Medium (caching, rate limits)   |
| Infrastructure | 5-10%      | Low (already optimized)         |

**Focus:** LLM costs offer biggest ROI

---

## Strategy 1: Model Selection

### Cost-Performance Tiers

| Model              | Input Cost  | Output Cost | Use Case                          |
| ------------------ | ----------- | ----------- | --------------------------------- |
| **GPT-4 Turbo**    | $0.01/1K    | $0.03/1K    | Complex reasoning, critical tasks |
| **GPT-3.5 Turbo**  | $0.0005/1K  | $0.0015/1K  | Simple tasks, high volume         |
| **Claude 3 Haiku** | $0.00025/1K | $0.00125/1K | Ultra-cheap, fast responses       |

**Savings:** GPT-3.5 is **20-60× cheaper** than GPT-4

### Smart Routing

```python
def route_to_model(task_complexity):
    """Use expensive models only when needed"""

    if task_complexity == "simple":
        return "gpt-3.5-turbo"  # $0.001/1K
    elif task_complexity == "medium":
        return "gpt-4"  # $0.03/1K
    else:  # complex
        return "gpt-4-turbo"  # $0.01/1K

def classify_complexity(query):
    """Fast classification"""
    if len(query) < 50 and "?" in query:
        return "simple"

    complex_keywords = ["analyze", "compare", "explain in detail"]
    if any(kw in query.lower() for kw in complex_keywords):
        return "complex"

    return "medium"

# Usage
query = "What's 2+2?"
model = route_to_model(classify_complexity(query))
# → Uses cheap model, saves 95%
```

**Savings example:**

- 70% of queries are simple → GPT-3.5
- 25% medium → GPT-4
- 5% complex → GPT-4 Turbo

**Before:** 100% GPT-4 = $3,000/month
**After:** Mixed routing = $600/month
**Savings: 80%**

---

## Strategy 2: Caching

### Response Caching

```python
import hashlib
import redis

cache = redis.Redis()

def cached_llm_call(prompt, model="gpt-4", ttl=3600):
    """Cache LLM responses"""

    # Create cache key
    cache_key = hashlib.sha256(
        f"{model}:{prompt}".encode()
    ).hexdigest()

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)

    # Call LLM
    response = llm.generate(prompt, model=model)

    # Store in cache
    cache.setex(cache_key, ttl, json.dumps(response))

    return response
```

**Savings:**

- 30% cache hit rate = 30% cost reduction
- Common queries ("What is AI?") cached for hours

### Semantic Caching

Cache by meaning, not exact match:

```python
def semantic_cache_lookup(query, threshold=0.95):
    """Find semantically similar cached queries"""

    query_embedding = embed(query)

    # Search vector DB for similar queries
    similar = vector_db.search(
        query_embedding,
        limit=1,
        threshold=threshold
    )

    if similar:
        return cache.get(similar[0].cache_key)

    return None

# Usage
# "What's the weather in NYC?" and
# "NYC weather?" → Same cache hit
```

---

## Strategy 3: Prompt Optimization

### Token Reduction

```python
# ❌ Verbose prompt (250 tokens)
prompt = """
You are a helpful assistant. Please help the user with their question.
Be polite and professional. Think step by step before answering.

User question: What's the capital of France?

Please provide a detailed and well-structured answer.
"""

# ✅ Concise prompt (20 tokens)
prompt = "Capital of France?"

# Savings: 92% fewer tokens
```

**Techniques:**

1. **Remove filler words** ("please", "thank you")
2. **Use abbreviations** where clear
3. **Avoid repetition** in system prompt
4. **JSON mode** instead of prose for structured output

### Few-Shot Optimization

```python
# ❌ 5 examples (1000 tokens)
examples = [ex1, ex2, ex3, ex4, ex5]

# ✅ 2 examples (400 tokens)
examples = [ex1, ex2]  # Often sufficient

# Test to find minimum effective examples
```

---

## Strategy 4: Output Control

### Limit Max Tokens

```python
# ❌ Unlimited output
response = llm.generate(prompt)

# ✅ Set max tokens
response = llm.generate(prompt, max_tokens=150)

# Prevents rambling, controls cost
```

### Stop Sequences

```python
# Stop generation early
response = llm.generate(
    prompt,
    stop=["###", "\n\n"]  # Stop at markers
)
```

---

## Strategy 5: Batching

### Batch API Calls

```python
# ❌ Individual calls
for item in items:
    result = llm.generate(f"Process: {item}")
    # 100 items = 100 API calls

# ✅ Batch processing
batch_prompt = "Process these items:\n" + "\n".join(items)
results = llm.generate(batch_prompt)
# 100 items = 1 API call (if fits in context)

# Savings: 50% (batch discount + fewer calls)
```

### Embedding Batching

```python
# Embed multiple texts at once
embeddings = openai.Embedding.create(
    model="text-embedding-3-small",
    input=[text1, text2, ..., text100]  # Batch of 100
)

# Faster + cheaper than 100 individual calls
```

---

## Strategy 6: Streaming vs Batch

### When to Stream

```python
# User-facing: Stream for perceived speed
def stream_response(prompt):
    for chunk in llm.stream(prompt):
        yield chunk  # User sees progress

# Cost: Same as non-streaming
# UX: Much better (feels faster)
```

### When to Batch

```python
# Background processing: Batch for efficiency
def process_feedback_batch(feedback_list):
    # Wait for 100 items or 5 minutes
    if len(queue) >= 100 or time_elapsed > 300:
        batch_process(queue)
```

---

## Strategy 7: Cheaper Alternatives

### Self-Hosted Models

```python
# Cloud API: $0.03/1K tokens
response = openai.chat.completions.create(...)

# Self-hosted (e.g., Llama 2 on AWS): $0.002/1K tokens
response = local_llm.generate(...)

# Savings: 93% for high-volume workloads
```

**Trade-offs:**

- Upfront infrastructure cost
- Operational complexity
- May have lower quality

**Break-even:** ~$500/month in API costs

---

## Cost Monitoring

### Real-Time Tracking

```python
class CostMonitor:
    def __init__(self):
        self.total_cost = 0
        self.call_count = 0

    def track_call(self, model, input_tokens, output_tokens):
        cost = calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        self.call_count += 1

        # Alert if threshold exceeded
        if self.total_cost > DAILY_BUDGET:
            alert("Daily budget exceeded!")
            raise BudgetExceededError()

    def get_metrics(self):
        return {
            "total_cost": self.total_cost,
            "calls": self.call_count,
            "avg_cost_per_call": self.total_cost / self.call_count if self.call_count else 0
        }
```

### Budget Limits

```python
def enforce_user_budget(user_id, estimated_cost):
    user_spent = cost_tracker.get_user_cost(user_id)

    if user_spent + estimated_cost > USER_MONTHLY_LIMIT:
        return {"error": "Monthly budget exceeded. Upgrade plan or wait until next month."}

    # Allow request
    return proceed_with_request()
```

---

## Example Optimizations

### Before vs After

**Scenario:** Customer support chatbot, 10K daily conversations

**Before optimization:**

- Model: GPT-4 for everything
- Avg tokens per conversation: 5,000
- Cache hit rate: 0%
- Cost: $15,000/month

**After optimization:**

- Model: GPT-3.5 for 80% of queries, GPT-4 for complex only
- Avg tokens: 3,000 (prompt optimization)
- Cache hit rate: 40%
- Batched embedding generation
- Cost: $2,400/month

**Savings: 84% ($12,600/month)**

---

## Common Pitfalls

### 1. Premature Optimization

**Don't:** Optimize before product-market fit
**Do:** Get it working first, optimize when costs hurt

### 2. Over-Caching

**Don't:** Cache everything forever
**Do:** Set appropriate TTLs, invalidate stale data

### 3. Sacrificing Quality

**Don't:** Use GPT-3.5 for tasks that need GPT-4
**Do:** A/B test quality vs cost trade-offs

### 4. Ignoring Hidden Costs

**Don't:** Only track LLM costs
**Do:** Include embeddings, tools, infrastructure

---

## References

- **OpenAI Pricing:** [Pricing Page](https://openai.com/pricing)
- **Token Counting:** [tiktoken](https://github.com/openai/tiktoken)
- **Cost Calculators:** [OpenAI Tokenizer](https://platform.openai.com/tokenizer)

---

## Next Steps

- **Need monitoring?** → See [Observability](./observability.md)
- **Rate limits?** → See [Rate Limiting](./rate-limiting.md)
- **Testing costs?** → See [Testing Strategies](./testing-strategies.md)
