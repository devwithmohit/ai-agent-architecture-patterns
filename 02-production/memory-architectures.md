# Memory Architectures for AI Agents

> Giving agents the ability to remember - from conversation history to long-term knowledge

## When to Use Memory

**Perfect for:**

- Multi-turn conversations requiring context
- Personalized experiences based on past interactions
- Agents that learn from user preferences
- Long-running tasks spanning multiple sessions
- Knowledge accumulation over time

**Ideal scenarios:**

- Personal assistants that remember your preferences
- Customer support bots that recall previous tickets
- Code assistants that learn your coding style
- Research agents that build knowledge graphs
- Tutoring systems that track learning progress

## When NOT to Use Memory

**❌ Avoid when:**

- **Stateless is sufficient** - One-off queries don't need history
- **Privacy concerns** - User data retention is problematic
- **Cost constraints** - Memory storage and retrieval adds expense
- **Simple tasks** - Calculator or weather lookup doesn't need memory
- **Regulatory compliance** - GDPR/CCPA may prohibit persistent storage

**Cost trap:** Infinite memory growth = infinite storage costs. Without pruning, costs compound over time.

## Types of Memory

### 1. Short-Term Memory (Working Memory)

**What it is:** Current conversation/task context held in the LLM prompt.

**Characteristics:**

- Lives only during current session
- Limited by context window (4K-200K tokens)
- Fast access (no external lookup)
- Cleared when session ends

**Implementation:**

```python
class ShortTermMemory:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens

    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._truncate_if_needed()

    def get_context(self):
        return self.messages

    def _truncate_if_needed(self):
        # Keep only recent messages if exceeding limit
        total_tokens = sum(len(m["content"]) // 4 for m in self.messages)

        while total_tokens > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)  # Remove oldest
            total_tokens = sum(len(m["content"]) // 4 for m in self.messages)
```

**Use cases:**

- Chatbot conversations
- Current task execution context
- Recent tool call results

---

### 2. Long-Term Memory (Persistent)

**What it is:** Information stored externally and retrieved when relevant.

**Characteristics:**

- Survives across sessions
- Unlimited capacity (external database)
- Requires retrieval mechanism
- Higher latency than short-term

**Storage options:**

- Key-value stores (Redis, DynamoDB)
- SQL databases (PostgreSQL, MySQL)
- Document stores (MongoDB)
- Vector databases (Pinecone, Weaviate, Chroma)

**Implementation:**

```python
class LongTermMemory:
    def __init__(self, db_connection):
        self.db = db_connection

    def store(self, user_id, key, value, metadata=None):
        """Store a memory with optional metadata"""
        self.db.insert({
            "user_id": user_id,
            "key": key,
            "value": value,
            "metadata": metadata or {},
            "timestamp": datetime.now(),
            "access_count": 0
        })

    def retrieve(self, user_id, key):
        """Retrieve specific memory"""
        memory = self.db.query({
            "user_id": user_id,
            "key": key
        }).first()

        if memory:
            # Track access for importance scoring
            self.db.update(memory.id, {"access_count": memory.access_count + 1})

        return memory["value"] if memory else None

    def search(self, user_id, query, limit=5):
        """Search memories by query"""
        results = self.db.query({
            "user_id": user_id,
            "value": {"$regex": query, "$options": "i"}
        }).limit(limit)

        return [r["value"] for r in results]
```

**Use cases:**

- User preferences (theme, language, notification settings)
- Conversation history across sessions
- Facts learned about the user
- Past task outcomes

---

### 3. Semantic Memory (Vector Store)

**What it is:** Embeddings-based retrieval for similarity search.

**Characteristics:**

- Retrieves based on semantic similarity, not keywords
- Requires embedding model
- Fast approximate nearest neighbor search
- Handles large knowledge bases efficiently

**How it works:**

```
1. Convert text to embedding vector
2. Store vector in vector database
3. Query with new text → find similar vectors
4. Return associated text/metadata
```

**Implementation:**

```python
from openai import OpenAI
import chromadb

class SemanticMemory:
    def __init__(self):
        self.client = OpenAI()
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection("memories")

    def embed(self, text):
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def store(self, user_id, text, metadata=None):
        """Store memory with semantic embedding"""
        embedding = self.embed(text)

        self.collection.add(
            ids=[f"{user_id}_{datetime.now().timestamp()}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "user_id": user_id,
                **(metadata or {})
            }]
        )

    def retrieve(self, user_id, query, n_results=5):
        """Retrieve semantically similar memories"""
        query_embedding = self.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id}
        )

        return results["documents"][0] if results["documents"] else []
```

**Use cases:**

- RAG (Retrieval-Augmented Generation)
- Knowledge base search
- Finding relevant past conversations
- Document Q&A systems

---

### 4. Episodic Memory (Event-Based)

**What it is:** Chronological record of specific events or interactions.

**Characteristics:**

- Time-ordered sequence
- Rich contextual metadata
- Can replay "what happened when"
- Supports temporal queries

**Implementation:**

```python
class EpisodicMemory:
    def __init__(self, db):
        self.db = db

    def record_event(self, user_id, event_type, data):
        """Record a timestamped event"""
        self.db.insert({
            "user_id": user_id,
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(),
            "session_id": get_current_session_id()
        })

    def get_timeline(self, user_id, start_date, end_date):
        """Get events in time range"""
        return self.db.query({
            "user_id": user_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp")

    def get_session(self, session_id):
        """Get all events from a specific session"""
        return self.db.query({
            "session_id": session_id
        }).sort("timestamp")
```

**Use cases:**

- "What did we discuss last Tuesday?"
- Debugging agent behavior
- Audit trails
- User activity timelines

---

## Memory Architecture Patterns

### Pattern 1: Hybrid Memory System

Combine short-term + long-term:

```python
class HybridMemory:
    def __init__(self):
        self.short_term = ShortTermMemory(max_tokens=4000)
        self.long_term = LongTermMemory(db)
        self.semantic = SemanticMemory()

    def add_message(self, user_id, role, content):
        # Always add to short-term (current context)
        self.short_term.add(role, content)

        # Selectively store important messages long-term
        if self._is_important(content):
            self.long_term.store(user_id, f"{role}_{content}", content)
            self.semantic.store(user_id, content, {"role": role})

    def get_context(self, user_id, query=None):
        # Start with short-term memory
        context = self.short_term.get_context()

        # Augment with relevant long-term memories
        if query:
            relevant_memories = self.semantic.retrieve(user_id, query, n_results=3)
            context = self._merge_context(context, relevant_memories)

        return context

    def _is_important(self, content):
        # Simple heuristic: longer messages or questions
        return len(content) > 50 or "?" in content
```

---

### Pattern 2: Hierarchical Memory

Organize by levels of abstraction:

```
Level 1 (Raw): Individual messages
Level 2 (Summaries): Conversation summaries
Level 3 (Facts): Extracted key facts
Level 4 (Knowledge): Structured knowledge graph
```

```python
class HierarchicalMemory:
    def process_conversation(self, messages):
        # Level 1: Store raw messages
        for msg in messages:
            self.store_raw(msg)

        # Level 2: Generate summary
        summary = self.llm.generate(f"Summarize: {messages}")
        self.store_summary(summary)

        # Level 3: Extract facts
        facts = self.llm.generate(f"Extract key facts from: {summary}")
        self.store_facts(facts)

        # Level 4: Update knowledge graph
        self.update_knowledge_graph(facts)
```

---

### Pattern 3: Memory Consolidation

Compress old memories to save space:

```python
def consolidate_old_memories(user_id, older_than_days=30):
    """
    Consolidate old detailed memories into summaries
    """
    cutoff_date = datetime.now() - timedelta(days=older_than_days)
    old_memories = db.query({
        "user_id": user_id,
        "timestamp": {"$lt": cutoff_date},
        "consolidated": False
    })

    # Group by week
    for week_memories in group_by_week(old_memories):
        # Summarize the week
        summary = llm.generate(
            f"Summarize these interactions:\n{week_memories}"
        )

        # Replace detailed memories with summary
        db.insert({
            "user_id": user_id,
            "type": "weekly_summary",
            "content": summary,
            "original_count": len(week_memories),
            "timestamp": week_memories[0].timestamp
        })

        # Mark originals as consolidated (or delete)
        db.update_many(
            {"id": {"$in": [m.id for m in week_memories]}},
            {"consolidated": True}
        )
```

---

## Tradeoffs Table

| Memory Type           | Retrieval Speed | Capacity                 | Accuracy    | Cost | Use Case             |
| --------------------- | --------------- | ------------------------ | ----------- | ---- | -------------------- |
| **Short-term**        | Instant         | Limited (context window) | Perfect     | $    | Current conversation |
| **Long-term (SQL)**   | Fast            | Large                    | Exact match | $$   | User preferences     |
| **Semantic (Vector)** | Fast            | Very large               | Approximate | $$$  | Knowledge search     |
| **Episodic**          | Medium          | Large                    | Perfect     | $$   | Event timelines      |

---

## Cost Analysis

**Scenario:** Chatbot with 1000 daily active users, avg 20 messages/day

### Short-Term Memory

- **Storage:** None (in prompt only)
- **Cost:** Included in LLM token cost
- **Total:** $0

### Long-Term Memory (PostgreSQL)

- **Storage:** 1000 users × 20 msgs/day × 500 bytes = 10 MB/day = 300 MB/month
- **Cost:** ~$0.23/month (AWS RDS)
- **Queries:** ~100K/month = ~$0.10
- **Total:** ~$0.33/month

### Semantic Memory (Pinecone)

- **Embeddings:** 20K msgs/day × $0.0001/1K tokens × 100 tokens/msg = $0.20/day = $6/month
- **Storage:** 1M vectors × $0.096/1M/month = $0.096/month
- **Queries:** 20K/day × $0.20/1M queries = $0.004/day = $0.12/month
- **Total:** ~$6.22/month

**Grand total:** ~$6.55/month for full memory system

**Optimization:**

- Only embed important messages (reduce by 80%) → $1.50/month
- Use cheaper embedding models
- Implement memory pruning

---

## Common Pitfalls

### 1. Infinite Memory Growth

**Problem:** Never delete old memories

```python
# ❌ Unbounded growth
for message in user_messages:
    memory.store(message)  # Forever
```

**Solution:** Implement retention policies

```python
# ✅ Auto-prune old memories
def prune_old_memories(user_id, max_age_days=90):
    cutoff = datetime.now() - timedelta(days=max_age_days)
    db.delete_many({
        "user_id": user_id,
        "timestamp": {"$lt": cutoff},
        "importance": {"$lt": 0.5}  # Keep important ones
    })
```

### 2. No Memory Prioritization

**Problem:** Retrieve irrelevant memories

```python
# ❌ Returns random old memories
memories = db.query({"user_id": user_id}).limit(5)
```

**Solution:** Score by relevance + recency

```python
# ✅ Prioritize relevant + recent
def get_relevant_memories(user_id, query):
    # Semantic similarity
    semantic_matches = semantic_memory.retrieve(user_id, query)

    # Recent memories
    recent = db.query({
        "user_id": user_id,
        "timestamp": {"$gte": datetime.now() - timedelta(days=7)}
    })

    # Combine with scoring
    scored = []
    for memory in semantic_matches + recent:
        score = (
            memory.similarity_score * 0.6 +
            memory.recency_score * 0.3 +
            memory.access_count * 0.1
        )
        scored.append((score, memory))

    return [m for _, m in sorted(scored, reverse=True)[:5]]
```

### 3. Context Window Overflow

**Problem:** Too many memories → exceed context limit

```python
# ❌ Blindly include all memories
context = short_term.get() + long_term.get_all()
# 50K tokens → exceeds 32K limit!
```

**Solution:** Budget token allocation

```python
# ✅ Token-aware context building
def build_context(user_id, query, max_tokens=4000):
    # Reserve tokens for each component
    system_tokens = 200
    query_tokens = len(query) // 4
    response_buffer = 1000

    available_for_memory = max_tokens - system_tokens - query_tokens - response_buffer

    # Get memories in priority order
    memories = get_relevant_memories(user_id, query)

    # Add until token budget exhausted
    context = []
    tokens_used = 0
    for memory in memories:
        memory_tokens = len(memory.content) // 4
        if tokens_used + memory_tokens <= available_for_memory:
            context.append(memory)
            tokens_used += memory_tokens
        else:
            break

    return context
```

### 4. Privacy Leakage

**Problem:** Storing sensitive data indefinitely

```python
# ❌ Storing credit card numbers, passwords
memory.store(user_id, "password", user_input)
```

**Solution:** Filter sensitive data

```python
# ✅ PII detection and redaction
def safe_store(user_id, content):
    # Detect sensitive patterns
    if contains_pii(content):
        content = redact_pii(content)

    if is_sensitive_category(content):
        # Don't store at all or encrypt
        content = encrypt(content)

    memory.store(user_id, content)

def contains_pii(text):
    patterns = [
        r'\d{3}-\d{2}-\d{4}',  # SSN
        r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}',  # Credit card
        r'password|passwd|pwd'  # Password keywords
    ]
    return any(re.search(pattern, text, re.I) for pattern in patterns)
```

### 5. Stale Memory Problem

**Problem:** Outdated facts persist

```python
memory.store("user_preference", "User likes Python")
# 6 months later, user now prefers Rust
# But old preference still retrieved
```

**Solution:** Version facts or timestamp + TTL

```python
def store_fact(user_id, fact_type, value):
    # Invalidate old version
    db.update_many(
        {"user_id": user_id, "fact_type": fact_type},
        {"valid": False}
    )

    # Store new version
    db.insert({
        "user_id": user_id,
        "fact_type": fact_type,
        "value": value,
        "valid": True,
        "version": get_next_version(),
        "timestamp": datetime.now()
    })
```

### 6. No Memory Verification

**Problem:** False memories accumulate

```python
# User says something, agent stores as fact
user: "I think Paris is in Germany"
agent: memory.store("Paris is in Germany")  # ❌ Storing misinformation
```

**Solution:** Fact verification before storage

```python
def store_with_verification(user_id, claim):
    # Verify against knowledge base
    verified = fact_check_api.verify(claim)

    if verified.confidence > 0.8:
        memory.store(user_id, claim, metadata={"verified": True})
    else:
        # Store with low confidence flag
        memory.store(user_id, claim, metadata={
            "verified": False,
            "confidence": verified.confidence
        })
```

---

## Advanced Patterns

### Memory Reflection

Agent periodically reviews and consolidates memories:

```python
def reflect_on_memories(user_id):
    """
    Periodic reflection to extract higher-level insights
    """
    recent_memories = db.query({
        "user_id": user_id,
        "timestamp": {"$gte": datetime.now() - timedelta(days=7)}
    })

    # Generate insights
    insights = llm.generate(f"""
    Analyze these interactions and extract key insights about the user:

    {recent_memories}

    What patterns do you notice?
    What are their preferences?
    What goals are they trying to achieve?
    """)

    # Store insights as meta-memory
    memory.store(user_id, "weekly_insights", insights, metadata={
        "type": "reflection",
        "timestamp": datetime.now()
    })
```

### Forgetting Mechanism

Mimic human memory decay:

```python
def calculate_memory_strength(memory):
    """
    Memory strength decays over time but reinforced by access
    """
    age_days = (datetime.now() - memory.timestamp).days

    # Decay factor
    decay = math.exp(-age_days / 30)  # Half-life of 30 days

    # Reinforcement from access
    reinforcement = 1 + (memory.access_count * 0.1)

    # Importance boost
    importance_boost = memory.metadata.get("importance", 0.5)

    strength = decay * reinforcement * importance_boost

    return strength

def prune_weak_memories(user_id, threshold=0.1):
    """Remove memories below strength threshold"""
    memories = db.query({"user_id": user_id})

    for memory in memories:
        if calculate_memory_strength(memory) < threshold:
            db.delete(memory.id)
```

---

## Production Considerations

### Memory Indexing

```python
# Create indexes for fast retrieval
db.create_index([
    ("user_id", 1),
    ("timestamp", -1)
])

db.create_index([
    ("user_id", 1),
    ("fact_type", 1),
    ("valid", 1)
])
```

### Memory Backup & Recovery

```python
def backup_user_memories(user_id):
    memories = db.query({"user_id": user_id})
    backup_data = {
        "user_id": user_id,
        "backup_date": datetime.now().isoformat(),
        "memories": [m.to_dict() for m in memories]
    }

    # Store in object storage (S3, GCS)
    storage.upload(
        f"memory_backups/{user_id}/{datetime.now().date()}.json",
        json.dumps(backup_data)
    )
```

### Memory Analytics

```python
def analyze_memory_usage(user_id):
    return {
        "total_memories": db.count({"user_id": user_id}),
        "storage_mb": calculate_storage_size(user_id),
        "avg_access_count": db.aggregate([
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": None, "avg": {"$avg": "$access_count"}}}
        ]),
        "oldest_memory": db.query({"user_id": user_id}).sort("timestamp").first(),
        "most_accessed": db.query({"user_id": user_id}).sort("access_count", -1).limit(10)
    }
```

---

## Testing Memory Systems

```python
def test_memory_retrieval():
    memory = HybridMemory()
    user_id = "test_user"

    # Store test memories
    memory.add_message(user_id, "user", "I love Python programming")
    memory.add_message(user_id, "user", "My favorite color is blue")

    # Test retrieval
    context = memory.get_context(user_id, "What language do I like?")

    assert "Python" in str(context)
    assert len(context) > 0

def test_memory_consolidation():
    # Create old memories
    for i in range(100):
        memory.store(user_id, f"message_{i}", timestamp=datetime.now() - timedelta(days=60))

    # Consolidate
    consolidate_old_memories(user_id, older_than_days=30)

    # Verify consolidation
    remaining = db.count({"user_id": user_id, "consolidated": False})
    summaries = db.count({"user_id": user_id, "type": "weekly_summary"})

    assert remaining < 100
    assert summaries > 0
```

---

## References

- **Vector Databases Comparison:** [Benchmark](https://github.com/erikbern/ann-benchmarks)
- **MemGPT:** [Paper](https://arxiv.org/abs/2310.08560) - OS-inspired memory management
- **Memory Networks:** [Paper](https://arxiv.org/abs/1410.3916) - Neural memory architectures
- **Pinecone Docs:** [Vector DB Guide](https://docs.pinecone.io/)
- **Chroma:** [Open-source vector DB](https://www.trychroma.com/)

---

## Next Steps

- **Need error handling?** → See [Error Handling](./error-handling.md)
- **Cost concerns?** → See [Cost Optimization](./cost-optimization.md)
- **Security?** → See [Security](./security.md)
- **Testing?** → See [Testing Strategies](./testing-strategies.md)
