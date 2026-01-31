# Error Handling for AI Agents

> Production-grade strategies for when things go wrong - because they will

## Why Error Handling Matters

**In production, failures are inevitable:**

- LLM APIs timeout or rate limit
- Tools return unexpected formats
- Network calls fail
- Parsing errors from malformed responses
- External services go down

**Without proper error handling:**

- Users see cryptic error messages
- Partial work is lost
- Costs balloon from infinite retries
- System becomes unreliable

---

## Error Categories

### 1. Transient Errors (Retry-able)

- **Rate limits** - 429 errors from API quotas
- **Timeouts** - Request exceeds deadline
- **Network issues** - Connection drops
- **Service overload** - 503 Service Unavailable

**Strategy:** Retry with exponential backoff

### 2. Permanent Errors (Non-retry-able)

- **Authentication failures** - Invalid API key
- **Invalid input** - Malformed request
- **Not found** - 404 errors
- **Forbidden** - 403 permission denied

**Strategy:** Fail fast, return error to user

### 3. Logic Errors (Agent-level)

- **Hallucinations** - LLM invents tools/facts
- **Infinite loops** - Agent repeats same action
- **Invalid tool arguments** - Wrong parameter types
- **Contradictions** - Agent output conflicts with constraints

**Strategy:** Validation, guards, fallbacks

---

## Retry Strategies

### Pattern 1: Simple Retry

````python
def simple_retry(func, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
    ```

**Pros:** Simple
**Cons:** Can hammer failing service

---

### Pattern 2: Exponential Backoff

```python
import time
import random

def exponential_backoff_retry(func, max_attempts=5, base_delay=1):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            # Calculate delay: 1s, 2s, 4s, 8s, 16s
            delay = base_delay * (2 ** attempt)

            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter

            print(f"Attempt {attempt + 1} failed. Retrying in {total_delay:.2f}s")
            time.sleep(total_delay)
````

**Pros:** Respects rate limits, reduces server load
**Cons:** Increases latency

---

### Pattern 3: Selective Retry (by error type)

```python
from openai import RateLimitError, APITimeoutError, APIError

def selective_retry(func, max_attempts=3):
    """Only retry transient errors"""

    RETRYABLE_ERRORS = (RateLimitError, APITimeoutError, ConnectionError)

    for attempt in range(max_attempts):
        try:
            return func()
        except RETRYABLE_ERRORS as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Transient error: {e}. Retrying...")
            time.sleep(2 ** attempt)
        except APIError as e:
            # Permanent error - don't retry
            print(f"Permanent error: {e}")
            raise
```

**Pros:** Fast-fails on permanent errors
**Cons:** Requires error classification

---

### Pattern 4: Retry with Tenacity Library

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
    reraise=True
)
def call_llm_with_retry(prompt):
    return llm.generate(prompt)
```

**Pros:** Declarative, battle-tested
**Cons:** Extra dependency

---

## Fallback Strategies

### Pattern 1: Cascading Fallbacks

```python
def call_with_fallbacks(query):
    """Try primary, then fallback options"""

    try:
        # Try primary (best model)
        return gpt4.generate(query)
    except RateLimitError:
        try:
            # Fallback 1: Cheaper model
            return gpt35.generate(query)
        except RateLimitError:
            try:
                # Fallback 2: Different provider
                return claude.generate(query)
            except Exception:
                # Fallback 3: Cached response or error
                return get_cached_response(query) or "Service temporarily unavailable"
```

**Use case:** High availability required

---

### Pattern 2: Partial Results

```python
def process_with_partial_results(items):
    """Return successes even if some fail"""

    results = []
    errors = []

    for item in items:
        try:
            result = process_item(item)
            results.append(result)
        except Exception as e:
            errors.append({"item": item, "error": str(e)})

    return {
        "results": results,
        "errors": errors,
        "success_rate": len(results) / len(items)
    }
```

**Use case:** Batch processing

---

### Pattern 3: Default Values

```python
def get_user_preference(user_id, key, default=None):
    """Return default if retrieval fails"""

    try:
        return preferences_db.get(user_id, key)
    except DatabaseError as e:
        log.warning(f"DB error getting preference: {e}")
        return default

# Usage
theme = get_user_preference(user_id, "theme", default="light")
```

**Use case:** Non-critical features

---

## Circuit Breaker Pattern

**Problem:** Repeatedly calling a failing service wastes resources

**Solution:** Open circuit after N failures, auto-close after cooldown

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func):
        if self.state == "open":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                # Try to close circuit
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func()

            # Success - reset or close circuit
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                print(f"Circuit breaker opened after {self.failure_count} failures")

            raise

# Usage
weather_circuit = CircuitBreaker(failure_threshold=3, timeout=30)

def get_weather(city):
    return weather_circuit.call(lambda: weather_api.get(city))
```

---

## Graceful Degradation

### Pattern 1: Feature Toggles

```python
class FeatureFlags:
    def __init__(self):
        self.flags = {
            "use_expensive_model": True,
            "enable_web_search": True,
            "enable_image_generation": True
        }

    def is_enabled(self, feature):
        return self.flags.get(feature, False)

    def disable(self, feature):
        self.flags[feature] = False

# Usage
flags = FeatureFlags()

def generate_response(query):
    if flags.is_enabled("use_expensive_model"):
        try:
            return gpt4.generate(query)
        except RateLimitError:
            # Degrade to cheaper model
            flags.disable("use_expensive_model")
            return gpt35.generate(query)
    else:
        return gpt35.generate(query)
```

---

### Pattern 2: Progressive Enhancement

```python
def enhanced_search(query):
    """Start with basic search, add enhancements if possible"""

    # Core functionality (always works)
    results = basic_search(query)

    # Enhancement 1: Semantic ranking
    try:
        results = semantic_rank(results)
    except Exception as e:
        log.warning(f"Semantic ranking failed: {e}")
        # Continue with basic results

    # Enhancement 2: Personalization
    try:
        results = personalize(results, user_id)
    except Exception as e:
        log.warning(f"Personalization failed: {e}")
        # Continue with non-personalized results

    return results
```

---

## Validation & Guards

### Input Validation

```python
from pydantic import BaseModel, validator

class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict

    @validator('tool_name')
    def validate_tool_exists(cls, v):
        if v not in VALID_TOOLS:
            raise ValueError(f"Unknown tool: {v}")
        return v

    @validator('arguments')
    def validate_args(cls, v, values):
        tool_name = values.get('tool_name')
        required_args = TOOL_SCHEMAS[tool_name].get('required', [])

        for arg in required_args:
            if arg not in v:
                raise ValueError(f"Missing required argument: {arg}")

        return v

# Usage
try:
    request = ToolCallRequest(
        tool_name="search_web",
        arguments={"query": "AI agents"}
    )
    result = execute_tool(request.tool_name, request.arguments)
except ValueError as e:
    return {"error": f"Invalid request: {e}"}
```

---

### Output Validation

```python
def validate_llm_output(output, expected_schema):
    """Ensure LLM output matches expected format"""

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("Output is not valid JSON")

    # Check required fields
    for field in expected_schema.get('required', []):
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")

    # Check field types
    for field, expected_type in expected_schema.get('properties', {}).items():
        if field in parsed:
            actual_type = type(parsed[field]).__name__
            if actual_type != expected_type:
                raise ValueError(f"Field {field}: expected {expected_type}, got {actual_type}")

    return parsed

# Usage
schema = {
    "required": ["sentiment", "score"],
    "properties": {
        "sentiment": "str",
        "score": "float"
    }
}

output = llm.generate("Analyze sentiment: I love this!")
try:
    validated = validate_llm_output(output, schema)
except ValueError as e:
    # Retry with more explicit instructions
    output = llm.generate(f"Return JSON with sentiment and score: I love this!")
    validated = validate_llm_output(output, schema)
```

---

### Loop Detection

```python
class LoopDetector:
    def __init__(self, max_repeats=3):
        self.action_history = []
        self.max_repeats = max_repeats

    def check(self, action):
        """Detect if action is being repeated"""
        self.action_history.append(action)

        # Keep only recent history
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        # Count recent occurrences of this action
        recent_count = self.action_history[-self.max_repeats:].count(action)

        if recent_count >= self.max_repeats:
            raise InfiniteLoopError(f"Action '{action}' repeated {recent_count} times")

        return True

# Usage in ReAct loop
detector = LoopDetector(max_repeats=3)

for iteration in range(max_iterations):
    action = llm.decide_action()

    try:
        detector.check(action)
    except InfiniteLoopError:
        # Break loop, try different approach
        action = force_different_action()

    execute_action(action)
```

---

## Common Pitfalls

### 1. Swallowing Errors Silently

```python
# ❌ Bad: Error disappears
try:
    result = call_api()
except Exception:
    pass  # Silent failure

# ✅ Good: Log and handle
try:
    result = call_api()
except Exception as e:
    log.error(f"API call failed: {e}", exc_info=True)
    return fallback_value
```

### 2. Retry Amplification

```python
# ❌ Bad: Nested retries multiply
@retry(max_attempts=5)
def outer():
    return inner()  # Also retries 5 times = 25 total calls!

@retry(max_attempts=5)
def inner():
    return api_call()

# ✅ Good: Only retry at one level
def outer():
    return inner()

@retry(max_attempts=5)
def inner():
    return api_call()
```

### 3. No Timeout on Retries

```python
# ❌ Bad: Could retry forever
@retry(wait_exponential(max=300))  # 5 min max wait
def call_api():
    ...

# ✅ Good: Set max total time
@retry(
    wait_exponential(max=10),
    stop=stop_after_delay(60)  # Max 1 minute total
)
def call_api():
    ...
```

### 4. Not Distinguishing Error Types

```python
# ❌ Bad: Retries everything
try:
    result = api_call()
except Exception:
    retry()  # Even for 404s, auth errors, etc

# ✅ Good: Selective retry
try:
    result = api_call()
except RateLimitError:
    retry_with_backoff()
except (TimeoutError, ConnectionError):
    retry_immediately()
except AuthenticationError:
    raise  # Don't retry auth errors
```

### 5. Losing Context on Error

```python
# ❌ Bad: User gets generic error
return "Something went wrong"

# ✅ Good: Helpful error message
return f"Unable to search web due to rate limit. Please try again in {retry_after}s"
```

---

## Production Monitoring

### Error Rate Tracking

```python
from collections import defaultdict

class ErrorTracker:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.total_calls = 0

    def record_call(self, success, error_type=None):
        self.total_calls += 1
        if not success:
            self.error_counts[error_type] += 1

    def get_error_rate(self):
        if self.total_calls == 0:
            return 0
        return sum(self.error_counts.values()) / self.total_calls

    def get_stats(self):
        return {
            "total_calls": self.total_calls,
            "error_rate": self.get_error_rate(),
            "errors_by_type": dict(self.error_counts)
        }

# Usage
tracker = ErrorTracker()

try:
    result = api_call()
    tracker.record_call(success=True)
except RateLimitError:
    tracker.record_call(success=False, error_type="rate_limit")
except TimeoutError:
    tracker.record_call(success=False, error_type="timeout")
```

### Alerting

```python
def check_and_alert():
    stats = tracker.get_stats()

    if stats["error_rate"] > 0.1:  # 10% error rate
        send_alert(
            severity="high",
            message=f"Error rate: {stats['error_rate']:.1%}",
            details=stats["errors_by_type"]
        )
```

---

## Testing Error Handling

```python
import pytest
from unittest.mock import patch, Mock

def test_retry_on_rate_limit():
    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        RateLimitError(),
        RateLimitError(),
        "Success!"
    ]

    result = exponential_backoff_retry(mock_llm.generate)

    assert result == "Success!"
    assert mock_llm.generate.call_count == 3

def test_circuit_breaker_opens():
    breaker = CircuitBreaker(failure_threshold=3)
    failing_func = Mock(side_effect=Exception("Fail"))

    # Trigger failures
    for _ in range(3):
        with pytest.raises(Exception):
            breaker.call(failing_func)

    # Circuit should be open
    assert breaker.state == "open"

    # Next call should fail fast
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        breaker.call(failing_func)

def test_fallback_chain():
    primary = Mock(side_effect=RateLimitError())
    fallback = Mock(return_value="Fallback result")

    result = call_with_fallbacks_mock(primary, fallback)

    assert result == "Fallback result"
    assert primary.call_count == 1
    assert fallback.call_count == 1
```

---

## References

- **Tenacity Library:** [GitHub](https://github.com/jd/tenacity)
- **Circuit Breaker Pattern:** [Martin Fowler](https://martinfowler.com/bliki/CircuitBreaker.html)
- **AWS Exponential Backoff:** [Docs](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- **Google SRE Book:** [Handling Overload](https://sre.google/sre-book/handling-overload/)

---

## Next Steps

- **Need observability?** → See [Observability](./observability.md)
- **Cost concerns?** → See [Cost Optimization](./cost-optimization.md)
- **Rate limiting?** → See [Rate Limiting](./rate-limiting.md)
- **Security?** → See [Security](./security.md)
