# Rate Limiting for AI Agents

> Handling API quotas and preventing system overload

## Why Rate Limiting Matters

**Without rate limiting:**

- Hit API quotas → 429 errors
- Overwhelm downstream services
- Unpredictable costs
- Poor user experience during spikes

**With rate limiting:**

- Stay within quotas
- Predictable costs
- Graceful degradation under load
- Fair resource allocation

---

## Types of Rate Limits

### 1. API Provider Limits (External)

- **OpenAI:** 10K requests/min, 200K tokens/min (tier-based)
- **Anthropic:** Similar tiered limits
- **Tool APIs:** Varies by service

### 2. Self-Imposed Limits (Internal)

- Per-user quotas
- Cost budgets
- Resource protection

---

## Rate Limiting Strategies

### Token Bucket Algorithm

```python
import time
from threading import Lock

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def consume(self, tokens=1):
        """Try to consume tokens. Returns True if allowed."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill bucket
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

# Usage
limiter = TokenBucket(rate=10, capacity=100)  # 10 req/sec, burst of 100

if limiter.consume():
    result = call_api()
else:
    return "Rate limit exceeded. Please retry later."
```

---

### Sliding Window

```python
from collections import deque

class SlidingWindowRateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()

    def allow_request(self):
        """Check if request is allowed"""
        now = time.time()
        cutoff = now - self.window_seconds

        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

        # Check limit
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

# Usage: Max 100 requests per minute
limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
```

---

### Distributed Rate Limiting (Redis)

```python
import redis

class RedisRateLimiter:
    def __init__(self, redis_client, key_prefix="ratelimit"):
        self.redis = redis_client
        self.key_prefix = key_prefix

    def check_limit(self, identifier, max_requests, window_seconds):
        """
        Check rate limit for identifier (user_id, IP, etc.)
        """
        key = f"{self.key_prefix}:{identifier}"

        pipe = self.redis.pipeline()
        now = time.time()

        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window_seconds)

        # Count recent requests
        pipe.zcard(key)

        # Add current request timestamp
        pipe.zadd(key, {now: now})

        # Set expiry
        pipe.expire(key, window_seconds)

        results = pipe.execute()
        current_requests = results[1]

        return current_requests < max_requests

# Usage
redis_client = redis.Redis()
limiter = RedisRateLimiter(redis_client)

if limiter.check_limit(user_id, max_requests=100, window_seconds=60):
    proceed_with_request()
else:
    return_rate_limit_error()
```

---

## Queuing Strategies

### Basic Queue

```python
from queue import Queue
from threading import Thread

class RequestQueue:
    def __init__(self, rate_limit_per_second):
        self.queue = Queue()
        self.rate_limit = rate_limit_per_second
        self.running = True

        # Start worker
        Thread(target=self._worker, daemon=True).start()

    def add_request(self, func, callback):
        """Add request to queue"""
        self.queue.put((func, callback))

    def _worker(self):
        """Process queue at rate limit"""
        while self.running:
            if not self.queue.empty():
                func, callback = self.queue.get()

                try:
                    result = func()
                    callback(result)
                except Exception as e:
                    callback(None, error=e)

                # Wait to respect rate limit
                time.sleep(1 / self.rate_limit)
            else:
                time.sleep(0.1)

# Usage
queue = RequestQueue(rate_limit_per_second=10)

def process_result(result, error=None):
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")

queue.add_request(
    lambda: call_expensive_api(),
    process_result
)
```

---

### Priority Queue

```python
from queue import PriorityQueue

class PriorityRequestQueue:
    def __init__(self, rate_limit_per_second):
        self.queue = PriorityQueue()
        self.rate_limit = rate_limit_per_second
        Thread(target=self._worker, daemon=True).start()

    def add_request(self, func, callback, priority=5):
        """
        Add request with priority (lower number = higher priority)
        """
        self.queue.put((priority, func, callback))

    def _worker(self):
        while True:
            priority, func, callback = self.queue.get()

            try:
                result = func()
                callback(result)
            except Exception as e:
                callback(None, error=e)

            time.sleep(1 / self.rate_limit)

# Usage
queue = PriorityRequestQueue(rate_limit_per_second=10)

# High priority user
queue.add_request(api_call, callback, priority=1)

# Low priority batch job
queue.add_request(api_call, callback, priority=10)
```

---

## Backpressure Handling

### Circuit Breaker Integration

```python
class RateLimitedCircuitBreaker:
    def __init__(self, rate_limiter, circuit_breaker):
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker

    def call(self, func):
        # Check rate limit first
        if not self.rate_limiter.allow_request():
            raise RateLimitError("Rate limit exceeded")

        # Then check circuit breaker
        return self.circuit_breaker.call(func)
```

---

### Adaptive Rate Limiting

```python
class AdaptiveRateLimiter:
    def __init__(self, initial_rate=100):
        self.current_rate = initial_rate
        self.error_count = 0
        self.success_count = 0

    def record_result(self, success):
        if success:
            self.success_count += 1

            # Gradually increase rate if successful
            if self.success_count >= 10:
                self.current_rate = min(
                    self.current_rate * 1.1,
                    1000  # max rate
                )
                self.success_count = 0
        else:
            self.error_count += 1

            # Decrease rate on errors
            if self.error_count >= 3:
                self.current_rate = max(
                    self.current_rate * 0.5,
                    10  # min rate
                )
                self.error_count = 0

    def get_current_rate(self):
        return self.current_rate
```

---

## User-Level Quotas

### Tiered Limits

```python
class TieredRateLimiter:
    def __init__(self):
        self.tiers = {
            "free": {"requests_per_day": 100, "tokens_per_day": 10000},
            "pro": {"requests_per_day": 5000, "tokens_per_day": 1000000},
            "enterprise": {"requests_per_day": float('inf'), "tokens_per_day": float('inf')}
        }
        self.usage = {}

    def check_user_limit(self, user_id, user_tier):
        limits = self.tiers[user_tier]
        usage = self.usage.get(user_id, {"requests": 0, "tokens": 0})

        if usage["requests"] >= limits["requests_per_day"]:
            return False, "Daily request limit exceeded"

        if usage["tokens"] >= limits["tokens_per_day"]:
            return False, "Daily token limit exceeded"

        return True, None

    def record_usage(self, user_id, requests=1, tokens=0):
        if user_id not in self.usage:
            self.usage[user_id] = {"requests": 0, "tokens": 0}

        self.usage[user_id]["requests"] += requests
        self.usage[user_id]["tokens"] += tokens
```

---

## Handling 429 Errors

### Retry with Backoff

```python
def handle_rate_limit(func, max_retries=5):
    """Handle 429 errors with exponential backoff"""

    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise

            # Check retry-after header
            retry_after = e.response.headers.get("Retry-After", 60)
            wait_time = min(int(retry_after), 300)  # Max 5 min

            print(f"Rate limited. Waiting {wait_time}s before retry")
            time.sleep(wait_time)
```

---

## Production Best Practices

### 1. Monitor Rate Limit Usage

```python
class RateLimitMonitor:
    def __init__(self):
        self.limit_hits = 0
        self.total_requests = 0

    def record_request(self, was_limited):
        self.total_requests += 1
        if was_limited:
            self.limit_hits += 1

        # Alert if >10% requests are rate limited
        if self.total_requests % 100 == 0:
            hit_rate = self.limit_hits / self.total_requests
            if hit_rate > 0.1:
                alert(f"High rate limit hit rate: {hit_rate:.1%}")
```

### 2. Provide Clear Error Messages

```python
def rate_limit_response(user_tier, reset_time):
    return {
        "error": "Rate limit exceeded",
        "message": f"You've reached your {user_tier} tier limit",
        "reset_at": reset_time.isoformat(),
        "upgrade_url": "/pricing"
    }
```

### 3. Implement Gradual Rollout

```python
# Start conservative, increase gradually
initial_limit = 100  # requests/min
target_limit = 1000  # requests/min

# Increase 10% per day if no issues
current_limit = initial_limit * (1.1 ** days_since_launch)
```

---

## References

- **Rate Limiting Algorithms:** [NGINX Docs](https://www.nginx.com/blog/rate-limiting-nginx/)
- **Token Bucket:** [Wikipedia](https://en.wikipedia.org/wiki/Token_bucket)
- **Redis Rate Limiting:** [Redis Patterns](https://redis.io/docs/manual/patterns/distributed-locks/)

---

## Next Steps

- **Error handling?** → See [Error Handling](./error-handling.md)
- **Cost optimization?** → See [Cost Optimization](./cost-optimization.md)
- **Security?** → See [Security](./security.md)
