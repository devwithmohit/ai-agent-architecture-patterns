# Testing Strategies for AI Agents

> How to test non-deterministic systems that call LLMs

## The Testing Challenge

**Why agent testing is hard:**

- **Non-deterministic:** Same input → different outputs
- **External dependencies:** LLM APIs, tools, databases
- **Complex workflows:** Multi-step reasoning chains
- **Expensive:** Every test costs money (API calls)
- **Slow:** Network latency for each call

**Solution:** Layer testing strategies

---

## Testing Pyramid for Agents

```
        /\
       /  \
      /E2E \    ← Few (expensive, slow)
     /------\
    /  Inte- \  ← Some (moderate cost)
   /  gration \
  /------------\
 /  Unit Tests  \ ← Many (cheap, fast)
/________________\
```

---

## Unit Testing

### Test Deterministic Components

```python
def test_parse_tool_call():
    """Test parsing logic without LLM"""

    response = 'Action: search_web\nAction Input: {"query": "AI"}'

    parsed = parse_tool_call(response)

    assert parsed["tool"] == "search_web"
    assert parsed["args"]["query"] == "AI"

def test_validate_tool_args():
    """Test validation logic"""

    args = {"query": "test", "limit": 10}
    is_valid = validate_args("search_web", args)

    assert is_valid == True

    # Test invalid args
    invalid_args = {"invalid_param": "value"}
    is_valid = validate_args("search_web", invalid_args)

    assert is_valid == False
```

### Mock LLM Responses

```python
from unittest.mock import Mock, patch

def test_agent_with_mock_llm():
    """Test agent logic without calling real LLM"""

    mock_llm = Mock()
    mock_llm.generate.return_value = "The capital of France is Paris."

    agent = Agent(llm=mock_llm)
    result = agent.answer("What's the capital of France?")

    assert "Paris" in result
    mock_llm.generate.assert_called_once()
```

---

## Integration Testing

### Test Tool Execution

```python
def test_tool_execution_integration():
    """Test tool calls with real execution"""

    # Use test environment
    tool = SearchWebTool(api_key=TEST_API_KEY)

    result = tool.execute(query="test query")

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0
```

### Test Full Agent Flow (Mocked LLM)

```python
def test_agent_flow():
    """Test multi-step agent execution"""

    mock_llm = Mock()
    mock_llm.generate.side_effect = [
        "I should search for information.\nAction: search_web",
        "Based on the results, the answer is X."
    ]

    agent = ReActAgent(llm=mock_llm, tools=tools)
    result = agent.run("Test query")

    assert result is not None
    assert mock_llm.generate.call_count == 2
```

---

## End-to-End Testing

### Test with Real LLM (Sparingly)

```python
import pytest

@pytest.mark.slow
@pytest.mark.integration
def test_agent_e2e():
    """Full integration test with real APIs"""

    agent = Agent(llm=real_llm, tools=real_tools)

    result = agent.run("What's the weather in Seattle?")

    # Fuzzy assertions for non-deterministic outputs
    assert result is not None
    assert "Seattle" in result or "weather" in result
    assert len(result) > 10
```

### Assertion Strategies for Non-Deterministic Outputs

```python
def test_sentiment_analysis():
    """Test LLM sentiment with fuzzy matching"""

    positive_text = "I absolutely love this product!"
    result = agent.analyze_sentiment(positive_text)

    # Instead of exact match, check sentiment category
    assert result["sentiment"] in ["positive", "very positive"]
    assert result["score"] > 0.7

def test_output_contains_required_info():
    """Check output has required elements"""

    result = agent.generate_report(data)

    # Verify structure, not exact content
    assert "summary" in result
    assert "recommendations" in result
    assert len(result["recommendations"]) >= 3
```

---

## Evaluation Frameworks

### LLM-as-Judge

```python
def evaluate_output_quality(output, expected_criteria):
    """Use LLM to evaluate another LLM's output"""

    eval_prompt = f"""
Evaluate this AI response based on these criteria:
{expected_criteria}

Response to evaluate:
{output}

Rate each criterion from 1-5:
- Accuracy:
- Completeness:
- Clarity:

Provide JSON output.
"""

    evaluation = eval_llm.generate(eval_prompt)
    return json.loads(evaluation)

# Usage in test
def test_response_quality():
    response = agent.answer("Explain quantum computing")

    scores = evaluate_output_quality(
        response,
        criteria=["Accuracy", "Completeness", "Clarity"]
    )

    assert scores["Accuracy"] >= 4
    assert scores["Completeness"] >= 3
```

### Golden Dataset Testing

```python
class GoldenDatasetEvaluator:
    def __init__(self, golden_dataset):
        self.golden_dataset = golden_dataset

    def evaluate_agent(self, agent):
        """Test agent against curated examples"""

        results = []

        for example in self.golden_dataset:
            output = agent.run(example["input"])

            # Compare with expected output
            similarity = calculate_similarity(
                output,
                example["expected_output"]
            )

            results.append({
                "input": example["input"],
                "similarity": similarity,
                "passed": similarity > 0.8
            })

        pass_rate = sum(r["passed"] for r in results) / len(results)

        return {
            "pass_rate": pass_rate,
            "results": results
        }

# Golden dataset
golden_dataset = [
    {
        "input": "What's 2+2?",
        "expected_output": "4"
    },
    {
        "input": "Capital of France?",
        "expected_output": "Paris"
    }
]

evaluator = GoldenDatasetEvaluator(golden_dataset)
report = evaluator.evaluate_agent(my_agent)

assert report["pass_rate"] > 0.9
```

---

## Regression Testing

### Test Suite for Critical Paths

```python
@pytest.mark.regression
class TestCriticalPaths:
    """Ensure critical workflows don't break"""

    def test_user_onboarding_flow(self):
        """Critical: New user sign-up"""
        agent = OnboardingAgent()
        result = agent.onboard_user(test_user_data)
        assert result["status"] == "success"

    def test_payment_processing(self):
        """Critical: Payment flow"""
        agent = PaymentAgent()
        result = agent.process_payment(test_payment)
        assert result["status"] == "completed"
```

### Monitor for Degradation

```python
class RegressionMonitor:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics

    def check_regression(self, current_metrics):
        """Alert if performance degrades"""

        degradations = []

        for metric, baseline_value in self.baseline.items():
            current_value = current_metrics.get(metric)

            # Allow 10% degradation
            if current_value < baseline_value * 0.9:
                degradations.append({
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "degradation": (baseline_value - current_value) / baseline_value
                })

        if degradations:
            alert(f"Performance regression detected: {degradations}")

        return len(degradations) == 0
```

---

## Cost-Effective Testing

### Caching LLM Responses

```python
import hashlib
import pickle

class TestLLMCache:
    def __init__(self, cache_file="test_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def get_or_call(self, prompt, llm_func):
        """Cache LLM responses for tests"""

        cache_key = hashlib.sha256(prompt.encode()).hexdigest()

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Call real LLM
        response = llm_func(prompt)

        # Cache for future tests
        self.cache[cache_key] = response
        self._save_cache()

        return response

    def _save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

# Usage
cache = TestLLMCache()

def test_with_caching():
    response = cache.get_or_call(
        "What is AI?",
        lambda p: llm.generate(p)
    )
    # First run: calls LLM, costs money
    # Subsequent runs: uses cache, free
```

---

## Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=100))
def test_agent_handles_any_input(user_input):
    """Test agent doesn't crash on arbitrary input"""

    try:
        result = agent.run(user_input)
        # Should always return something
        assert result is not None
    except Exception as e:
        # Should handle errors gracefully
        assert "Internal error" not in str(e)
```

---

## Monitoring in Production (Testing in Prod)

### Synthetic Monitoring

```python
def synthetic_test():
    """Periodically test agent in production"""

    test_queries = [
        "What's 2+2?",  # Should always work
        "What's the weather?",  # Tests tool calling
    ]

    for query in test_queries:
        start_time = time.time()

        try:
            result = agent.run(query)
            duration = time.time() - start_time

            log_metric("synthetic_test_success", 1)
            log_metric("synthetic_test_duration", duration)

        except Exception as e:
            log_metric("synthetic_test_failure", 1)
            alert(f"Synthetic test failed: {e}")

# Run every 5 minutes
schedule.every(5).minutes.do(synthetic_test)
```

---

## Common Pitfalls

### 1. Testing Too Much with Real LLMs

**Cost:** $100s-$1000s in test runs
**Solution:** Mock LLMs, cache responses

### 2. Flaky Tests

**Problem:** Non-deterministic outputs fail randomly
**Solution:** Fuzzy assertions, test multiple runs

### 3. No Golden Dataset

**Problem:** Can't track regression
**Solution:** Curate critical examples

### 4. Testing Only Happy Path

**Problem:** Edge cases break in production
**Solution:** Property-based testing, error injection

---

## References

- **pytest:** [Testing Framework](https://docs.pytest.org/)
- **Hypothesis:** [Property Testing](https://hypothesis.readthedocs.io/)
- **LangChain Evaluation:** [Eval Tools](https://python.langchain.com/docs/guides/evaluation)
- **OpenAI Evals:** [Evals Framework](https://github.com/openai/evals)

---

## Next Steps

- **Need observability?** → See [Observability](./observability.md)
- **Cost concerns?** → See [Cost Optimization](./cost-optimization.md)
- **Security testing?** → See [Security](./security.md)
