# Case Study: Data Analyst Agent

> SQL-generating agent that answers business questions from natural language

## Problem Statement

**Company:** E-commerce platform with complex data warehouse
**Challenge:** Business teams wait days for SQL queries from data team
**Goal:** Enable non-technical users to query data independently

### Pain Points Before Agent

- **Long wait times:** 2-5 days for data team to write queries
- **Bottleneck:** 3 data analysts for 50+ stakeholders
- **Query errors:** 30% of requests misunderstood or incorrectly implemented
- **Iteration cycles:** Multiple rounds of back-and-forth
- **Limited self-service:** Business users can't explore data independently

### Success Metrics

- Reduce query response time from days to minutes
- Handle 70%+ of routine data requests autonomously
- Maintain >95% query accuracy
- Enable 40+ business users to self-serve
- Reduce data team backlog by 60%

---

## Pattern Selection

### Why Tool Calling + Chain-of-Thought + Feedback Loop?

**Decision process:**

1. **Not RAG alone:** Need to execute SQL, not just retrieve docs
2. **Not Simple Tool Calling:** Complex queries require reasoning
3. **✅ Chain-of-Thought:** Break down complex questions into steps
4. **✅ Tool Calling:** Execute SQL, fetch data, generate visualizations
5. **✅ Feedback Loop:** Validate results, refine if needed

### Architecture Choice

```
Business Question
    ↓
Chain-of-Thought Reasoning:
    1. Understand question
    2. Identify required tables
    3. Plan SQL approach
    ↓
Tool Calling:
    - Generate SQL
    - Execute query
    - Validate results
    ↓
Feedback Loop:
    - Check if answer matches question
    - Refine query if needed
    ↓
Present Results + Visualization
```

---

## Architecture Diagram

```mermaid
graph TD
    A[Natural Language Question] --> B[Question Analyzer]

    B --> C[Chain-of-Thought Planner]

    C --> D{Query Complexity}

    D -->|Simple| E[Direct SQL Generation]
    D -->|Complex| F[Multi-Step Plan]

    E --> G[SQL Generator]
    F --> G

    G --> H[SQL Validator]

    H --> I{Valid SQL?}

    I -->|No| J[Error Analysis]
    J --> G

    I -->|Yes| K[Execute Query]

    K --> L[Data Warehouse]

    L --> M[Results Processor]

    M --> N{Results Valid?}

    N -->|No| O[Feedback Loop]
    O --> C

    N -->|Yes| P[Generate Explanation]

    P --> Q[Visualization Generator]

    Q --> R[Chart/Table]

    R --> S[Present to User]

    B --> T[Schema Retriever]
    T --> U[Vector Store]
    U --> G
```

---

## Implementation Details

### 1. Question Analysis

```python
class QuestionAnalyzer:
    def __init__(self, llm, schema_retriever):
        self.llm = llm
        self.schema = schema_retriever

    async def analyze(self, question):
        """Understand the business question"""

        prompt = f"""Analyze this data question:

Question: {question}

Extract:
1. **Intent:** What does the user want to know?
2. **Entities:** What data entities are involved? (users, orders, products, etc.)
3. **Metrics:** What needs to be calculated? (count, sum, average, etc.)
4. **Filters:** Any time ranges or conditions?
5. **Grouping:** How should data be grouped?
6. **Complexity:** simple, medium, or complex?

Return JSON.
"""

        response = await self.llm.generate(prompt, temperature=0.1)
        analysis = json.loads(response)

        # Retrieve relevant schema
        relevant_tables = await self.schema.find_tables(
            entities=analysis["entities"]
        )

        analysis["relevant_tables"] = relevant_tables

        return analysis
```

### 2. Schema Retriever (RAG for Database Schema)

```python
class SchemaRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.schema_cache = {}

    async def index_schema(self, database_url):
        """Index database schema for retrieval"""

        # Connect to database
        engine = create_engine(database_url)
        metadata = MetaData()
        metadata.reflect(bind=engine)

        # Index each table
        for table_name, table in metadata.tables.items():
            # Create searchable description
            description = self.describe_table(table)

            # Embed and store
            embedding = await self.embed(description)

            await self.vector_store.upsert(
                id=table_name,
                vector=embedding,
                metadata={
                    "table_name": table_name,
                    "description": description,
                    "columns": [
                        {
                            "name": col.name,
                            "type": str(col.type),
                            "nullable": col.nullable,
                            "description": col.comment
                        }
                        for col in table.columns
                    ],
                    "sample_queries": self.generate_sample_queries(table)
                }
            )

    def describe_table(self, table):
        """Generate natural language table description"""

        return f"""
Table: {table.name}
Purpose: {table.comment or "Data table"}
Columns:
{chr(10).join([f"- {col.name} ({col.type}): {col.comment or 'No description'}" for col in table.columns])}
"""

    async def find_tables(self, entities):
        """Find relevant tables for query"""

        query = " ".join(entities)
        embedding = await self.embed(query)

        results = await self.vector_store.query(
            vector=embedding,
            top_k=5
        )

        return [r.metadata for r in results.matches]
```

### 3. SQL Generator with Chain-of-Thought

```python
class SQLGenerator:
    def __init__(self, llm):
        self.llm = llm

    async def generate_sql(self, question_analysis, schema_info):
        """Generate SQL using chain-of-thought reasoning"""

        # Build schema context
        schema_context = self.format_schema(schema_info)

        prompt = f"""Generate SQL query for this question using step-by-step reasoning.

Question Analysis:
{json.dumps(question_analysis, indent=2)}

Available Tables & Columns:
{schema_context}

Think through this step by step:

Step 1: Identify which tables to use
Step 2: Determine what columns to SELECT
Step 3: Identify any JOINs needed
Step 4: Determine WHERE conditions
Step 5: Decide on GROUP BY and aggregations
Step 6: Add ORDER BY and LIMIT if needed

Then write the final SQL query.

Return JSON:
{{
    "reasoning": {{
        "step_1": "...",
        "step_2": "...",
        ...
    }},
    "sql": "SELECT ..."
}}
"""

        response = await self.llm.generate(prompt, temperature=0.1)
        result = json.loads(response)

        return result

    def format_schema(self, schema_info):
        """Format schema for LLM context"""

        formatted = []
        for table in schema_info:
            columns = "\n  ".join([
                f"- {col['name']} ({col['type']})"
                for col in table['columns']
            ])

            formatted.append(f"""
Table: {table['table_name']}
Columns:
  {columns}
Example queries:
  {chr(10).join(['  - ' + q for q in table.get('sample_queries', [])])}
""")

        return "\n".join(formatted)
```

### 4. SQL Validator

```python
class SQLValidator:
    def __init__(self, db_engine):
        self.engine = db_engine

    async def validate(self, sql):
        """Validate SQL without executing"""

        errors = []

        # 1. Syntax check using EXPLAIN
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
        except Exception as e:
            errors.append({
                "type": "syntax",
                "message": str(e)
            })
            return {"valid": False, "errors": errors}

        # 2. Check for dangerous operations
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT"]
        for keyword in dangerous_keywords:
            if keyword in sql.upper():
                errors.append({
                    "type": "safety",
                    "message": f"Query contains {keyword} which is not allowed"
                })

        # 3. Estimate query cost
        cost = await self.estimate_cost(sql)
        if cost > 1000:  # Arbitrary threshold
            errors.append({
                "type": "performance",
                "message": f"Query cost too high: {cost}. Consider adding filters."
            })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "estimated_cost": cost
        }

    async def estimate_cost(self, sql):
        """Estimate query execution cost"""

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"EXPLAIN (FORMAT JSON) {sql}"))
                plan = json.loads(result.scalar())
                return plan[0]["Plan"]["Total Cost"]
        except:
            return 0
```

### 5. Query Executor

```python
class QueryExecutor:
    def __init__(self, db_engine, timeout=30):
        self.engine = db_engine
        self.timeout = timeout

    async def execute(self, sql, max_rows=1000):
        """Execute SQL with safety limits"""

        # Add row limit if not present
        if "LIMIT" not in sql.upper():
            sql = f"{sql} LIMIT {max_rows}"

        try:
            # Execute with timeout
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(sql).execution_options(timeout=self.timeout)
                )

                # Fetch results
                rows = result.fetchall()
                columns = result.keys()

                # Convert to list of dicts
                data = [
                    dict(zip(columns, row))
                    for row in rows
                ]

                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "columns": list(columns)
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

### 6. Results Validator (Feedback Loop)

```python
class ResultsValidator:
    def __init__(self, llm):
        self.llm = llm

    async def validate_results(self, question, sql, results):
        """Check if results answer the question"""

        # Quick checks
        if not results["success"]:
            return {
                "valid": False,
                "reason": "query_failed",
                "suggestion": "Fix SQL error"
            }

        if results["row_count"] == 0:
            return {
                "valid": False,
                "reason": "no_results",
                "suggestion": "Relax filters or check data availability"
            }

        # LLM validation
        prompt = f"""Check if these query results answer the question.

Question: {question}

SQL: {sql}

Results (first 5 rows):
{json.dumps(results['data'][:5], indent=2)}

Total rows: {results['row_count']}

Does this answer the question? If not, what's wrong?

Return JSON:
{{
    "valid": true/false,
    "reason": "...",
    "suggestion": "how to fix" (if invalid)
}}
"""

        response = await self.llm.generate(prompt, temperature=0.1)
        validation = json.loads(response)

        return validation
```

### 7. Visualization Generator

```python
class VisualizationGenerator:
    def __init__(self, llm):
        self.llm = llm

    async def generate_viz(self, question, results):
        """Determine best visualization for data"""

        # Analyze data characteristics
        num_columns = len(results["columns"])
        num_rows = results["row_count"]
        column_types = self.infer_types(results["data"])

        prompt = f"""Recommend the best visualization for this data.

Question: {question}
Number of rows: {num_rows}
Columns: {json.dumps(column_types, indent=2)}

Visualization options:
- table: Detailed tabular view
- bar_chart: Compare categories
- line_chart: Show trends over time
- pie_chart: Show proportions
- scatter_plot: Show relationships

Recommend one with rationale. Return JSON:
{{
    "type": "...",
    "x_axis": "column_name",
    "y_axis": "column_name",
    "rationale": "why this viz"
}}
"""

        response = await self.llm.generate(prompt, temperature=0.2)
        viz_config = json.loads(response)

        # Generate chart
        chart = self.create_chart(results["data"], viz_config)

        return {
            "config": viz_config,
            "chart": chart
        }

    def create_chart(self, data, config):
        """Create chart using plotting library"""

        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(data)

        if config["type"] == "bar_chart":
            fig = px.bar(df, x=config["x_axis"], y=config["y_axis"])
        elif config["type"] == "line_chart":
            fig = px.line(df, x=config["x_axis"], y=config["y_axis"])
        elif config["type"] == "pie_chart":
            fig = px.pie(df, names=config["x_axis"], values=config["y_axis"])
        else:
            # Default to table
            return df.to_html()

        return fig.to_html()
```

### 8. Main Agent Orchestrator

```python
class DataAnalystAgent:
    def __init__(self, llm, db_engine, schema_retriever):
        self.question_analyzer = QuestionAnalyzer(llm, schema_retriever)
        self.sql_generator = SQLGenerator(llm)
        self.sql_validator = SQLValidator(db_engine)
        self.query_executor = QueryExecutor(db_engine)
        self.results_validator = ResultsValidator(llm)
        self.viz_generator = VisualizationGenerator(llm)

        self.max_retries = 3

    async def answer_question(self, question):
        """Main orchestration loop"""

        # Step 1: Analyze question
        analysis = await self.question_analyzer.analyze(question)

        # Retry loop for query refinement
        for attempt in range(self.max_retries):
            # Step 2: Generate SQL
            sql_result = await self.sql_generator.generate_sql(
                analysis,
                analysis["relevant_tables"]
            )

            sql = sql_result["sql"]
            reasoning = sql_result["reasoning"]

            logger.info(f"Attempt {attempt + 1}: Generated SQL: {sql}")

            # Step 3: Validate SQL
            validation = await self.sql_validator.validate(sql)

            if not validation["valid"]:
                logger.warning(f"SQL validation failed: {validation['errors']}")
                # Provide feedback to generator
                analysis["validation_errors"] = validation["errors"]
                continue

            # Step 4: Execute query
            results = await self.query_executor.execute(sql)

            if not results["success"]:
                logger.error(f"Query execution failed: {results['error']}")
                analysis["execution_error"] = results["error"]
                continue

            # Step 5: Validate results answer the question
            results_validation = await self.results_validator.validate_results(
                question,
                sql,
                results
            )

            if not results_validation["valid"]:
                logger.warning(f"Results invalid: {results_validation['reason']}")
                analysis["results_feedback"] = results_validation["suggestion"]
                continue

            # Step 6: Generate visualization
            viz = await self.viz_generator.generate_viz(question, results)

            # Step 7: Generate natural language explanation
            explanation = await self.generate_explanation(
                question,
                sql,
                reasoning,
                results
            )

            return {
                "question": question,
                "sql": sql,
                "reasoning": reasoning,
                "results": results["data"],
                "row_count": results["row_count"],
                "visualization": viz,
                "explanation": explanation,
                "attempts": attempt + 1
            }

        # Failed after retries
        return {
            "error": "Could not generate valid query after 3 attempts",
            "question": question
        }

    async def generate_explanation(self, question, sql, reasoning, results):
        """Generate natural language explanation"""

        prompt = f"""Explain these query results in simple terms.

Question: {question}

SQL: {sql}

Results summary:
- {results['row_count']} rows returned
- Sample: {json.dumps(results['data'][:3], indent=2)}

Provide:
1. Direct answer to the question (2-3 sentences)
2. Key insights from the data
3. Any caveats or limitations

Use business-friendly language, avoid technical jargon.
"""

        explanation = await self.llm.generate(prompt, temperature=0.3)

        return explanation
```

---

## Tools Integrated

### 1. Database Connectors

```python
# PostgreSQL, MySQL, Snowflake, BigQuery
from sqlalchemy import create_engine

engines = {
    "postgres": create_engine("postgresql://user:pass@host/db"),
    "snowflake": create_engine("snowflake://user:pass@account/db")
}
```

### 2. Visualization Libraries

```python
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
```

### 3. Schema Documentation

```python
# Auto-generate data dictionary
# Integrate with dbt docs, Atlan, etc.
```

---

## Cost Breakdown

**Monthly costs (40 users, ~500 queries/month):**

### LLM Costs

| **Task**               | **Volume**                   | **Model** | **Cost**    |
| ---------------------- | ---------------------------- | --------- | ----------- |
| Question analysis      | 500 queries                  | GPT-4     | $25         |
| SQL generation         | 500 queries (avg 2 attempts) | GPT-4     | $200        |
| Results validation     | 500 queries                  | GPT-4     | $50         |
| Explanation generation | 500 queries                  | GPT-4     | $75         |
| Visualization config   | 500 queries                  | GPT-3.5   | $10         |
| **Total LLM**          |                              |           | **$360/mo** |

### Infrastructure

| **Component**          | **Cost**    |
| ---------------------- | ----------- |
| Vector DB (schema)     | $50         |
| API hosting            | $40         |
| Database read replicas | $100        |
| **Total**              | **$190/mo** |

**Total Monthly Cost:** $550/mo

### ROI

**Before (data team doing queries):**

- 500 queries/month × 1 hour/query × $75/hr = $37,500/mo

**After (agent):**

- $550/mo cost
- Data team focused on complex analysis

**Savings:** $36,950/mo (98% reduction)

---

## Error Handling

### SQL Injection Prevention

```python
# Use parameterized queries
# Block dangerous keywords
# Validate all user input
```

### Query Timeout

```python
# 30-second timeout
# Fallback to simpler query
# Suggest optimization
```

### Graceful Degradation

```python
# If LLM fails: Provide template queries
# If viz fails: Return raw table
# If validation fails: Show results with warning
```

---

## Results & Learnings

### Metrics After 6 Months

| **Metric**             | **Before**  | **After**   | **Change**    |
| ---------------------- | ----------- | ----------- | ------------- |
| Query response time    | 2-5 days    | 2 minutes   | 99% faster    |
| Data team backlog      | 80 requests | 10 requests | 88% reduction |
| Query accuracy         | 70%         | 94%         | +34%          |
| Self-service adoption  | 0 users     | 42 users    | ✅            |
| Data team satisfaction | Low         | High        | ✅            |

### What Worked

✅ **Chain-of-thought:** Reasoning steps improved SQL quality
✅ **Schema RAG:** Relevant table retrieval was accurate
✅ **Feedback loop:** Iterative refinement fixed 85% of errors
✅ **Visualization:** Charts made insights clearer
✅ **Natural language explanations:** Non-technical users understood results

### Challenges

❌ **Problem:** Complex joins (>3 tables) failed initially
**Fix:** Added few-shot examples of complex joins in prompt

❌ **Problem:** Ambiguous questions generated wrong queries
**Fix:** Added clarifying questions before SQL generation

❌ **Problem:** Performance issues on large tables
**Fix:** Auto-added LIMIT, suggested indexes

### Key Learnings

1. **Schema context is critical:** RAG over database schema was game-changer
2. **Validation prevents bad queries:** 95%+ validation rate before execution
3. **Feedback loops work:** 2-3 attempts usually succeed
4. **Visualizations matter:** Users want charts, not just tables
5. **Trust through transparency:** Showing SQL builds confidence

---

## Code References

**Repository:**
🔗 [GitHub: data-analyst-agent](https://github.com/example/data-analyst-agent)

**Key Files:**

- `agent.py` - Main orchestrator
- `sql_generator.py` - Chain-of-thought SQL generation
- `schema_retriever.py` - RAG for database schema
- `validators.py` - SQL and results validation
- `visualization.py` - Chart generation

---

## Next Steps

- **More case studies?** → See [Customer Support](./customer-support-agent.md) or [Code Review](./code-review-agent.md)
- **Pattern details?** → See [Tool Calling](../01-patterns/tool-calling.md) or [Feedback Loop](../01-patterns/feedback-loop.md)
- **Production concerns?** → See [Error Handling](../02-production/error-handling.md)
