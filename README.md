## GenAI Proof of Concept - Generate Training & Inference Code from a Model Whitepaper

The purpose of this proof of concept is to find out if an LLM can take an econometric model whitepaper and generate the training and inference code. The whitepaper used for this PoC is FHFA's Repeat-Sales Aggregation Index: https://www.fhfa.gov/document/wp2101.pdf (saved as [hpi_fhfa_whitepaper.pdf](hpi_fhfa_whitepaper.pdf))

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

### Development Process: 
* Step 1 - request Opus 4 LLM to generate a Product Requirements Document (PRD) by analyzing the provided model whitepaper
* Step 2 - request Claude Code (using Opus 4 as the model) to generate a detailed implementation plan by analyzing the PRD generated in Step 1. For our POC, we will request Claude Code to generate 3 separate implementation plans:
  1) For Python and Pandas and save it as [IMPLEMENTATION_PLAN_PANDAS.md](IMPLEMENTATION_PLAN_PANDAS.md)
  2) For Python and Polars and save it as [IMPLEMENTATION_PLAN_POLARS.md](IMPLEMENTATION_PLAN_POLARS.md)
  3) For Python and PySpark/MLlib ans save it as [IMPLEMENTATION_PLAN_PYSPARK.md](IMPLEMENTATION_PLAN_PYSPARK.md)
* Step 3 - request Claude Code (with Opus 4) to implement 3 separate implementations of the code using the 3 implementation plans. Each plan includes requirements for comprehensive test coverage (both unit and integration tests)

### Implementation Details
* The pandas implementation resides under impl-pandas/ directory.
  * See [CLAUDE_CODE_SESSION_PANDAS.md](CLAUDE_CODE_SESSION_PANDAS.md) for all prompts issued to Claude Code. A summary response to each prompt by Claude Code is also included. Take time to review each Claude Code session to experience how the LLM thinks and uses train-of-thought to iterate through and fix each test failure.
  * See [impl-pandas/README.md](impl-pandas/README.md) details on the Pandas implementation.
  * Time took to implement: 1 hour
* The pandas implementation resides under impl-polars/ directory.
  * See [CLAUDE_CODE_SESSION_POLARS.md](CLAUDE_CODE_SESSION_POLARS.md) for all prompts issued to Claude Code.
  * See [impl-polars/README.md](impl-polars/README.md) details on the Polars implementation.
  * Time took to implement: 1.5 hours
* The polars implementation resides under impl-pyspark/ directory.
  * See [CLAUDE_CODE_SESSION_PYSPARK.md](CLAUDE_CODE_SESSION_PYSPARK.md) for all prompts issued to Claude Code.
  * See [impl-pyspark/README.md](impl-pyspark/README.md) details on the PySpark implementation.
  * Time took to implement: 3 hours

### Running the Generated Code
* For the Pandas implementation, refer to [impl-pandas/README.md](impl-pandas/README.md)
* For the Polars implementation, refer to [impl-polars/README.md](impl-polars/README.md)
* For the PySpark implementation, refer to [impl-pyspark/README.md](impl-pyspark/README.md)
