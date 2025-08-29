# aws-bedrock-agent-core
aws-bedrock-agent-core

Certainly. **README.md draft:**

```markdown
# Generic AgentCore System (LangChain + LangGraph + Bedrock)

Universal, extensible AI agent system with multiple agent types (Chat, ReAct, Workflow), Bedrock-backed LLMs, built-in tools, session-scoped memory, and simple deploy/run workflows.

---

## Features

- Agent patterns: Chat, ReAct (tools), Workflow (LangGraph)
- Bedrock LLMs via LangChain (Claude 3 / 3.5 pre-wired)
- Built-in tools: Calculator, Text Processor, Web Search (mock), Data Analyzer, original Claude tool
- Session memory for LangGraph agents (per-session checkpointer configuration)
- Quick-start scripts, interactive console, and server-mode
- Backward compatible with the original project files

---

## Project Structure

```

.
├─ generic\_agentcore\_system.py   # Core system: agents, tools, orchestrator
├─ main\_generic.py               # Entry point: console/server/config modes
├─ tools\_enhanced.py             # Enhanced tools (plus original Claude tool)
├─ quick\_start.py                # Dependency check, smoke tests, demo
├─ requirements\_working.txt      # Minimal dependencies to run now
├─ requirements.txt              # Original dependency list (kept)
├─ tools.py                      # Original file (backward compatible)
├─ graph.py                      # Original file (workflow helper)
├─ agent.py                      # Original file (agent helper)
└─ Dockerfile                    # Original container file (if used)

````

---

## Prerequisites

- Python 3.10+
- AWS credentials configured for Bedrock access

PowerShell:
```powershell
setx AWS_ACCESS_KEY_ID "YOUR_KEY"
setx AWS_SECRET_ACCESS_KEY "YOUR_SECRET"
setx AWS_DEFAULT_REGION "us-east-1"
````

Or via AWS CLI:

```bash
aws configure
```

---

## Installation

Minimal, ready-to-run set:

```bash
pip install -r requirements_working.txt
```

(Alternatively, install from `requirements.txt` if preferred.)

---

## Quick Start

Smoke test and demo:

```bash
python quick_start.py test
```

Interactive console:

```bash
python main_generic.py console
```

Server mode (endpoint simulation / deployment integration):

```bash
python main_generic.py server
```

Generate configuration template:

```bash
python main_generic.py generate-config
```

Load customized config:

```bash
python main_generic.py load-config
```

---

## Usage (Console)

After:

```bash
python main_generic.py console
```

Switch agents:

```
switch chat_agent
switch react_agent
```

List agents:

```
agents
```

Show tools for current agent:

```
tools
```

Exit:

```
exit
```

Examples:

* Chat: `Hello, what can be done here?`
* Math (ReAct): `Calculate sqrt(144) + 5 * 3`
* Text analysis (ReAct):

  ```
  Analyze this text: count
  This is a sample for analysis.
  ```
* Search (mock, ReAct): `Search for information about Python programming`

---

## Available Agents

* `chat_agent`: Simple conversational agent (LLM only)
* `react_agent`: ReAct-style reasoning + tools (Calculator, Text Processor, Web Search)
* `high_performance_agent`: ReAct-style with extended token budget (Claude 3.5 Sonnet pre-set)

Additional demo agents may be registered by scripts (e.g., math specialist, research assistant).

---

## Built-in Tools

* **Calculator**
  Supported: `+ - * / ** % sqrt() sin() cos() tan() log() exp() abs() round() pi e`

* **Text Processor**
  Commands: `count`, `upper`, `lower`, `title`, `reverse`, `words`, `summary`
  Format:

  ```
  command
  your text here...
  ```

* **Web Search (mock)**: Returns representative results (no external network calls by default)

* **Data Analyzer**: Light analysis for JSON and CSV

* **Original Claude Tool**: Preserved for backward compatibility

---

## Configuration & Memory

* Session memory for LangGraph agents is enabled by passing a per-session configuration:

  * `configurable.thread_id`: unique session identifier
  * `configurable.checkpoint_ns`: agent namespace
* The orchestrator handles this automatically when memory is enabled, ensuring persistent, session-scoped state for ReAct/workflow agents.

---

## Environment Variables

* `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (default `us-east-1`)
* Optional model/provider environment variables if extended in the future

---

## Troubleshooting

* **Checkpointer error about `thread_id/checkpoint_ns/checkpoint_id`:**
  The orchestrator passes required per-session config for memory-enabled agents. Ensure the updated code is running and rerun the test.

* **AWS credential issues:**
  Verify credentials and region, test with:

  ```bash
  aws sts get-caller-identity
  ```

* **Latency/token limits:**
  Use `high_performance_agent` or adjust `max_tokens` and model settings.

---

## Extending

Add a custom tool:

```python
from generic_agentcore_system import GenericTool

class MyTool(GenericTool):
    def __init__(self):
        super().__init__("my_tool", "Describe what this tool does")

    async def execute(self, **kwargs):
        # implement logic
        return {"result": "tool output", "success": True}
```

Register a custom agent:

```python
from generic_agentcore_system import GenericConfig, AgentType

orchestrator.register_agent(
    "my_agent",
    GenericConfig(agent_type=AgentType.REACT, temperature=0.1),
    tools=[MyTool()]
)
```

---

## License

Add license details here (e.g., MIT).

```

Recommendation: Save this as `README.md` in the project root.  
Next step: Run `python quick_start.py test` to verify dependencies and confirm the README instructions align with your environment.
```
