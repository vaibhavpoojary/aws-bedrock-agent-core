#!/usr/bin/env python3
"""
Generic AgentCore System - Working Implementation
Universal, extensible, production-ready AI agent system
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Core imports
try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp
    AGENTCORE_AVAILABLE = True
except ImportError:
    AGENTCORE_AVAILABLE = False
    print("⚠️  AgentCore SDK not available - running in standalone mode")

# LangChain imports
from langchain_aws import ChatBedrock
from langchain.tools import Tool
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Initialize application
if AGENTCORE_AVAILABLE:
    app = BedrockAgentCoreApp()
else:
    app = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================================================================================
# CONFIGURATION SYSTEM
# ================================================================================================

class AgentType(Enum):
    CHAT = "chat"
    REACT = "react"
    WORKFLOW = "workflow"
    CUSTOM = "custom"

class ModelProvider(Enum):
    BEDROCK_CLAUDE = "bedrock_claude"
    BEDROCK_TITAN = "bedrock_titan"
    CUSTOM = "custom"

@dataclass
class GenericConfig:
    """Universal configuration for agents"""
    agent_type: AgentType = AgentType.REACT
    model_provider: ModelProvider = ModelProvider.BEDROCK_CLAUDE
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    region: str = "us-east-1"
    temperature: float = 0.0
    max_tokens: int = 4000
    timeout: int = 300
    max_iterations: int = 10
    enable_memory: bool = True
    enable_tools: bool = True
    verbose: bool = False
    custom_instructions: Optional[str] = None

# ================================================================================================
# MODEL FACTORY
# ================================================================================================

class ModelFactory:
    """Factory for creating different models"""
    
    @staticmethod
    def create_model(config: GenericConfig):
        """Create model based on configuration"""
        try:
            if config.model_provider == ModelProvider.BEDROCK_CLAUDE:
                return ChatBedrock(
                    model_id=config.model_id,
                    region_name=config.region,
                    model_kwargs={
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens
                    }
                )
            else:
                raise ValueError(f"Unsupported model provider: {config.model_provider}")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            # Return a mock model for testing
            return MockModel()

class MockModel:
    """Mock model for testing when Bedrock is not available"""
    
    def invoke(self, messages):
        if isinstance(messages, list):
            last_message = messages[-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
            else:
                content = str(last_message)
        else:
            content = str(messages)
        
        return AIMessage(content=f"Mock response to: {content}")
    
    async def ainvoke(self, messages):
        return self.invoke(messages)

# ================================================================================================
# GENERIC TOOL SYSTEM
# ================================================================================================

class GenericTool(ABC):
    """Abstract base class for all tools"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        pass
    
    def to_langchain_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        def sync_execute(input_text: str):
            try:
                # Run async function in sync context
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.execute(input_text=input_text))
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.execute(input_text=input_text))
                    return result
                finally:
                    loop.close()
            except Exception as e:
                return {"error": str(e), "success": False}
        
        return Tool(
            name=self.name,
            func=sync_execute,
            description=self.description
        )

class CalculatorTool(GenericTool):
    """Enhanced calculator tool"""
    
    def __init__(self):
        super().__init__(
            "calculator",
            "Perform mathematical calculations. Support basic operations, functions like sqrt, sin, cos, log."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        input_text = kwargs.get('input_text', kwargs.get('input', str(kwargs)))
        
        try:
            import ast
            import operator
            import math
            
            # Supported operations
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }
            
            # Safe functions
            safe_functions = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'abs': abs,
                'round': round,
                'pi': math.pi,
                'e': math.e
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                elif isinstance(node, ast.Name):
                    if node.id in safe_functions:
                        return safe_functions[node.id]
                    else:
                        raise ValueError(f"Name {node.id} not allowed")
                elif isinstance(node, ast.Call):
                    func_name = node.func.id
                    if func_name in safe_functions:
                        args = [eval_expr(arg) for arg in node.args]
                        return safe_functions[func_name](*args)
                    else:
                        raise ValueError(f"Function {func_name} not allowed")
                else:
                    raise TypeError(f"Unsupported operation: {type(node)}")
            
            result = eval_expr(ast.parse(input_text, mode='eval').body)
            
            return {
                "result": f"Calculation result: {result}",
                "value": result,
                "expression": input_text,
                "success": True
            }
            
        except Exception as e:
            return {
                "result": f"Calculation error: {str(e)}",
                "error": str(e),
                "expression": input_text,
                "success": False
            }

class WebSearchTool(GenericTool):
    """Web search tool (mock implementation)"""
    
    def __init__(self):
        super().__init__(
            "web_search",
            "Search the web for current information on any topic."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get('input_text', kwargs.get('input', str(kwargs)))
        
        # Mock search results
        await asyncio.sleep(0.1)  # Simulate API call
        
        mock_results = [
            {
                "title": f"Latest information about {query}",
                "snippet": f"Current information about {query} from reliable sources.",
                "url": "https://example.com/result1"
            },
            {
                "title": f"{query} - Complete Guide",
                "snippet": f"Comprehensive guide covering all aspects of {query}.",
                "url": "https://example.com/result2"
            }
        ]
        
        result_text = f"Search results for '{query}':\n\n"
        for i, result in enumerate(mock_results, 1):
            result_text += f"{i}. {result['title']}\n{result['snippet']}\n\n"
        
        return {
            "result": result_text,
            "results": mock_results,
            "query": query,
            "success": True
        }

class TextProcessorTool(GenericTool):
    """Text processing tool"""
    
    def __init__(self):
        super().__init__(
            "text_processor",
            "Process text: count words/chars, change case, analyze content."
        )
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        input_text = kwargs.get('input_text', kwargs.get('input', str(kwargs)))
        
        try:
            word_count = len(input_text.split())
            char_count = len(input_text)
            
            analysis = {
                "original_text": input_text,
                "word_count": word_count,
                "character_count": char_count,
                "uppercase": input_text.upper(),
                "lowercase": input_text.lower(),
                "title_case": input_text.title()
            }
            
            result_text = f"Text Analysis:\n"
            result_text += f"- Words: {word_count}\n"
            result_text += f"- Characters: {char_count}\n"
            result_text += f"- Uppercase: {analysis['uppercase']}\n"
            
            return {
                "result": result_text,
                "analysis": analysis,
                "success": True
            }
            
        except Exception as e:
            return {
                "result": f"Text processing error: {str(e)}",
                "error": str(e),
                "success": False
            }

class LLMTool(GenericTool):
    """Generic LLM calling tool"""
    
    def __init__(self, model, name: str = "llm_tool"):
        super().__init__(
            name,
            "Call the language model for reasoning and responses"
        )
        self.model = model
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        input_text = kwargs.get('input_text', kwargs.get('input', str(kwargs)))
        
        try:
            messages = [HumanMessage(content=input_text)]
            
            if hasattr(self.model, 'ainvoke'):
                response = await self.model.ainvoke(messages)
            else:
                response = self.model.invoke(messages)
            
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "result": content,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"LLM Tool error: {str(e)}")
            return {
                "result": f"LLM error: {str(e)}",
                "error": str(e),
                "success": False
            }

# ================================================================================================
# GENERIC AGENT FACTORY
# ================================================================================================

class GenericAgentFactory:
    """Factory for creating different types of agents"""
    
    @staticmethod
    def create_agent(config: GenericConfig, tools: List[GenericTool] = None):
        """Create agent based on configuration"""
        
        model = ModelFactory.create_model(config)
        langchain_tools = []
        
        # Convert tools to LangChain format
        if tools and config.enable_tools:
            langchain_tools = [tool.to_langchain_tool() for tool in tools]
        
        # Add default LLM tool if no tools provided
        if not tools:
            llm_tool = LLMTool(model)
            langchain_tools = [llm_tool.to_langchain_tool()]
        
        # Create agent based on type
        if config.agent_type == AgentType.REACT:
            try:
                from langgraph.prebuilt.chat_agent_executor import create_react_agent
                return create_react_agent(
                    model=model,
                    tools=langchain_tools,
                    checkpointer=MemorySaver() if config.enable_memory else None
                )
            except ImportError:
                # Fallback to simple agent
                return GenericAgentFactory._create_simple_agent(model, langchain_tools, config)
        
        elif config.agent_type == AgentType.CHAT:
            return GenericAgentFactory._create_chat_agent(model, config)
        
        elif config.agent_type == AgentType.WORKFLOW:
            return GenericAgentFactory._create_workflow_agent(model, langchain_tools, config)
        
        else:
            return GenericAgentFactory._create_simple_agent(model, langchain_tools, config)
    
    @staticmethod
    def _create_simple_agent(model, tools, config: GenericConfig):
        """Create simple agent without complex dependencies"""
        # --- in GenericAgentFactory._create_simple_agent(...) ---
        # in GenericAgentFactory._create_simple_agent(...)
        # --- in GenericAgentFactory._create_simple_agent(...)

        class SimpleAgent:
            def __init__(self, model, tools):
                self.model = model
                self.tools = {tool.name: tool for tool in tools} if tools else {}
                self.memory = [] if config.enable_memory else None

            async def ainvoke(self, state: Dict[str, Any], config_override: Dict = None):
                messages = state.get("messages", [])
                if config.enable_memory and self.memory:
                    messages = self.memory + messages
                response = await self.model.ainvoke(messages)
                if config.enable_memory:
                    self.memory.extend(messages + [response])
                    if len(self.memory) > 20:
                        self.memory = self.memory[-20:]
                return {"messages": messages + [response]}

            def invoke(self, state: Dict[str, Any], config_override: Dict = None):
                try:
                    loop = asyncio.get_event_loop()
                    return loop.run_until_complete(self.ainvoke(state, config_override))
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.ainvoke(state, config_override))
                    finally:
                        loop.close()

    
    @staticmethod
    def _create_chat_agent(model, config: GenericConfig):
        """Create simple chat agent"""
        return GenericAgentFactory._create_simple_agent(model, [], config)
    
    @staticmethod
    def _create_workflow_agent(model, tools, config: GenericConfig):
        """Create workflow agent with basic graph"""
        def agent_node(state):
            messages = state.get("messages", [])
            response = model.invoke(messages)
            return {"messages": messages + [response]}
        
        # Build simple workflow
        workflow = StateGraph(dict)
        workflow.add_node("agent", agent_node)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        
        return workflow.compile(checkpointer=MemorySaver() if config.enable_memory else None)

# ================================================================================================
# GENERIC ORCHESTRATOR
# ================================================================================================

class GenericOrchestrator:
    """Universal orchestrator for managing agents"""
    
    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.configs: Dict[str, GenericConfig] = {}
        self.tools: Dict[str, List[GenericTool]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
        # Setup default agents
        self._setup_default_agents()
    
    def _setup_default_agents(self):
        """Setup default agents"""
        try:
            # Default chat agent
            self.register_agent(
                agent_id="chat_agent",
                config=GenericConfig(agent_type=AgentType.CHAT),
                tools=[]
            )
            
            # Enhanced agent with tools
            tools = [
                CalculatorTool(),
                WebSearchTool(),
                TextProcessorTool()
            ]
            
            self.register_agent(
                agent_id="react_agent",
                config=GenericConfig(
                    agent_type=AgentType.REACT,
                    enable_tools=True,
                    verbose=True
                ),
                tools=tools
            )
            
            # High performance agent
            self.register_agent(
                agent_id="high_performance_agent",
                config=GenericConfig(
                    agent_type=AgentType.REACT,
                    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    temperature=0.1,
                    max_tokens=8000
                ),
                tools=tools
            )
            
            logger.info(f"✅ Setup {len(self.agents)} default agents")
            
        except Exception as e:
            logger.error(f"Failed to setup default agents: {e}")
    
    def register_agent(self, agent_id: str, config: GenericConfig, tools: List[GenericTool] = None):
        """Register a new agent"""
        try:
            self.configs[agent_id] = config
            self.tools[agent_id] = tools or []
            self.agents[agent_id] = GenericAgentFactory.create_agent(config, tools)
            
            logger.info(f"Registered agent: {agent_id} ({config.agent_type.value})")
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
    
    async def execute_agent(self, agent_id: str, input_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Execute specific agent"""
        
        if agent_id not in self.agents:
            return {
                "error": f"Agent {agent_id} not found",
                "available_agents": list(self.agents.keys()),
                "success": False
            }
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            agent = self.agents[agent_id]
            config = self.configs[agent_id]
            
            # Prepare input
            if "messages" not in input_data:
                user_input = input_data.get("input", "")
                input_data["messages"] = [HumanMessage(content=user_input)]
            
            # Execute agent
            if hasattr(agent, 'ainvoke'):
                result = await agent.ainvoke(input_data)
            else:
                # Run sync method in async context
                result = agent.invoke(input_data)
            
            # Extract response
            response_text = "No response generated"
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = str(last_message)
            else:
                response_text = str(result)
            
            return {
                "agent_id": agent_id,
                "session_id": session_id,
                "result": result,
                "response": response_text,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Agent execution error for {agent_id}: {str(e)}")
            return {
                "agent_id": agent_id,
                "session_id": session_id,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about registered agents"""
        return {
            "total_agents": len(self.agents),
            "agents": {
                agent_id: {
                    "type": config.agent_type.value,
                    "model": config.model_provider.value,
                    "model_id": config.model_id,
                    "tools_count": len(self.tools.get(agent_id, []))
                }
                for agent_id, config in self.configs.items()
            }
        }

# ================================================================================================
# AGENTCORE ENDPOINTS (if available)
# ================================================================================================

# Initialize orchestrator
orchestrator = GenericOrchestrator()

if AGENTCORE_AVAILABLE and app:
    
    @app.entrypoint
    # --- in GenericOrchestrator.execute_agent(...) ---
    async def execute_agent(self, agent_id: str, input_data: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found", "available_agents": list(self.agents.keys()), "success": False}

        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            agent = self.agents[agent_id]
            config = self.configs[agent_id]

            # Prepare input
            if "messages" not in input_data:
                user_input = input_data.get("input", "")
                input_data["messages"] = [HumanMessage(content=user_input)]

            # IMPORTANT: Provide checkpointer config for LangGraph agents
# in GenericOrchestrator.execute_agent(...)

# Build the LangGraph checkpointer config when memory is enabled
# Build the LangGraph checkpointer config when memory is enabled
            agent_config = {}
            if config.enable_memory:
                agent_config = {
                    "configurable": {
                        "thread_id": session_id,                  # required per-session identifier
                        "checkpoint_ns": f"agent_{agent_id}"      # namespace for this agent
                        # Optional: "checkpoint_id": "stable-resume-id"
                    }
                }

            # Execute with config (SimpleAgent ignores it safely)
            if hasattr(agent, 'ainvoke'):
                result = await agent.ainvoke(input_data, agent_config)
            elif hasattr(agent, 'invoke'):
                result = agent.invoke(input_data, agent_config)
            else:
                result = agent.invoke(input_data)
                
            # Extract response...
            response_text = "No response generated"
            if isinstance(result, dict) and "messages" in result:
                last_message = result["messages"][-1]
                response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
            else:
                response_text = str(result)

            return {
                "agent_id": agent_id,
                "session_id": session_id,
                "result": result,
                "response": response_text,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Agent execution error for {agent_id}: {str(e)}")
            return {
                "agent_id": agent_id,
                "session_id": session_id,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }

    
    @app.entrypoint
    async def agent_info_endpoint(payload: Dict[str, Any], context) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_info": orchestrator.get_agent_info(),
            "supported_types": [t.value for t in AgentType],
            "supported_models": [m.value for m in ModelProvider],
            "agent_type": "agent_info",
            "timestamp": datetime.now().isoformat()
        }

# ================================================================================================
# TESTING FUNCTIONS
# ================================================================================================

async def test_generic_system():
    """Test the generic system"""
    
    print("🧪 Testing Generic AgentCore System")
    print("=" * 50)
    
    # Test agent info
    print("\n1️⃣ Agent Information:")
    info = orchestrator.get_agent_info()
    print(f"Total agents: {info['total_agents']}")
    for agent_id, agent_info in info['agents'].items():
        print(f"  • {agent_id}: {agent_info['type']} ({agent_info['model_id']})")
    
    # Test chat agent
    print("\n2️⃣ Testing Chat Agent")
    result = await orchestrator.execute_agent(
        agent_id="chat_agent",
        input_data={"input": "Hello! What can you help me with?"}
    )
    print(f"Success: {result.get('success')}")
    if result.get('response'):
        print(f"Response: {result['response'][:100]}...")
    
    # Test react agent with tools
    print("\n3️⃣ Testing ReAct Agent with Tools")
    result = await orchestrator.execute_agent(
        agent_id="react_agent",
        input_data={"input": "Calculate 25 * 4 + 10 and tell me about it"}
    )
    print(f"Success: {result.get('success')}")
    if result.get('response'):
        print(f"Response: {result['response'][:150]}...")
    
    # Test individual tool
    print("\n4️⃣ Testing Calculator Tool")
    calc_tool = CalculatorTool()
    calc_result = await calc_tool.execute(input_text="sqrt(144)")
    print(f"Calculator result: {calc_result}")
    
    print("\n✅ Generic system testing completed!")

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    print("🚀 Generic AgentCore System")
    print("Running in standalone mode..." if not AGENTCORE_AVAILABLE else "AgentCore SDK available")
    
    # Run tests
    asyncio.run(test_generic_system())