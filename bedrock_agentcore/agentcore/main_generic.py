#!/usr/bin/env python3
"""
Main Generic AgentCore Application - Working Implementation
Flexible entry point that adapts to any setup
"""

import sys
import asyncio
import json
import argparse
from datetime import datetime
from typing import Dict, Any

# Import the generic system
from generic_agentcore_system import (
    GenericOrchestrator, GenericConfig, AgentType, ModelProvider,
    CalculatorTool, WebSearchTool, TextProcessorTool, LLMTool,
    ModelFactory, logger
)

class GenericAgentCoreApp:
    """Main application class"""
    
    def __init__(self):
        self.orchestrator = GenericOrchestrator()
        self.setup_enhanced_agents()
    
    def setup_enhanced_agents(self):
        """Setup enhanced agents with different configurations"""
        
        try:
            # Math specialist
            math_tools = [CalculatorTool()]
            self.orchestrator.register_agent(
                agent_id="math_specialist",
                config=GenericConfig(
                    agent_type=AgentType.REACT,
                    temperature=0.0,
                    custom_instructions="You are a mathematics expert. Always show your work."
                ),
                tools=math_tools
            )
            
            # Research assistant  
            research_tools = [WebSearchTool(), TextProcessorTool()]
            self.orchestrator.register_agent(
                agent_id="research_assistant",
                config=GenericConfig(
                    agent_type=AgentType.REACT,
                    temperature=0.2,
                    custom_instructions="You are a research assistant. Provide comprehensive, well-sourced answers."
                ),
                tools=research_tools
            )
            
            # All-purpose agent
            all_tools = [CalculatorTool(), WebSearchTool(), TextProcessorTool()]
            self.orchestrator.register_agent(
                agent_id="all_purpose_agent",
                config=GenericConfig(
                    agent_type=AgentType.REACT,
                    temperature=0.1,
                    max_tokens=8000,
                    custom_instructions="You are a helpful AI assistant with access to multiple tools."
                ),
                tools=all_tools
            )
            
            print(f"✅ Enhanced setup complete - {len(self.orchestrator.agents)} agents available")
            
        except Exception as e:
            logger.error(f"Enhanced setup failed: {e}")
    
    async def run_console_mode(self):
        """Interactive console mode"""
        print("\\n🤖 Generic AgentCore Console")
        print("=" * 40)
        
        # Show available agents
        info = self.orchestrator.get_agent_info()
        print(f"\\n📋 Available Agents ({info['total_agents']}):")
        for agent_id, agent_info in info['agents'].items():
            print(f"  • {agent_id}: {agent_info['type']} - {agent_info['tools_count']} tools")
        
        print("\\n💡 Commands:")
        print("  switch <agent_id> - Change agent")
        print("  agents - List agents")
        print("  tools - Show current agent tools")
        print("  exit - Quit")
        print()
        
        current_agent = "react_agent"
        session_id = f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        while True:
            try:
                user_input = input(f"[{current_agent}] You: ")
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break
                
                if user_input.startswith("switch "):
                    new_agent = user_input.split("switch ")[1].strip()
                    if new_agent in self.orchestrator.agents:
                        current_agent = new_agent
                        print(f"🔄 Switched to: {current_agent}")
                        continue
                    else:
                        print(f"❌ Agent '{new_agent}' not found")
                        continue
                
                if user_input == "agents":
                    info = self.orchestrator.get_agent_info()
                    print("\\n📋 Available Agents:")
                    for agent_id, agent_info in info['agents'].items():
                        status = "→ CURRENT" if agent_id == current_agent else ""
                        print(f"  • {agent_id}: {agent_info['type']} {status}")
                    print()
                    continue
                
                if user_input == "tools":
                    agent_tools = self.orchestrator.tools.get(current_agent, [])
                    print(f"\\n🛠️ {current_agent} tools:")
                    if agent_tools:
                        for tool in agent_tools:
                            print(f"  • {tool.name}: {tool.description}")
                    else:
                        print("  No tools configured")
                    print()
                    continue
                
                if not user_input.strip():
                    continue
                
                # Execute agent
                print("🤔 Thinking...")
                result = await self.orchestrator.execute_agent(
                    agent_id=current_agent,
                    input_data={"input": user_input},
                    session_id=session_id
                )
                
                if result.get("success"):
                    response = result.get("response", "No response generated")
                    print(f"Agent: {response}")
                else:
                    error = result.get("error", "Unknown error")
                    print(f"❌ Error: {error}")
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def run_server_mode(self):
        """Run as server (for AgentCore deployment)"""
        print("🚀 Generic AgentCore Server Mode")
        print("=" * 40)
        
        # Show agent information
        info = self.orchestrator.get_agent_info()
        print(f"\\n✅ Server ready with {info['total_agents']} agents:")
        for agent_id, agent_info in info['agents'].items():
            print(f"  • {agent_id}: {agent_info['type']} ({agent_info['model_id']})")
        
        print("\\n📡 Endpoints available:")
        print("  POST /invocations - Main agent endpoint")
        print("  GET  /ping - Health check")
        
        print("\\n🔧 Deployment commands:")
        print("  agentcore configure --entrypoint generic_agentcore_system.py")
        print("  agentcore launch")
        
        # Keep server running
        try:
            print("\\n🎯 Server running... (Ctrl+C to stop)")
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n👋 Server stopped")
    
    def generate_config_template(self):
        """Generate configuration template"""
        config_template = {
            "agents": {
                "custom_chat_agent": {
                    "agent_type": "chat", 
                    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "temperature": 0.1,
                    "custom_instructions": "You are a helpful AI assistant."
                },
                "custom_research_agent": {
                    "agent_type": "react",
                    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "temperature": 0.2,
                    "tools": ["web_search", "text_processor"],
                    "custom_instructions": "You are a research specialist."
                },
                "custom_math_agent": {
                    "agent_type": "react",
                    "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "temperature": 0.0,
                    "tools": ["calculator"],
                    "custom_instructions": "You are a mathematics expert."
                }
            },
            "global_settings": {
                "region": "us-east-1",
                "max_tokens": 4000,
                "timeout": 300,
                "enable_memory": True,
                "verbose": False
            },
            "tools": {
                "calculator": {
                    "enabled": True,
                    "description": "Mathematical calculations"
                },
                "web_search": {
                    "enabled": True,
                    "description": "Web search capabilities"
                },
                "text_processor": {
                    "enabled": True,
                    "description": "Text analysis and processing"
                }
            }
        }
        
        config_file = "generic_agentcore_config.json"
        with open(config_file, "w") as f:
            json.dump(config_template, f, indent=2)
        
        print(f"✅ Configuration template saved to: {config_file}")
        print("\\n📝 Customization instructions:")
        print("1. Edit the JSON file to configure your agents")
        print("2. Run: python main_generic.py load-config")
        print("3. Test with: python main_generic.py console")
    
    def load_config(self, config_file: str = "generic_agentcore_config.json"):
        """Load configuration from file"""
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            
            agents_config = config.get("agents", {})
            
            for agent_id, agent_config in agents_config.items():
                # Parse configuration
                agent_type = AgentType(agent_config.get("agent_type", "react"))
                model_id = agent_config.get("model_id", "anthropic.claude-3-5-sonnet-20241022-v2:0")
                temperature = agent_config.get("temperature", 0.1)
                
                # Create tools
                tool_names = agent_config.get("tools", [])
                tools = []
                for tool_name in tool_names:
                    if tool_name == "calculator":
                        tools.append(CalculatorTool())
                    elif tool_name == "web_search":
                        tools.append(WebSearchTool())
                    elif tool_name == "text_processor":
                        tools.append(TextProcessorTool())
                
                # Create configuration
                config_obj = GenericConfig(
                    agent_type=agent_type,
                    model_id=model_id,
                    temperature=temperature,
                    custom_instructions=agent_config.get("custom_instructions")
                )
                
                # Register agent
                self.orchestrator.register_agent(agent_id, config_obj, tools)
                print(f"✅ Loaded agent: {agent_id}")
            
            print(f"\\n🎯 Successfully loaded {len(agents_config)} custom agents")
            
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_file}")
            print("Run 'python main_generic.py generate-config' to create a template")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
    
    async def run_single_query(self, agent_id: str, query: str):
        """Run a single query and return result"""
        result = await self.orchestrator.execute_agent(
            agent_id=agent_id,
            input_data={"input": query}
        )
        
        if result.get("success"):
            print(f"✅ Result: {result.get('response', 'No response')}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return result

def main():
    """Main entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(description="Generic AgentCore System")
    parser.add_argument(
        "mode", 
        nargs="?", 
        choices=["console", "server", "generate-config", "load-config", "test"],
        default="console",
        help="Run mode (default: console)"
    )
    parser.add_argument("--agent", help="Agent ID for single query mode")
    parser.add_argument("--query", help="Query for single query mode")
    parser.add_argument("--config", help="Config file path", default="generic_agentcore_config.json")
    
    args = parser.parse_args()
    
    # Initialize app
    app = GenericAgentCoreApp()
    
    try:
        if args.mode == "console":
            asyncio.run(app.run_console_mode())
        
        elif args.mode == "server":
            app.run_server_mode()
        
        elif args.mode == "generate-config":
            app.generate_config_template()
        
        elif args.mode == "load-config":
            app.load_config(args.config)
        
        elif args.mode == "test":
            # Run quick test
            async def test():
                print("🧪 Quick Test Mode")
                result = await app.orchestrator.execute_agent(
                    "react_agent", 
                    {"input": "What is 2 + 2 * 3?"}
                )
                print(f"Test result: {result.get('response', 'No response')}")
            
            asyncio.run(test())
        
        # Single query mode
        elif args.agent and args.query:
            asyncio.run(app.run_single_query(args.agent, args.query))
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)

# Backward compatibility with your original main.py
def run_console_agent():
    """Your original function - now powered by generic system"""
    app = GenericAgentCoreApp()
    asyncio.run(app.run_console_mode())

if __name__ == "__main__":
    main()