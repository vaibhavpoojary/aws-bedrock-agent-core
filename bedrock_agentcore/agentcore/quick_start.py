#!/usr/bin/env python3
"""
Quick Start Script for Generic AgentCore System
Run this to test everything works immediately
"""

import sys
import asyncio
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    missing_deps = []
    
    try:
        import langchain
        print("✅ LangChain available")
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import langchain_aws
        print("✅ LangChain AWS available")
    except ImportError:
        missing_deps.append("langchain-aws")
    
    try:
        import langgraph
        print("✅ LangGraph available")
    except ImportError:
        missing_deps.append("langgraph")
    
    try:
        import boto3
        print("✅ Boto3 available")
    except ImportError:
        missing_deps.append("boto3")
    
    try:
        from bedrock_agentcore.runtime import BedrockAgentCoreApp
        print("✅ AgentCore SDK available")
    except ImportError:
        print("⚠️  AgentCore SDK not available (will run in standalone mode)")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements_working.txt")
        return False
    
    print("\n✅ All core dependencies available!")
    return True

async def quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running Quick Test...")
    
    try:
        # Import our system
        from generic_agentcore_system import GenericOrchestrator, CalculatorTool
        
        # Test orchestrator
        orchestrator = GenericOrchestrator()
        
        # Test basic functionality
        result = await orchestrator.execute_agent(
            agent_id="react_agent",
            input_data={"input": "What is 2 + 2?"}
        )
        
        if result.get("success"):
            print("✅ Basic agent test passed")
            print(f"Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"❌ Agent test failed: {result.get('error')}")
            return False
        
        # Test calculator tool
        calc_tool = CalculatorTool()
        calc_result = await calc_tool.execute(input_text="sqrt(16)")
        
        if calc_result.get("success"):
            print("✅ Calculator tool test passed")
            print(f"Result: {calc_result.get('result')}")
        else:
            print(f"❌ Calculator test failed: {calc_result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_usage():
    """Show usage instructions"""
    print("\n🚀 Generic AgentCore System - Quick Start")
    print("=" * 50)
    print("\n📋 Available Commands:")
    print("  python quick_start.py check     - Check dependencies")
    print("  python quick_start.py test      - Run quick test")
    print("  python quick_start.py console   - Start console mode")
    print("  python quick_start.py demo      - Run full demo")
    print("\n🔧 Main Application:")
    print("  python main_generic.py console  - Interactive mode")
    print("  python main_generic.py server   - Server mode")
    print("  python main_generic.py --help   - Full help")

async def run_demo():
    """Run a full demonstration"""
    print("\n🎭 Generic AgentCore Demo")
    print("=" * 30)
    
    try:
        from generic_agentcore_system import GenericOrchestrator
        from main_generic.py import GenericAgentCoreApp
        
        app = GenericAgentCoreApp()
        
        # Show available agents
        info = app.orchestrator.get_agent_info()
        print(f"\n📋 Available Agents ({info['total_agents']}):")
        for agent_id, agent_info in info['agents'].items():
            print(f"  • {agent_id}: {agent_info['type']} - {agent_info['tools_count']} tools")
        
        # Demo different capabilities
        demos = [
            ("Math Calculation", "react_agent", "Calculate the area of a circle with radius 5"),
            ("Text Processing", "react_agent", "Process this text: count\nHello world, this is a test message!"),
            ("Web Search", "research_assistant", "Search for information about Python programming"),
            ("Simple Chat", "chat_agent", "Hello! Tell me about artificial intelligence."),
        ]
        
        for demo_name, agent_id, query in demos:
            print(f"\n🎯 Demo: {demo_name}")
            print(f"Agent: {agent_id}")
            print(f"Query: {query}")
            print("Response:", end=" ")
            
            result = await app.orchestrator.execute_agent(
                agent_id=agent_id,
                input_data={"input": query}
            )
            
            if result.get("success"):
                response = result.get("response", "No response")
                print(f"{response[:150]}..." if len(response) > 150 else response)
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Small delay for readability
            await asyncio.sleep(1)
        
        print("\n✅ Demo completed! Try 'python main_generic.py console' for interactive mode.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements_working.txt")

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "check":
        check_dependencies()
    
    elif command == "test":
        if not check_dependencies():
            return
        
        print("\n🧪 Running system test...")
        result = asyncio.run(quick_test())
        
        if result:
            print("\n✅ All tests passed! System is ready.")
            print("Try: python main_generic.py console")
        else:
            print("\n❌ Tests failed. Check your setup.")
    
    elif command == "console":
        if not check_dependencies():
            return
        
        try:
            from main_generic import GenericAgentCoreApp
            app = GenericAgentCoreApp()
            asyncio.run(app.run_console_mode())
        except Exception as e:
            print(f"❌ Failed to start console: {e}")
    
    elif command == "demo":
        if not check_dependencies():
            return
        
        asyncio.run(run_demo())
    
    else:
        show_usage()

if __name__ == "__main__":
    main()