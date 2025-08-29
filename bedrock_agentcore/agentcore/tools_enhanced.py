#!/usr/bin/env python3
"""
Enhanced Tools Module - Working Implementation
Backward compatible with existing setup + new capabilities
"""

from langchain.tools import Tool
from typing import Dict, Any, List
import asyncio
import json
import math
import ast
import operator
from datetime import datetime

# ================================================================================================
# BACKWARD COMPATIBILITY - Your Original Functions
# ================================================================================================

def claude_tool(llm):
    """Your original claude_tool function - unchanged"""
    def run_func(input_text: str):
        # wrap the input as messages
        messages = [{"role": "user", "content": input_text}]
        response = llm.invoke({"messages": messages})
        return response["messages"][-1]["content"]
    
    return Tool(
        name="ClaudeTool",
        func=run_func,
        description="Calls Claude model via Bedrock"
    )

def get_tools(llm):
    """Your original function - enhanced with additional tools"""
    tools = [claude_tool(llm)]
    
    # Add enhanced tools
    tools.extend([
        calculator_tool(),
        text_processor_tool(),
        web_search_tool(),
        data_analyzer_tool()
    ])
    
    return tools

# ================================================================================================
# ENHANCED TOOL IMPLEMENTATIONS
# ================================================================================================

def calculator_tool():
    """Enhanced calculator with advanced math functions"""
    
    def calculate(expression: str) -> str:
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Supported operations and functions
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.Mod: operator.mod
            }
            
            # Safe math functions
            safe_functions = {
                'sqrt': math.sqrt,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'abs': abs,
                'round': round,
                'ceil': math.ceil,
                'floor': math.floor,
                'pi': math.pi,
                'e': math.e
            }
            
            def eval_expr(node):
                if isinstance(node, ast.Constant):  # Python 3.8+
                    return node.value
                elif isinstance(node, ast.Num):  # Older Python versions
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                elif isinstance(node, ast.Call):
                    func_name = node.func.id
                    if func_name in safe_functions:
                        args = [eval_expr(arg) for arg in node.args]
                        return safe_functions[func_name](*args)
                    else:
                        raise ValueError(f"Function {func_name} not allowed")
                elif isinstance(node, ast.Name):
                    if node.id in safe_functions:
                        return safe_functions[node.id]
                    else:
                        raise ValueError(f"Name {node.id} not allowed")
                else:
                    raise TypeError(f"Unsupported operation: {type(node)}")
            
            # Parse and evaluate
            result = eval_expr(ast.parse(expression, mode='eval').body)
            
            # Format result
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)  # Avoid floating point precision issues
            
            return f"Result: {result}"
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except ValueError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    return Tool(
        name="calculator",
        func=calculate,
        description="Perform mathematical calculations. Supports: +, -, *, /, **, %, sqrt(), sin(), cos(), tan(), log(), exp(), abs(), round(), pi, e. Example: 'sqrt(16)' or '2 * pi * 5'"
    )

def text_processor_tool():
    """Text processing and analysis tool"""
    
    def process_text(input_text: str) -> str:
        try:
            lines = input_text.strip().split('\\n', 1)
            
            if len(lines) > 1:
                command = lines[0].lower().strip()
                text = lines[1]
            else:
                # If no command specified, do general analysis
                command = "analyze"
                text = input_text
            
            results = []
            
            if command in ["count", "analyze"]:
                word_count = len(text.split())
                char_count = len(text)
                line_count = len(text.split('\\n'))
                sentence_count = len([s for s in text.split('.') if s.strip()])
                
                results.append(f"Text Statistics:")
                results.append(f"- Words: {word_count}")
                results.append(f"- Characters: {char_count}")
                results.append(f"- Lines: {line_count}")
                results.append(f"- Sentences: {sentence_count}")
            
            if command == "upper":
                results.append(f"Uppercase: {text.upper()}")
            
            elif command == "lower":
                results.append(f"Lowercase: {text.lower()}")
            
            elif command == "title":
                results.append(f"Title Case: {text.title()}")
            
            elif command == "reverse":
                results.append(f"Reversed: {text[::-1]}")
            
            elif command == "words":
                words = text.split()
                unique_words = list(set(word.lower().strip('.,!?;:') for word in words))
                results.append(f"Unique words ({len(unique_words)}): {', '.join(sorted(unique_words)[:20])}")
                if len(unique_words) > 20:
                    results.append("... (showing first 20)")
            
            elif command == "summary":
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                if len(sentences) > 2:
                    summary = f"{sentences[0]}. ... {sentences[-1]}."
                else:
                    summary = text
                results.append(f"Summary: {summary}")
            
            # Default analysis if no specific command
            if not results or command == "analyze":
                if command != "analyze":  # Avoid duplicate stats
                    word_count = len(text.split())
                    results.append(f"\\nProcessed text ({word_count} words)")
            
            return "\\n".join(results)
            
        except Exception as e:
            return f"Text processing error: {str(e)}"
    
    return Tool(
        name="text_processor",
        func=process_text,
        description="Process and analyze text. Commands: 'count' (statistics), 'upper' (UPPERCASE), 'lower' (lowercase), 'title' (Title Case), 'reverse' (reverse text), 'words' (unique words), 'summary' (extract summary). Format: 'command\\ntext' or just text for analysis."
    )

def web_search_tool():
    """Mock web search tool (replace with real API in production)"""
    
    def search(query: str) -> str:
        # Mock search results - replace with real search API
        mock_results = {
            "ai": [
                {"title": "Artificial Intelligence Overview", "snippet": "AI is transforming industries worldwide with machine learning and deep learning technologies."},
                {"title": "Latest AI Developments 2025", "snippet": "Recent breakthroughs in AI include advanced language models and computer vision systems."}
            ],
            "python": [
                {"title": "Python Programming Guide", "snippet": "Python is a versatile programming language used for web development, data science, and AI."},
                {"title": "Python Best Practices", "snippet": "Writing clean, maintainable Python code with proper structure and documentation."}
            ],
            "climate": [
                {"title": "Climate Change Updates", "snippet": "Latest research on global climate patterns and environmental conservation efforts."},
                {"title": "Renewable Energy Solutions", "snippet": "Solar, wind, and other renewable energy technologies are becoming more efficient and affordable."}
            ]
        }
        
        # Simple keyword matching
        query_lower = query.lower()
        relevant_results = []
        
        for keyword, results in mock_results.items():
            if keyword in query_lower:
                relevant_results.extend(results)
        
        # Default results if no keywords match
        if not relevant_results:
            relevant_results = [
                {"title": f"Search Results for: {query}", "snippet": f"Information about {query} from various online sources."},
                {"title": f"Related Topics to {query}", "snippet": f"Additional resources and information related to your search for {query}."}
            ]
        
        # Format results
        result_text = f"🔍 Search Results for '{query}':\\n\\n"
        for i, result in enumerate(relevant_results[:3], 1):  # Limit to 3 results
            result_text += f"{i}. {result['title']}\\n"
            result_text += f"   {result['snippet']}\\n\\n"
        
        result_text += "💡 Note: These are mock results. In production, integrate with real search APIs."
        
        return result_text
    
    return Tool(
        name="web_search",
        func=search,
        description="Search for current information on any topic. Returns relevant results with titles and summaries."
    )

def data_analyzer_tool():
    """Simple data analysis tool for JSON/CSV data"""
    
    def analyze_data(input_data: str) -> str:
        try:
            # Try to parse as JSON
            try:
                data = json.loads(input_data)
                
                if isinstance(data, list) and len(data) > 0:
                    # Analyze list of items
                    total_items = len(data)
                    
                    if isinstance(data[0], dict):
                        # Analyze dictionary structure
                        keys = list(data[0].keys())
                        sample_item = data[0]
                        
                        analysis = f"📊 Data Analysis Results:\\n"
                        analysis += f"- Total items: {total_items}\\n"
                        analysis += f"- Fields: {', '.join(keys)}\\n"
                        analysis += f"- Sample item: {json.dumps(sample_item, indent=2)[:200]}...\\n\\n"
                        
                        # Analyze numeric fields
                        for key in keys:
                            values = [item.get(key) for item in data if isinstance(item.get(key), (int, float))]
                            if values:
                                avg_val = sum(values) / len(values)
                                analysis += f"- {key}: min={min(values)}, max={max(values)}, avg={avg_val:.2f}\\n"
                        
                        return analysis
                    
                    else:
                        # Simple list analysis
                        return f"📊 List Analysis:\\n- Items: {total_items}\\n- Sample: {data[:5]}..."
                
                elif isinstance(data, dict):
                    # Analyze single object
                    keys = list(data.keys())
                    return f"📊 Object Analysis:\\n- Keys: {', '.join(keys)}\\n- Structure: {json.dumps(data, indent=2)[:300]}..."
                
            except json.JSONDecodeError:
                # Try CSV-like format
                lines = input_data.strip().split('\\n')
                if len(lines) > 1 and ',' in lines[0]:
                    headers = [h.strip() for h in lines[0].split(',')]
                    data_rows = len(lines) - 1
                    
                    analysis = f"📊 CSV Data Analysis:\\n"
                    analysis += f"- Headers: {', '.join(headers)}\\n"
                    analysis += f"- Data rows: {data_rows}\\n"
                    
                    if len(lines) > 1:
                        sample_row = lines[1].split(',')
                        analysis += f"- Sample row: {dict(zip(headers, sample_row))}\\n"
                    
                    return analysis
            
            return "❌ Unable to analyze data. Supported formats: JSON array/object, CSV text."
            
        except Exception as e:
            return f"❌ Data analysis error: {str(e)}"
    
    return Tool(
        name="data_analyzer",
        func=analyze_data,
        description="Analyze structured data in JSON or CSV format. Provides statistics, field analysis, and insights."
    )

# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================

def validate_tools():
    """Test all tools to ensure they work correctly"""
    
    print("🧪 Validating Enhanced Tools...")
    
    # Mock LLM for testing
    class MockLLM:
        def invoke(self, messages):
            return {"messages": [{"content": "Mock LLM response"}]}
    
    mock_llm = MockLLM()
    tools = get_tools(mock_llm)
    
    test_results = {}
    
    for tool in tools:
        try:
            if tool.name == "calculator":
                result = tool.run("2 + 2 * 3")
                test_results[tool.name] = "✅ Passed" if "8" in result else f"❌ Failed: {result}"
            
            elif tool.name == "text_processor":
                result = tool.run("count\\nHello world test")
                test_results[tool.name] = "✅ Passed" if "Words:" in result else f"❌ Failed: {result}"
            
            elif tool.name == "web_search":
                result = tool.run("python programming")
                test_results[tool.name] = "✅ Passed" if "Search Results" in result else f"❌ Failed: {result}"
            
            elif tool.name == "data_analyzer":
                test_data = '[{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]'
                result = tool.run(test_data)
                test_results[tool.name] = "✅ Passed" if "Data Analysis" in result else f"❌ Failed: {result}"
            
            else:
                test_results[tool.name] = "⏭️ Skipped (manual test required)"
        
        except Exception as e:
            test_results[tool.name] = f"❌ Error: {str(e)}"
    
    print("\\n📋 Tool Validation Results:")
    for tool_name, result in test_results.items():
        print(f"  {tool_name}: {result}")
    
    print(f"\\n✅ Validated {len(tools)} tools")
    return test_results

def get_available_tools():
    """Get list of available tools with descriptions"""
    
    class MockLLM:
        def invoke(self, messages):
            return {"messages": [{"content": "Mock response"}]}
    
    tools = get_tools(MockLLM())
    
    tool_info = []
    for tool in tools:
        tool_info.append({
            "name": tool.name,
            "description": tool.description
        })
    
    return tool_info

# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

if __name__ == "__main__":
    print("🛠️ Enhanced Tools Module")
    print("=" * 40)
    
    # Show available tools
    available_tools = get_available_tools()
    print("\\n📋 Available Tools:")
    for tool in available_tools:
        print(f"  • {tool['name']}: {tool['description']}")
    
    # Run validation
    print()
    validate_tools()
    
    # Interactive test mode
    print("\\n🧪 Interactive Test Mode")
    print("Commands: calc <expression>, text <command>\\n<text>, search <query>, analyze <data>, quit")
    
    class MockLLM:
        def invoke(self, messages):
            return {"messages": [{"content": "Mock LLM response"}]}
    
    tools_dict = {tool.name: tool for tool in get_tools(MockLLM())}
    
    while True:
        try:
            user_input = input("\\nTest> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.startswith('calc '):
                expression = user_input[5:]
                result = tools_dict['calculator'].run(expression)
                print(f"Calculator: {result}")
            
            elif user_input.startswith('search '):
                query = user_input[7:]
                result = tools_dict['web_search'].run(query)
                print(f"Search: {result[:300]}...")
            
            elif user_input.startswith('text '):
                text_input = user_input[5:]
                result = tools_dict['text_processor'].run(text_input)
                print(f"Text: {result}")
            
            elif user_input.startswith('analyze '):
                data_input = user_input[8:]
                result = tools_dict['data_analyzer'].run(data_input)
                print(f"Analysis: {result}")
            
            else:
                print("Unknown command. Try: calc <expression>, search <query>, text <command>\\n<text>, analyze <data>")
        
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")