# Simple AgentCore Agents with Orchestration
# Step-by-step learning implementation

import asyncio
import json
import boto3
from datetime import datetime, timedelta
from typing import Dict, Any, List
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Initialize the AgentCore application
app = BedrockAgentCoreApp()

# ================================================================================================
# AGENT 1: SIMPLE CALCULATOR AGENT
# ================================================================================================

@app.entrypoint
async def calculator_agent(payload: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Simple calculator agent that performs basic mathematical operations
    
    Input: {"operation": "add", "numbers": [1, 2, 3]}
    Output: {"result": 6, "operation": "add", "timestamp": "..."}
    """
    
    operation = payload.get('operation', 'add')
    numbers = payload.get('numbers', [0])
    
    # Ping to maintain session health
    await context.ping(status="HEALTHY_BUSY", message=f"Calculating {operation}")
    
    try:
        if operation == 'add':
            result = sum(numbers)
        elif operation == 'subtract':
            result = numbers[0] - sum(numbers[1:]) if len(numbers) > 1 else numbers[0]
        elif operation == 'multiply':
            result = 1
            for num in numbers:
                result *= num
        elif operation == 'divide':
            result = numbers[0]
            for num in numbers[1:]:
                if num != 0:
                    result /= num
                else:
                    return {"error": "Division by zero", "agent_type": "calculator_agent"}
        elif operation == 'average':
            result = sum(numbers) / len(numbers) if numbers else 0
        else:
            return {"error": f"Unsupported operation: {operation}", "agent_type": "calculator_agent"}
        
        return {
            "result": result,
            "operation": operation,
            "numbers_processed": numbers,
            "timestamp": datetime.now().isoformat(),
            "agent_type": "calculator_agent"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "operation": operation,
            "agent_type": "calculator_agent"
        }


# ================================================================================================
# AGENT 2: TEXT PROCESSOR AGENT
# ================================================================================================

@app.entrypoint
async def text_processor_agent(payload: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Simple text processing agent
    
    Input: {"text": "Hello World", "operations": ["uppercase", "word_count"]}
    Output: {"processed_text": "HELLO WORLD", "word_count": 2, ...}
    """
    
    text = payload.get('text', '')
    operations = payload.get('operations', ['word_count'])
    
    await context.ping(status="HEALTHY_BUSY", message="Processing text")
    
    results = {
        "original_text": text,
        "operations_performed": operations,
        "results": {},
        "timestamp": datetime.now().isoformat(),
        "agent_type": "text_processor_agent"
    }
    
    try:
        for operation in operations:
            if operation == 'uppercase':
                results["results"]["uppercase"] = text.upper()
            elif operation == 'lowercase':
                results["results"]["lowercase"] = text.lower()
            elif operation == 'word_count':
                results["results"]["word_count"] = len(text.split())
            elif operation == 'character_count':
                results["results"]["character_count"] = len(text)
            elif operation == 'reverse':
                results["results"]["reverse"] = text[::-1]
            elif operation == 'title_case':
                results["results"]["title_case"] = text.title()
            else:
                results["results"][operation] = f"Unknown operation: {operation}"
        
        return results
        
    except Exception as e:
        return {
            "error": str(e),
            "original_text": text,
            "agent_type": "text_processor_agent"
        }


# ================================================================================================
# AGENT 3: DATA GENERATOR AGENT
# ================================================================================================

@app.entrypoint
async def data_generator_agent(payload: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Simple data generator agent that creates sample datasets
    
    Input: {"dataset_type": "customers", "count": 100}
    Output: {"dataset": [...], "metadata": {...}}
    """
    
    dataset_type = payload.get('dataset_type', 'customers')
    count = payload.get('count', 10)
    
    await context.ping(status="HEALTHY_BUSY", message=f"Generating {count} {dataset_type} records")
    
    import random
    
    try:
        if dataset_type == 'customers':
            # Generate customer data
            first_names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank']
            last_names = ['Smith', 'Johnson', 'Brown', 'Davis', 'Wilson', 'Taylor', 'Anderson', 'Thomas']
            domains = ['gmail.com', 'yahoo.com', 'company.com', 'business.org']
            
            dataset = []
            for i in range(count):
                first_name = random.choice(first_names)
                last_name = random.choice(last_names)
                customer = {
                    'id': f'CUST_{str(i+1).zfill(4)}',
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': f'{first_name.lower()}.{last_name.lower()}@{random.choice(domains)}',
                    'age': random.randint(18, 80),
                    'purchase_amount': round(random.uniform(10, 1000), 2),
                    'registration_date': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
                }
                dataset.append(customer)
                
        elif dataset_type == 'products':
            # Generate product data
            categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
            product_names = ['Widget', 'Gadget', 'Tool', 'Device', 'Item', 'Product', 'Thing']
            
            dataset = []
            for i in range(count):
                product = {
                    'id': f'PROD_{str(i+1).zfill(4)}',
                    'name': f'{random.choice(product_names)} {i+1}',
                    'category': random.choice(categories),
                    'price': round(random.uniform(5, 500), 2),
                    'stock': random.randint(0, 100),
                    'rating': round(random.uniform(1, 5), 1),
                    'created_date': (datetime.now() - timedelta(days=random.randint(0, 180))).strftime('%Y-%m-%d')
                }
                dataset.append(product)
                
        elif dataset_type == 'sales':
            # Generate sales data
            dataset = []
            for i in range(count):
                sale = {
                    'id': f'SALE_{str(i+1).zfill(4)}',
                    'customer_id': f'CUST_{random.randint(1, 100):04d}',
                    'product_id': f'PROD_{random.randint(1, 50):04d}',
                    'quantity': random.randint(1, 10),
                    'unit_price': round(random.uniform(10, 200), 2),
                    'total_amount': 0,  # Will calculate below
                    'sale_date': (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d')
                }
                sale['total_amount'] = round(sale['quantity'] * sale['unit_price'], 2)
                dataset.append(sale)
                
        else:
            return {
                "error": f"Unsupported dataset type: {dataset_type}",
                "supported_types": ["customers", "products", "sales"],
                "agent_type": "data_generator_agent"
            }
        
        # Calculate metadata
        metadata = {
            "dataset_type": dataset_type,
            "record_count": len(dataset),
            "generated_at": datetime.now().isoformat(),
            "sample_record": dataset[0] if dataset else None
        }
        
        if dataset_type in ['customers', 'sales']:
            if dataset_type == 'customers':
                total_purchase = sum([record['purchase_amount'] for record in dataset])
                avg_age = sum([record['age'] for record in dataset]) / len(dataset)
                metadata["total_purchase_amount"] = round(total_purchase, 2)
                metadata["average_age"] = round(avg_age, 1)
            elif dataset_type == 'sales':
                total_revenue = sum([record['total_amount'] for record in dataset])
                avg_quantity = sum([record['quantity'] for record in dataset]) / len(dataset)
                metadata["total_revenue"] = round(total_revenue, 2)
                metadata["average_quantity"] = round(avg_quantity, 1)
        
        return {
            "dataset": dataset,
            "metadata": metadata,
            "agent_type": "data_generator_agent"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "dataset_type": dataset_type,
            "agent_type": "data_generator_agent"
        }


# ================================================================================================
# AGENT 4: SIMPLE ANALYTICS AGENT
# ================================================================================================

@app.entrypoint
async def simple_analytics_agent(payload: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Simple analytics agent that processes data and generates insights
    
    Input: {"data": [...], "analysis_type": "summary"}
    Output: {"analytics": {...}, "insights": [...]}
    """
    
    data = payload.get('data', [])
    analysis_type = payload.get('analysis_type', 'summary')
    
    await context.ping(status="HEALTHY_BUSY", message=f"Analyzing {len(data)} records")
    
    if not data:
        return {
            "error": "No data provided for analysis",
            "agent_type": "simple_analytics_agent"
        }
    
    try:
        analytics = {
            "analysis_type": analysis_type,
            "record_count": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
        insights = []
        
        if analysis_type == 'summary':
            # Basic summary statistics
            if isinstance(data[0], dict):
                # Analyze dictionary data
                keys = data[0].keys()
                analytics["fields_analyzed"] = list(keys)
                
                for key in keys:
                    values = [record.get(key) for record in data if record.get(key) is not None]
                    
                    if values:
                        if isinstance(values[0], (int, float)):
                            # Numeric field analysis
                            analytics[f"{key}_stats"] = {
                                "min": min(values),
                                "max": max(values),
                                "avg": round(sum(values) / len(values), 2),
                                "count": len(values)
                            }
                            
                            if max(values) > sum(values) / len(values) * 2:
                                insights.append(f"High variability detected in {key}")
                                
                        elif isinstance(values[0], str):
                            # String field analysis
                            unique_values = list(set(values))
                            analytics[f"{key}_stats"] = {
                                "unique_count": len(unique_values),
                                "total_count": len(values),
                                "sample_values": unique_values[:5]
                            }
                            
                            if len(unique_values) < len(values) * 0.1:
                                insights.append(f"Low diversity in {key} - consider categorization")
            else:
                # Simple list analysis
                if all(isinstance(x, (int, float)) for x in data):
                    analytics["numeric_stats"] = {
                        "min": min(data),
                        "max": max(data),
                        "avg": round(sum(data) / len(data), 2),
                        "sum": sum(data)
                    }
                    
        elif analysis_type == 'trends':
            # Look for trends in the data
            if isinstance(data[0], dict) and 'date' in data[0]:
                # Date-based trend analysis
                dates = [record['date'] for record in data if 'date' in record]
                analytics["date_range"] = {
                    "earliest": min(dates),
                    "latest": max(dates),
                    "span_days": (datetime.strptime(max(dates), '%Y-%m-%d') - 
                                datetime.strptime(min(dates), '%Y-%m-%d')).days
                }
                insights.append("Time series data detected - suitable for trend analysis")
                
        elif analysis_type == 'quality':
            # Data quality analysis
            if isinstance(data[0], dict):
                total_records = len(data)
                quality_metrics = {}
                
                for key in data[0].keys():
                    null_count = sum(1 for record in data if record.get(key) is None or record.get(key) == '')
                    quality_metrics[key] = {
                        "completeness": round((total_records - null_count) / total_records, 3),
                        "missing_count": null_count
                    }
                    
                    if quality_metrics[key]["completeness"] < 0.9:
                        insights.append(f"Data quality issue: {key} has {quality_metrics[key]['completeness']:.1%} completeness")
                
                analytics["quality_metrics"] = quality_metrics
        
        # General insights
        if len(data) < 10:
            insights.append("Small dataset size - consider collecting more data for better analysis")
        elif len(data) > 1000:
            insights.append("Large dataset detected - suitable for advanced analytics")
        
        return {
            "analytics": analytics,
            "insights": insights,
            "agent_type": "simple_analytics_agent"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "analysis_type": analysis_type,
            "agent_type": "simple_analytics_agent"
        }


# ================================================================================================
# SIMPLE ORCHESTRATOR AGENT
# ================================================================================================

@app.entrypoint
async def simple_orchestrator(payload: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Simple orchestrator that coordinates multiple agents in sequence
    
    Input: {
        "workflow": [
            {"agent": "data_generator", "params": {...}},
            {"agent": "analytics", "params": {...}}
        ]
    }
    """
    
    workflow = payload.get('workflow', [])
    orchestration_id = f"simple_orch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    orchestration_results = {
        "orchestration_id": orchestration_id,
        "timestamp": datetime.now().isoformat(),
        "workflow_steps": len(workflow),
        "step_results": [],
        "overall_status": "in_progress"
    }
    
    await context.ping(status="HEALTHY_BUSY", message=f"Starting orchestration with {len(workflow)} steps")
    
    try:
        for step_index, step in enumerate(workflow):
            step_start_time = datetime.now()
            agent_type = step.get('agent')
            params = step.get('params', {})
            
            await context.ping(status="HEALTHY_BUSY", message=f"Executing step {step_index + 1}: {agent_type}")
            
            # Execute the appropriate agent based on type
            if agent_type == 'calculator':
                result = await calculator_agent(params, context)
            elif agent_type == 'text_processor':
                result = await text_processor_agent(params, context)
            elif agent_type == 'data_generator':
                result = await data_generator_agent(params, context)
            elif agent_type == 'analytics':
                result = await simple_analytics_agent(params, context)
            else:
                result = {"error": f"Unknown agent type: {agent_type}"}
            
            step_duration = (datetime.now() - step_start_time).total_seconds()
            
            step_result = {
                "step_index": step_index + 1,
                "agent_type": agent_type,
                "status": "success" if "error" not in result else "failed",
                "duration_seconds": step_duration,
                "result": result
            }
            
            orchestration_results["step_results"].append(step_result)
            
            # If this step failed, we might want to continue or stop
            if "error" in result:
                orchestration_results["overall_status"] = "failed_with_errors"
                # For now, continue with remaining steps
        
        # Determine overall status
        successful_steps = len([step for step in orchestration_results["step_results"] 
                               if step["status"] == "success"])
        total_steps = len(orchestration_results["step_results"])
        
        if successful_steps == total_steps:
            orchestration_results["overall_status"] = "completed_successfully"
        elif successful_steps > 0:
            orchestration_results["overall_status"] = "partially_completed"
        else:
            orchestration_results["overall_status"] = "failed"
        
        # Add summary
        orchestration_results["summary"] = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": total_steps - successful_steps,
            "total_duration_seconds": sum([step["duration_seconds"] for step in orchestration_results["step_results"]]),
            "completion_time": datetime.now().isoformat()
        }
        
        orchestration_results["agent_type"] = "simple_orchestrator"
        
        return orchestration_results
        
    except Exception as e:
        orchestration_results["overall_status"] = "orchestration_failed"
        orchestration_results["error"] = str(e)
        return orchestration_results


# ================================================================================================
# WORKFLOW TEMPLATES
# ================================================================================================

def get_sample_workflows():
    """
    Predefined workflow templates for common use cases
    """
    
    workflows = {
        "data_processing_workflow": {
            "description": "Generate data, then analyze it",
            "workflow": [
                {
                    "agent": "data_generator",
                    "params": {
                        "dataset_type": "customers",
                        "count": 50
                    }
                },
                {
                    "agent": "analytics",
                    "params": {
                        "data": "{{previous_result.dataset}}",  # Placeholder for chaining
                        "analysis_type": "summary"
                    }
                }
            ]
        },
        
        "text_and_math_workflow": {
            "description": "Process text and do calculations",
            "workflow": [
                {
                    "agent": "text_processor",
                    "params": {
                        "text": "Hello AgentCore World",
                        "operations": ["uppercase", "word_count", "character_count"]
                    }
                },
                {
                    "agent": "calculator",
                    "params": {
                        "operation": "multiply",
                        "numbers": [10, 20, 3]
                    }
                }
            ]
        },
        
        "comprehensive_workflow": {
            "description": "Multi-step workflow using all agents",
            "workflow": [
                {
                    "agent": "data_generator",
                    "params": {
                        "dataset_type": "sales",
                        "count": 25
                    }
                },
                {
                    "agent": "analytics",
                    "params": {
                        "data": "{{previous_result.dataset}}",
                        "analysis_type": "summary"
                    }
                },
                {
                    "agent": "text_processor", 
                    "params": {
                        "text": "Analysis Complete: Data Quality Check Passed",
                        "operations": ["uppercase", "word_count"]
                    }
                },
                {
                    "agent": "calculator",
                    "params": {
                        "operation": "average",
                        "numbers": [95, 87, 92, 89, 94]  # Quality scores
                    }
                }
            ]
        }
    }
    
    return workflows


# ================================================================================================
# TESTING AND DEPLOYMENT FUNCTIONS
# ================================================================================================

async def test_individual_agents():
    """Test each agent individually"""
    
    print("🧪 Testing Individual Agents")
    print("=" * 40)
    
    # Mock context for testing
    class MockContext:
        async def ping(self, status, message):
            print(f"🔄 {message}")
    
    context = MockContext()
    
    # Test Calculator Agent
    print("\n1️⃣ Testing Calculator Agent")
    calc_result = await calculator_agent({
        "operation": "add",
        "numbers": [10, 20, 30]
    }, context)
    print(f"Result: {calc_result}")
    
    # Test Text Processor Agent
    print("\n2️⃣ Testing Text Processor Agent")
    text_result = await text_processor_agent({
        "text": "Amazon Bedrock AgentCore",
        "operations": ["uppercase", "word_count", "reverse"]
    }, context)
    print(f"Result: {text_result}")
    
    # Test Data Generator Agent
    print("\n3️⃣ Testing Data Generator Agent")
    data_result = await data_generator_agent({
        "dataset_type": "customers",
        "count": 5
    }, context)
    print(f"Generated {len(data_result['dataset'])} records")
    print(f"Sample record: {data_result['dataset'][0]}")
    
    # Test Analytics Agent
    print("\n4️⃣ Testing Analytics Agent")
    analytics_result = await simple_analytics_agent({
        "data": data_result['dataset'],
        "analysis_type": "summary"
    }, context)
    print(f"Analytics: {analytics_result['analytics']}")
    print(f"Insights: {analytics_result['insights']}")
    
    return {
        "calculator": calc_result,
        "text_processor": text_result, 
        "data_generator": data_result,
        "analytics": analytics_result
    }


async def test_orchestration():
    """Test the orchestration capabilities"""
    
    print("\n🔀 Testing Orchestration")
    print("=" * 40)
    
    class MockContext:
        async def ping(self, status, message):
            print(f"🔄 {message}")
    
    context = MockContext()
    
    # Test simple workflow
    simple_workflow = {
        "workflow": [
            {
                "agent": "data_generator",
                "params": {
                    "dataset_type": "products",
                    "count": 10
                }
            },
            {
                "agent": "calculator",
                "params": {
                    "operation": "average",
                    "numbers": [100, 200, 150, 300, 250]
                }
            },
            {
                "agent": "text_processor",
                "params": {
                    "text": "Orchestration Test Complete",
                    "operations": ["title_case", "word_count"]
                }
            }
        ]
    }
    
    orchestration_result = await simple_orchestrator(simple_workflow, context)
    
    print(f"\nOrchestration Status: {orchestration_result['overall_status']}")
    print(f"Steps Completed: {orchestration_result['summary']['successful_steps']}/{orchestration_result['summary']['total_steps']}")
    print(f"Total Duration: {orchestration_result['summary']['total_duration_seconds']:.2f} seconds")
    
    return orchestration_result


def generate_deployment_config():
    """Generate deployment configuration for AgentCore"""
    
    config = {
        "agentcore_config": {
            "entry_point": "simple_agentcore_system.py",
            "runtime_settings": {
                "timeout_hours": 1,
                "memory_mb": 512,
                "environment_variables": {
                    "AGENT_ENVIRONMENT": "development"
                }
            },
            "agents": [
                {
                    "name": "calculator_agent",
                    "description": "Performs basic mathematical operations",
                    "entry_point": "calculator_agent"
                },
                {
                    "name": "text_processor_agent", 
                    "description": "Processes and transforms text",
                    "entry_point": "text_processor_agent"
                },
                {
                    "name": "data_generator_agent",
                    "description": "Generates sample datasets",
                    "entry_point": "data_generator_agent"
                },
                {
                    "name": "simple_analytics_agent",
                    "description": "Performs basic data analysis",
                    "entry_point": "simple_analytics_agent"
                },
                {
                    "name": "simple_orchestrator",
                    "description": "Coordinates multiple agents",
                    "entry_point": "simple_orchestrator"
                }
            ]
        }
    }
    
    return config


# ================================================================================================
# MAIN EXECUTION
# ================================================================================================

async def main():
    """Main function to test the simple agent system"""
    
    print("🤖 Simple AgentCore System with Orchestration")
    print("=" * 50)
    
    try:
        # Test individual agents
        individual_results = await test_individual_agents()
        
        # Test orchestration
        orchestration_result = await test_orchestration()
        
        # Show available workflows
        workflows = get_sample_workflows()
        print(f"\n📋 Available Workflow Templates:")
        for name, workflow in workflows.items():
            print(f"  • {name}: {workflow['description']}")
        
        # Generate deployment config
        deployment_config = generate_deployment_config()
        
        print(f"\n✅ Simple AgentCore system tested successfully!")
        print(f"🚀 Ready for deployment to AgentCore Runtime")
        
        return {
            "individual_tests": individual_results,
            "orchestration_test": orchestration_result,
            "deployment_config": deployment_config
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())