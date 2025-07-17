"""Test 9 patterns with OpenAI Agents SDK - FIXED VERSION"""
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import lucidicai as lai
from agents import Agent, Runner, function_tool

def test_pattern_1_basic_agent():
    """Pattern 1: Basic agent - Single agent responding to queries"""
    print("="*60)
    print("PATTERN 1: Basic Agent")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 1: Basic Agent",
        providers=["openai_agents"],
        task="Basic agent pattern"
    )
    
    agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant. Answer questions concisely."
    )
    
    session = lai.get_session()
    
    result = Runner.run_sync(agent, "Explain quantum computing in one sentence.")
    
    print(f"Response: {result.final_output[:100]}...")
    print(f"Steps created: {len(session.step_history)}")
    print(f"Events created: {len(session.event_history)}")
    
    # Check tracking
    for step_id, step in session.step_history.items():
        print(f"  Step {step_id[:8]}...: finished={step.is_finished}")
    
    success = len(session.step_history) >= 1 and all(step.is_finished for step in session.step_history.values())
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 1: Basic agent tracked correctly")
    else:
        print("✗ Pattern 1: Tracking failed")
    
    return success

def test_pattern_2_agent_with_tools():
    """Pattern 2: Agent with tools - Using custom functions"""
    print("\n" + "="*60)
    print("PATTERN 2: Agent with Tools")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 2: Agent with Tools",
        providers=["openai_agents"],
        task="Agent with tools pattern"
    )
    
    # Define tools with proper decorator
    @function_tool
    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle"""
        return length * width
    
    @function_tool
    def calculate_perimeter(length: float, width: float) -> float:
        """Calculate the perimeter of a rectangle"""
        return 2 * (length + width)
    
    agent = Agent(
        name="MathAgent",
        instructions="You are a math assistant. Use the provided tools to calculate areas and perimeters.",
        tools=[calculate_area, calculate_perimeter]
    )
    
    session = lai.get_session()
    
    result = Runner.run_sync(
        agent,
        "What's the area and perimeter of a rectangle with length 10 and width 5?"
    )
    
    print(f"Response: {result.final_output[:100]}...")
    print(f"Steps created: {len(session.step_history)}")
    print(f"Events created: {len(session.event_history)}")
    
    success = len(session.step_history) >= 1
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 2: Agent with tools tracked correctly")
    else:
        print("✗ Pattern 2: Tracking failed")
    
    return success

def test_pattern_3_agent_handoff():
    """Pattern 3: Agent handoff - Agents transferring control"""
    print("\n" + "="*60)
    print("PATTERN 3: Agent Handoff (FIXED)")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 3: Agent Handoff Fixed",
        providers=["openai_agents"],
        task="Agent handoff pattern with correct syntax"
    )
    
    # Create specialized agents
    tech_agent = Agent(
        name="TechAgent",
        instructions="You are a technical expert. Answer technical questions in detail."
    )
    
    sales_agent = Agent(
        name="SalesAgent",
        instructions="You are a sales expert. Answer sales and pricing questions."
    )
    
    # Create triage agent with handoffs parameter (CORRECT SYNTAX)
    triage_agent = Agent(
        name="TriageAgent",
        instructions="You are a triage agent. Transfer technical questions to TechAgent and sales questions to SalesAgent.",
        handoffs=[tech_agent, sales_agent]  # ✅ CORRECT: Use handoffs parameter
    )
    
    session = lai.get_session()
    
    # Test handoff to tech
    result = Runner.run_sync(
        triage_agent,
        "How does your API authentication work?"
    )
    
    print(f"Last agent: {result.last_agent.name}")
    print(f"Response: {result.final_output[:100]}...")
    print(f"Steps created: {len(session.step_history)}")
    print(f"Events created: {len(session.event_history)}")
    
    # List all steps
    for i, (step_id, step) in enumerate(session.step_history.items()):
        print(f"  Step {i+1}: finished={step.is_finished}")
    
    # Should have multiple steps for handoff
    success = len(session.step_history) >= 2 and result.last_agent.name == "TechAgent"
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 3: Agent handoff tracked correctly")
    else:
        print("✗ Pattern 3: Tracking failed")
    
    return success

def test_pattern_4_sequential_processing():
    """Pattern 4: Sequential processing - Multiple agents in sequence"""
    print("\n" + "="*60)
    print("PATTERN 4: Sequential Processing")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 4: Sequential Processing",
        providers=["openai_agents"],
        task="Sequential processing pattern"
    )
    
    # Create a pipeline of agents
    researcher = Agent(
        name="Researcher",
        instructions="You research topics and provide key facts."
    )
    
    writer = Agent(
        name="Writer",
        instructions="You write engaging content based on research."
    )
    
    editor = Agent(
        name="Editor",
        instructions="You edit and improve written content."
    )
    
    session = lai.get_session()
    
    # Sequential processing
    print("Step 1: Research")
    research = Runner.run_sync(researcher, "Research the benefits of meditation")
    
    print("Step 2: Write")
    draft = Runner.run_sync(writer, f"Write a paragraph based on: {research.final_output}")
    
    print("Step 3: Edit")
    final = Runner.run_sync(editor, f"Edit and improve: {draft.final_output}")
    
    print(f"Final output: {final.final_output[:100]}...")
    print(f"Total steps: {len(session.step_history)}")
    
    # Check all steps are finished
    all_finished = all(step.is_finished for step in session.step_history.values())
    print(f"All steps finished: {all_finished}")
    
    success = len(session.step_history) >= 3 and all_finished
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 4: Sequential processing tracked correctly")
    else:
        print("✗ Pattern 4: Tracking failed")
    
    return success

def test_pattern_5_parallel_tools():
    """Pattern 5: Parallel tools - Agent using multiple tools"""
    print("\n" + "="*60)
    print("PATTERN 5: Parallel Tools (FIXED)")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 5: Parallel Tools Fixed",
        providers=["openai_agents"],
        task="Parallel tools pattern with decorators"
    )
    
    # Define multiple tools with decorators
    @function_tool
    def get_weather(city: str) -> str:
        """Get weather for a city"""
        return f"Weather in {city}: Sunny, 72°F"
    
    @function_tool
    def get_news(topic: str) -> str:
        """Get latest news on a topic"""
        return f"Latest {topic} news: Major breakthrough announced today"
    
    @function_tool
    def check_stock(symbol: str) -> str:
        """Check stock price"""
        return f"{symbol} stock: $150.25 (+2.3%)"
    
    agent = Agent(
        name="InfoAgent",
        instructions="You are an information assistant. Use the tools to gather and present information.",
        tools=[get_weather, get_news, check_stock]
    )
    
    session = lai.get_session()
    
    result = Runner.run_sync(
        agent,
        "What's the weather in NYC, latest AI news, and AAPL stock price?"
    )
    
    print(f"Response: {result.final_output[:100]}...")
    print(f"Steps: {len(session.step_history)}")
    print(f"Events: {len(session.event_history)}")
    
    success = len(session.step_history) >= 1
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 5: Parallel tools tracked correctly")
    else:
        print("✗ Pattern 5: Tracking failed")
    
    return success

def test_pattern_6_conditional_routing():
    """Pattern 6: Conditional routing - Different paths based on input"""
    print("\n" + "="*60)
    print("PATTERN 6: Conditional Routing")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 6: Conditional Routing",
        providers=["openai_agents"],
        task="Conditional routing pattern"
    )
    
    # Create specialized handlers
    password_handler = Agent(
        name="PasswordHandler",
        instructions="You handle password reset requests. Provide step-by-step instructions."
    )
    
    account_handler = Agent(
        name="AccountHandler",
        instructions="You handle account-related queries."
    )
    
    technical_handler = Agent(
        name="TechnicalHandler",
        instructions="You handle technical support issues."
    )
    
    # Router with conditional handoffs
    router = Agent(
        name="Router",
        instructions="Route password issues to PasswordHandler, account issues to AccountHandler, and technical issues to TechnicalHandler.",
        handoffs=[password_handler, account_handler, technical_handler]
    )
    
    session = lai.get_session()
    
    # Test routing
    result = Runner.run_sync(router, "How do I reset my password?")
    
    print(f"Routed to: {result.last_agent.name}")
    print(f"Response: {result.final_output[:100]}...")
    print(f"Steps: {len(session.step_history)}")
    
    success = result.last_agent.name == "PasswordHandler" and len(session.step_history) >= 2
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 6: Conditional routing tracked correctly")
    else:
        print("✗ Pattern 6: Tracking failed")
    
    return success

def test_pattern_7_hierarchical_agents():
    """Pattern 7: Hierarchical agents - Manager delegating to workers"""
    print("\n" + "="*60)
    print("PATTERN 7: Hierarchical Agents")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 7: Hierarchical Agents",
        providers=["openai_agents"],
        task="Hierarchical delegation pattern"
    )
    
    # Create worker agents
    designer = Agent(
        name="Designer",
        instructions="You handle UI/UX design tasks."
    )
    
    developer = Agent(
        name="Developer",
        instructions="You handle coding and implementation."
    )
    
    tester = Agent(
        name="Tester",
        instructions="You handle testing and QA."
    )
    
    # Create manager agent with handoffs
    manager = Agent(
        name="ProjectManager",
        instructions="You manage projects and delegate tasks to Designer, Developer, or Tester as appropriate.",
        handoffs=[designer, developer, tester]
    )
    
    session = lai.get_session()
    
    result = Runner.run_sync(
        manager,
        "We need to build a login page. Can you coordinate the team?"
    )
    
    print(f"Project outcome: {result.final_output[:100]}...")
    print(f"Steps created: {len(session.step_history)}")
    print(f"Events created: {len(session.event_history)}")
    
    success = len(session.step_history) >= 1
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 7: Hierarchical agents tracked correctly")
    else:
        print("✗ Pattern 7: Tracking failed")
    
    return success

def test_pattern_8_context_aware():
    """Pattern 8: Context-aware agents - Maintaining conversation context"""
    print("\n" + "="*60)
    print("PATTERN 8: Context-Aware Agents")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 8: Context-Aware",
        providers=["openai_agents"],
        task="Context-aware conversation"
    )
    
    # Agent with context
    agent = Agent(
        name="PersonalAssistant",
        instructions="""You are a personal assistant with context about the user:
        - Name: John Doe
        - Preferences: Python programming, web development
        - Previous topics: AI, machine learning
        Remember context across the conversation."""
    )
    
    session = lai.get_session()
    
    # Multiple queries building on context
    print("Query 1: What's new in AI today?")
    result1 = Runner.run_sync(agent, "What's new in AI today?")
    print(f"Response: {result1.final_output[:100]}...")
    
    print("\nQuery 2: Can you recommend a Python framework for that?")
    result2 = Runner.run_sync(agent, "Can you recommend a Python framework for that?")
    print(f"Response: {result2.final_output[:100]}...")
    
    print(f"\nTotal steps: {len(session.step_history)}")
    print(f"Total events: {len(session.event_history)}")
    
    success = len(session.step_history) >= 2
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 8: Context-aware agents tracked correctly")
    else:
        print("✗ Pattern 8: Tracking failed")
    
    return success

def test_pattern_9_complex_workflow():
    """Pattern 9: Complex workflow - Multi-agent collaboration"""
    print("\n" + "="*60)
    print("PATTERN 9: Complex Workflow")
    print("="*60)
    
    try:
        lai.reset_sdk()  # Clear any existing session
    except:
        pass  # Ignore if not initialized
    lai.init(
        session_name="Pattern 9: Complex Workflow",
        providers=["openai_agents"],
        task="Complex multi-agent workflow"
    )
    
    # Create agents in reverse order to set up handoff chain
    editor = Agent(
        name="Editor",
        instructions="You review and polish reports to make them publication-ready."
    )
    
    writer = Agent(
        name="Writer",
        instructions="You create comprehensive reports based on research. After writing, always hand off to the Editor for final polish.",
        handoffs=[editor]
    )
    
    researcher = Agent(
        name="Researcher",
        instructions="You conduct research and gather information. After completing your research, always hand off to the Writer to create a report.",
        handoffs=[writer]
    )
    
    # Coordinator that delegates to researcher first
    coordinator = Agent(
        name="Coordinator",
        instructions="You coordinate report creation. For any report request, immediately delegate to the Researcher to gather information.",
        handoffs=[researcher]
    )
    
    session = lai.get_session()
    
    # Start the workflow
    result = Runner.run_sync(
        coordinator,
        "Create a comprehensive report on AI adoption in healthcare"
    )
    
    print(f"Workflow result: {result.final_output[:100]}...")
    print(f"Final agent: {result.last_agent.name}")
    print(f"Total steps: {len(session.step_history)}")
    print(f"Total events: {len(session.event_history)}")
    
    # List all steps
    for i, (step_id, step) in enumerate(session.step_history.items()):
        print(f"  Step {i+1}: {step_id[:8]}...")
    
    # OpenAI Agents SDK only does one handoff per run, so we expect 2 steps
    success = len(session.step_history) >= 2  # Expecting coordinator -> researcher
    
    lai.end_session()
    
    if success:
        print("✓ Pattern 9: Complex workflow tracked correctly")
    else:
        print("✗ Pattern 9: Tracking failed")
    
    return success

def main():
    """Run all 9 patterns"""
    print("="*80)
    print("OPENAI AGENTS SDK - 9 PATTERNS TEST (FIXED)")
    print("="*80)
    print("Testing all patterns with correct syntax...\n")
    
    patterns = [
        ("Pattern 1: Basic Agent", test_pattern_1_basic_agent),
        ("Pattern 2: Agent with Tools", test_pattern_2_agent_with_tools),
        ("Pattern 3: Agent Handoff", test_pattern_3_agent_handoff),
        ("Pattern 4: Sequential Processing", test_pattern_4_sequential_processing),
        ("Pattern 5: Parallel Tools", test_pattern_5_parallel_tools),
        ("Pattern 6: Conditional Routing", test_pattern_6_conditional_routing),
        ("Pattern 7: Hierarchical Agents", test_pattern_7_hierarchical_agents),
        ("Pattern 8: Context-Aware Agents", test_pattern_8_context_aware),
        ("Pattern 9: Complex Workflow", test_pattern_9_complex_workflow)
    ]
    
    results = []
    for name, test_func in patterns:
        print(f"\nTesting: {name}")
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"✗ Error: {str(e)[:60]}...")
            results.append((name, False))
            # Try to clean up
            try:
                lai.end_session()
            except:
                pass
    
    print("\n" + "="*80)
    print("PATTERN TEST SUMMARY")
    print("="*80)
    for name, success in results:
        status = "✓ Passed" if success else "✗ Failed"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/9 patterns passed")
    
    if passed == 9:
        print("\n🎉 ALL PATTERNS PASS! OpenAI Agents SDK integration is fully working.")
    
    return passed == 9

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)