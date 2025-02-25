import time
import os
from dotenv import load_dotenv
import lucidicai
from enhanced_langchain_agent import LangchainAgent
from lucidicai.langchain import LucidicLangchainHandler

# Test cases for different Langchain features
TEST_CASES = [
    {
        "name": "basic_chat",
        "type": "chat",
        "description": "Testing basic chat model responses",
        "input": "Explain what makes a good test case."
    },
    {
        "name": "analysis_chain",
        "type": "analysis",
        "description": "Testing analysis chain with complex topic",
        "input": "The impact of artificial intelligence on modern software development"
    },
    {
        "name": "memory_conversation",
        "type": "memory",
        "description": "Testing conversation memory",
        "input": [
            "What is machine learning?",
            "How does it relate to AI?",
            "Can you give some practical examples?"
        ]
    }
]

def run_tests():
    """Run comprehensive tests of Langchain integration"""
    
    print("\nStarting Langchain Integration Tests...")
    
    # Setup
    load_dotenv()
    lucidic_api_key = os.getenv("LUCIDIC_API_KEY")
    agent_id = os.getenv("AGENT_ID")
    
    if not all([lucidic_api_key, agent_id]):
        raise ValueError("Missing required environment variables")

    print("\nInitializing Lucidic session...")
    # Initialize Lucidic
    session = lucidicai.init(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name="langchain_integration_test",
        provider="langchain"
    )

    print("\nCreating and configuring agent...")
    # Create agent
    agent = LangchainAgent()
    
    # Create handler directly from langchain module
    handler = LucidicLangchainHandler(lucidicai.Client())
    
    # Attach handler to all models
    print("\nAttaching handlers...")
    handler.attach_to_llms(agent.chat_model)
    handler.attach_to_llms(agent.analysis_model)
    handler.attach_to_llms(agent.summary_model)

    results_summary = {
        "total_tests": len(TEST_CASES),
        "successful_tests": 0,
        "failed_tests": 0,
        "errors": []
    }

    print("\nRunning test cases...")
    # Process each test case
    for test_case in TEST_CASES:
        print(f"\n{'='*50}")
        print(f"Running test: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"{'='*50}")
        
        lucidicai.create_step(
            state=f"Preparing to run {test_case['name']}",
            action=f"Initializing {test_case['name']} test",
            goal=test_case['description']
        )

        try:
            # Run appropriate test based on type
            if test_case['type'] == 'chat':
                print(f"\nTesting chat with prompt: {test_case['input'][:50]}...")
                lucidicai.update_step(
                    state=f"Running chat test",
                    action=f"Sending prompt to chat model"
                )
                result = agent.test_chat(test_case['input'])
                success = result['success']
                
            elif test_case['type'] == 'analysis':
                print(f"\nTesting analysis with topic: {test_case['input'][:50]}...")
                lucidicai.update_step(
                    state=f"Running analysis test",
                    action=f"Sending topic to analysis chain"
                )
                result = agent.test_analysis(test_case['input'])
                success = result['success']
                
            elif test_case['type'] == 'memory':
                print(f"\nTesting memory with {len(test_case['input'])} messages...")
                lucidicai.update_step(
                    state=f"Running memory test",
                    action=f"Processing conversation with {len(test_case['input'])} messages"
                )
                results = agent.test_memory(test_case['input'])
                success = all(r['success'] for r in results)
                result = {
                    'success': success,
                    'responses': results
                }
                
            # Update results summary
            if success:
                results_summary['successful_tests'] += 1
            else:
                results_summary['failed_tests'] += 1
                results_summary['errors'].append({
                    'test_case': test_case['name'],
                    'error': result.get('error', 'Unknown error')
                })
            
            # Complete step
            lucidicai.finish_step(
                is_successful=success,
                state="completed" if success else "failed",
                action=f"Completed {test_case['name']} - {'Success' if success else 'Failed'}"
            )

            # Print results
            print(f"\nResults for {test_case['name']}:")
            if test_case['type'] == 'memory':
                for i, response in enumerate(result['responses'], 1):
                    print(f"Message {i}: {'Success' if response['success'] else 'Failed'}")
                    if response['success']:
                        print(f"Response: {response['response'][:100]}...")
                    else:
                        print(f"Error: {response.get('error', 'Unknown error')}")
            else:
                print(f"Status: {'Success' if success else 'Failed'}")
                if success:
                    print(f"Response: {result['response'][:100]}...")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error in test case {test_case['name']}: {e}")
            results_summary['failed_tests'] += 1
            results_summary['errors'].append({
                'test_case': test_case['name'],
                'error': str(e)
            })
            
            lucidicai.finish_step(
                is_successful=False,
                state="error",
                action=f"Error in {test_case['name']}: {str(e)}"
            )

        # Clear memory between tests
        agent.clear_memory()
        time.sleep(2)  # Pause between tests

    # Print final summary
    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    print(f"Total tests: {results_summary['total_tests']}")
    print(f"Successful tests: {results_summary['successful_tests']}")
    print(f"Failed tests: {results_summary['failed_tests']}")
    if results_summary['errors']:
        print("\nErrors encountered:")
        for error in results_summary['errors']:
            print(f"- {error['test_case']}: {error['error']}")

    # Print detailed session information
    print("\n" + "="*50)
    print("Detailed Session Information:")
    print("="*50)
    lucidicai.Client().session.print_all()
    
    # End session
    print("\nEnding session...")
    lucidicai.end_session(is_successful=(results_summary['failed_tests'] == 0))

if __name__ == "__main__":
    run_tests()