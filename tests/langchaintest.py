import time
import os
from dotenv import load_dotenv
import lucidicai
from langchainagent import LangchainAgent

# Test problems
PROBLEMS = [
    "If 3x + 5 = 20, what is the value of x?",
    "A rectangle has a length that is twice its width. If the perimeter is 36 units, what is the area of the rectangle?",
    "A train travels 150 miles at a constant speed of 60 miles per hour. How many minutes does the trip take?",
    "Solve for y: 2y - 7 = 15.",
    "The sum of three consecutive even integers is 48. What are the integers?",
    "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse?"
]

def main():
    load_dotenv()
    
    # Get API keys
    lucidic_api_key = os.getenv("LUCIDIC_API_KEY")
    if not lucidic_api_key:
        raise ValueError("LUCIDIC_API_KEY environment variable is not set")
        
    agent_id = os.getenv("AGENT_ID")
    if not agent_id:
        raise ValueError("AGENT_ID environment variable is not set")

    # Initialize Lucidic with Langchain support
    session = lucidicai.init(
        lucidic_api_key=lucidic_api_key,
        agent_id=agent_id,
        session_name="langchain_sat_problems",
        provider="langchain"  # This will set up all the needed handlers
    )

    # Create agent
    agent = LangchainAgent()
    
    # Get the Lucidic handler and attach it to the agent
    handler = lucidicai.LucidicLangchainHandler(lucidicai.Client())
    handler.attach_to_llms(agent)

    # Process each problem as a step
    for i, question in enumerate(PROBLEMS, 1):
        print(f"\nSolving problem {i}:")
        print(f"Question: {question}")
        
        lucidicai.create_step(
            state="started",
            action="solving math problem",
            goal=f"Solve problem {i}: {question}"
        )
        
        solution = agent.solve(question)
        
        lucidicai.finish_step(
            is_successful="error" not in solution,
            state="completed",
            action=f"solved: {solution.get('final_solution', solution.get('error', 'No solution'))}"
        )
        
        time.sleep(1)

    lucidicai.Client().session.print_all()
    lucidicai.end_session(is_successful=True)

if __name__ == "__main__":
    main()