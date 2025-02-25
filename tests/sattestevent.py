import time
import sys
import os
import lucidicai
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# Simple list of problems
PROBLEMS = [
    "If 3x + 5 = 20, what is the value of x?",
    "A rectangle has a length that is twice its width. If the perimeter is 36 units, what is the area of the rectangle?",
    "A train travels 150 miles at a constant speed of 60 miles per hour. How many minutes does the trip take?",
    "Solve for y: 2y - 7 = 15.",
    "The sum of three consecutive even integers is 48. What are the integers?",
    "A right triangle has legs of length 6 and 8. What is the length of the hypotenuse?"
 ]


def main():
    agent = Agent()
    load_dotenv()  # Load environment variables from .env file
    apikey = os.getenv("LUCIDIC_API_KEY")
    if not apikey:
        raise ValueError("LUCIDIC_API_KEY environment variable is not set")
    
    agent_id = os.getenv("AGENT_ID")
    if not agent_id:
        raise ValueError("AGENT_ID environment variable is not set")

    lucidicai.init(
        lucidic_api_key=apikey,
        agent_id=agent_id,
        session_name="sat_problem_set",
        provider="openai"
    )

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
            is_successful=True,
            state="completed",
            action=f"solved:"
        )
        
        time.sleep(1) 

    lucidicai.Client().session.print_all()
    lucidicai.end_session(is_successful=True)

if __name__ == "__main__":
    main()