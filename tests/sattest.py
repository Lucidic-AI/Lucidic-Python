import time
import sys
import os
import lucidicai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent

# Simple list of problems
PROBLEMS = [
    "If 3x + 5 = 20, what is the value of x?",
    "A rectangle has a length that is twice its width. If the perimeter is 36 units, what is the area of the rectangle?",
    "A train travels 150 miles at a constant speed of 60 miles per hour. How many minutes does the trip take?"
]

def main():
    agent = Agent()
    apikey = "RFxWd1F1.Jr4mLoh8liZXe4hFF17y95cYwwuvEeEu"
    agent_id = "14061d9c-4aac-4d54-a23a-313a692ed8b1"
    
    # Process each problem
    for i, question in enumerate(PROBLEMS, 1):
        print(f"\nSolving problem {i}:")
        print(f"Question: {question}")
        
        print("I am here")
        # Start new session
        lucidicai.init(
            lucidic_api_key=apikey,
            agent_id=agent_id,
            session_name=f"sat_problem_{i}",
            provider="openai"
        )
        
        try:
            # Solve the problem 
            solution = agent.solve(question)
            lucidicai.end_session(is_successful=True)
            
        except Exception as e:
            print(f"sattest Error: {e}")
            lucidicai.end_session(is_successful=False)
        
        time.sleep(1)  # Brief pause between problems

if __name__ == "__main__":
    main()