from typing import List, Dict
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

class LangchainAgent:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        
        # Initialize models for different steps
        self.breakdown_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.solve_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.verify_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.units_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
    def solve(self, prompt: str) -> Dict:
        try:
            # Step 1: Break down the problem
            descriptions = self._break_down_problem(prompt)
            
            # Step 2: Solve the problem
            solution = self._solve_problem(prompt, descriptions)
            
            # Step 3: Verify the solution
            is_correct = self._verify_solution(prompt, solution)
            
            # Step 4: Check units
            final_solution = self._check_units(solution)
            
            return {
                "descriptions": descriptions,
                "solution": solution,
                "is_correct": is_correct,
                "final_solution": final_solution
            }
             
        except Exception as e:
            print(f"Agent Error: {e}")
            return {"error": str(e)}
    
    def _break_down_problem(self, prompt: str) -> List[str]:
        messages = [
            SystemMessage(content="""You are a problem decomposition expert. Your job is to:
            1. Break down complex problems into clear, logical descriptions.
            2. Identify key variables and their relationships.
            3. List any assumptions that need to be made.
            4. Return descriptions as a numbered list.
            Be concise and clear."""),
            HumanMessage(content=f"Break down this problem into descriptions: {prompt}")
        ]
        
        response = self.breakdown_llm(messages)
        descriptions = response.content.split("\n")
        return [description.strip() for description in descriptions if description.strip()]

    def _solve_problem(self, prompt: str, descriptions: List[str]) -> str:
        messages = [
            SystemMessage(content="""You are a problem-solving expert. Your job is to:
            1. Follow the given descriptions to solve the problem.
            2. Show your work clearly.
            3. Explain your reasoning.
            4. Provide a final result with units.
            Be thorough but concise."""),
            HumanMessage(content=f"Problem: {prompt}\nDescriptions to follow:\n" + "\n".join(descriptions))
        ]
        
        response = self.solve_llm(messages)
        return response.content
        
    def _verify_solution(self, prompt: str, solution: str) -> bool:
        messages = [
            SystemMessage(content="""You are a solution verifier. Your job is to:
            1. Solve the problem independently.
            2. Compare your solution to the given solution.
            3. Be lenient - solutions within 10% are acceptable.
            4. Return "True" if correct, "False" if incorrect.
            Be thorough but forgiving of minor differences."""),
            HumanMessage(content=f"Problem: {prompt}\nProposed solution:\n{solution}")
        ]
        
        response = self.verify_llm(messages)
        return "true" in response.content.strip().lower()
        
    def _check_units(self, solution: str) -> str:
        messages = [
            SystemMessage(content="""You are a units expert. Your job is to:
            1. Identify all units in the solution.
            2. Verify unit consistency.
            3. Convert to standard units if needed.
            4. Add units where missing.
            Be thorough with unit analysis."""),
            HumanMessage(content=f"Check units in this solution:\n{solution}")
        ]
        
        response = self.units_llm(messages)
        return response.content