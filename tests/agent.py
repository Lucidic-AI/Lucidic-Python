from typing import List, Dict
import openai
import os
import traceback
class Agent:
    def __init__(self):
        self.api_key = "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"
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def solve(self, prompt: str) -> Dict:
        try:
            # Step 1: Break down the problem
            steps = self._break_down_problem(prompt)
            
            # Step 2: Solve the problem
            solution = self._solve_problem(prompt, steps)
            
            # Step 3: Verify the solution
            is_correct = self._verify_solution(prompt, solution)
            
            # Step 4: Check units
            final_solution = self._check_units(solution)
            
            return {
                "steps": steps,
                "solution": solution,
                "is_correct": is_correct,
                "final_solution": final_solution
            }
             
        except Exception as e:
            print(f"Agent Error: {e}")
            traceback.print_exc(file=sys.stdout)
            return {"error": str(e)}
    
    def _break_down_problem(self, prompt: str) -> List[str]:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a problem decomposition expert. Your job is to:
                1. Break down complex problems into clear, logical steps.
                2. Identify key variables and their relationships.
                3. List any assumptions that need to be made.
                4. Return steps as a numbered list.
                Be concise and clear."""},
                {"role": "user", "content": f"Break down this problem into steps: {prompt}"}
            ]
        )
        
        steps = response.choices[0].message.content.split("\n")
        return [step.strip() for step in steps if step.strip()]
        
    def _solve_problem(self, prompt: str, steps: List[str]) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a problem-solving expert. Your job is to:
                1. Follow the given steps to solve the problem.
                2. Show your work clearly.
                3. Explain your reasoning.
                4. Provide a final answer with units.
                Be thorough but concise."""},
                {"role": "user", "content": f"Problem: {prompt}\nSteps to follow:\n" + "\n".join(steps)}
            ]
        )
        
        return response.choices[0].message.content
        
    def _verify_solution(self, prompt: str, solution: str) -> bool:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a solution verifier. Your job is to:
                1. Solve the problem independently.
                2. Compare your solution to the given solution.
                3. Be lenient - solutions within 10% are acceptable.
                4. Return "True" if correct, "False" if incorrect.
                Be thorough but forgiving of minor differences."""},
                {"role": "user", "content": f"Problem: {prompt}\nProposed solution:\n{solution}"}
            ]
        )
        
        return "true" in response.choices[0].message.content.strip().lower()
        
    def _check_units(self, solution: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are a units expert. Your job is to:
                1. Identify all units in the solution.
                2. Verify unit consistency.
                3. Convert to standard units if needed.
                4. Add units where missing.
                Be thorough with unit analysis."""},
                {"role": "user", "content": f"Check units in this solution:\n{solution}"}
            ]
        )
        
        return response.choices[0].message.content
