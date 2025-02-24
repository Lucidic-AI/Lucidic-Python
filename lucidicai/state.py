from typing import Optional

class State:
    def __init__(self, description: Optional[str] = None):
        self.description = description or "state not provided"
        
    def __str__(self) -> str:
        return self.description
