from typing import Optional

class Action:
    def __init__(self, description: Optional[str] = None):
        self.description = description or "action not provided"
        
    def __str__(self) -> str:
        return self.description
