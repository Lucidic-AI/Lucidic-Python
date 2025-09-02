"""Minimal example showing the simplest usage of Lucidic decorators."""
import os
from dotenv import load_dotenv
import lucidicai as lai

# Load environment variables
load_dotenv()

# Initialize Lucidic
lai.init(session_name="Minimal Decorator Example", providers=[])

# Simple event decorator example (single immutable event)
@lai.event(description="Starting calculation: Perform math operation (Get result)")
def calculate(x: int, y: int) -> int:
    """Simple calculation wrapped in a step."""
    print(f"Calculating {x} + {y}")
    return x + y

# Simple event decorator example
@lai.event(description="Double a number")
def double(n: int) -> int:
    """Double a number - tracked as an event."""
    return n * 2

# Using event with nested events together
@lai.event(description="Processing: Transform data")
def process_data(value: int) -> dict:
    """Process data with nested event tracking."""
    
    # This creates an event inside the step
    doubled = double(value)
    
    # Another event with auto-generated description
    @lai.event()
    def square(n: int) -> int:
        return n ** 2
    
    squared = square(value)
    
    return {
        'original': value,
        'doubled': doubled,
        'squared': squared
    }

# Run examples
print("=== Running Examples ===\n")

# Example 1: Simple step
result1 = calculate(5, 3)
print(f"Calculate result: {result1}\n")

# Example 2: Simple event
result2 = double(10)
print(f"Double result: {result2}\n")

# Example 3: Nested usage
result3 = process_data(7)
print(f"Process result: {result3}\n")

# End session
lai.end_session(is_successful=True)
print("Session completed!")