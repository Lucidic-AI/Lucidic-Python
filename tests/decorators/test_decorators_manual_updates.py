"""Example demonstrating immutable decorators (no manual updates)."""
from dotenv import load_dotenv
import lucidicai as lai
import time

# Load environment variables
load_dotenv()

# Initialize Lucidic
lai.init(
    session_name="Manual Updates Demo",
    providers=[],
    task="Demonstrate manual step and event updates within decorators"
)


@lai.event(
    description="Starting long process: Process data in stages"
)
def multi_stage_processor(data: dict, stages: int = 3) -> dict:
    """Process data across multiple stages; single event emitted at completion."""
    
    results = {}
    
    for i in range(stages):
        print(f"\nStage {i+1}/{stages}")
        # Simulate processing
        time.sleep(0.5)
        results[f"stage_{i+1}"] = f"Processed {data.get('input', 'data')} in stage {i+1}"
    
    return results


@lai.event(
    description="Initial calculation",
    model="calculator-v1",
    cost_added=0.001
)
def adaptive_calculator(operation: str, a: float, b: float) -> float:
    """Perform calculation; errors are captured by decorator and emitted once."""
    
    # Update based on the operation type
    if operation == "divide" and b == 0:
        raise ValueError("Division by zero")
    
    # Perform calculation
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
        "power": lambda x, y: x ** y
    }
    
    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")
    
    # Note: Complex operation metadata would be reflected in the single event's payload,
    # but we avoid manual updates; decorator emits one immutable event at completion.
    
    result = operations[operation](a, b)
    
    return result


@lai.event(description="Data validation and processing")
def validate_and_process(data: dict) -> dict:
    """Nested events without manual updates; outer event captures overall result."""
    
    # Validation phase
    @lai.event(description="Validating input data")
    def validate(input_data: dict) -> bool:
        required_fields = ["name", "value", "type"]
        missing = [f for f in required_fields if f not in input_data]
        
        return not missing
    
    # Processing phase
    @lai.event(description="Processing data", model="processor-v1")
    def process(input_data: dict) -> dict:
        data_type = input_data.get("type", "unknown")
        # Process based on type
        if data_type == "numeric":
            result = {"processed_value": input_data["value"] * 2}
        elif data_type == "text":
            result = {"processed_value": input_data["value"].upper()}
        else:
            result = {"processed_value": str(input_data["value"])}
        
        return result
    
    # Execute validation
    is_valid = validate(data)
    
    if not is_valid:
        return {"status": "failed", "reason": "validation_error"}
    
    # Process the data
    result = process(data)
    
    return {"status": "success", "data": result}


def main():
    """Run demonstrations of immutable decorator behavior."""
    
    print("=== Immutable Decorators Demo ===\n")
    
    # Demo 1: Multi-stage processing with progress updates
    print("1. Multi-stage Processing:")
    stage_result = multi_stage_processor({"input": "test_data"}, stages=4)
    print(f"Result: {stage_result}\n")
    
    # Demo 2: Adaptive calculator with conditional updates
    print("2. Adaptive Calculator:")
    
    # Simple operation
    result1 = adaptive_calculator("add", 10, 5)
    print(f"10 + 5 = {result1}")
    
    # Complex operation (triggers different model/cost)
    result2 = adaptive_calculator("power", 2, 8)
    print(f"2 ^ 8 = {result2}")
    
    # Error case
    try:
        result3 = adaptive_calculator("divide", 10, 0)
    except ValueError as e:
        print(f"Error caught: {e}")
    
    print()
    
    # Demo 3: Validation workflow
    print("3. Validation Workflow:")
    
    # Valid data
    valid_data = {"name": "test", "value": 42, "type": "numeric"}
    result_valid = validate_and_process(valid_data)
    print(f"Valid data result: {result_valid}")
    
    # Invalid data (missing fields)
    invalid_data = {"name": "test"}
    result_invalid = validate_and_process(invalid_data)
    print(f"Invalid data result: {result_invalid}")
    
    # End session
    lai.end_session()
    
    print("\n=== Demo Completed ===")
    print("\nKey takeaways:")
    print("- Decorators emit a single immutable event at completion")
    print("- Nested decorators create properly nested events via context propagation")
    print("- No manual update APIs are needed in the immutable model")


if __name__ == "__main__":
    main()