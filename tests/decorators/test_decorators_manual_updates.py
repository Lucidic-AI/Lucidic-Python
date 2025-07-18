"""Example demonstrating manual updates within decorated functions."""
import os
from dotenv import load_dotenv
import lucidicai as lai
from lucidicai.decorators import get_decorator_step, get_decorator_event
import time

# Load environment variables
load_dotenv()

# Initialize Lucidic
lai.init(
    session_name="Manual Updates Demo",
    providers=[],
    task="Demonstrate manual step and event updates within decorators"
)


@lai.step(
    state="Starting long process",
    action="Process data in stages",
    goal="Complete multi-stage processing"
)
def multi_stage_processor(data: dict, stages: int = 3) -> dict:
    """Demonstrates updating step progress during execution."""
    
    # Get the current step ID to manually update it
    step_id = get_decorator_step()
    print(f"Step ID: {step_id}")
    
    results = {}
    
    for i in range(stages):
        print(f"\nStage {i+1}/{stages}")
        
        # Update step progress
        progress = (i + 1) / stages
        lai.update_step(
            step_id=step_id,
            state=f"Processing stage {i+1} of {stages}",
            eval_score=progress,
            eval_description=f"Completed {i+1}/{stages} stages ({progress*100:.0f}%)"
        )
        
        # Simulate processing
        time.sleep(0.5)
        results[f"stage_{i+1}"] = f"Processed {data.get('input', 'data')} in stage {i+1}"
    
    # Final update before completion
    lai.update_step(
        step_id=step_id,
        state="All stages completed",
        eval_score=1.0,
        eval_description="Successfully processed all stages"
    )
    
    return results


@lai.event(
    description="Initial calculation",
    model="calculator-v1",
    cost_added=0.001
)
def adaptive_calculator(operation: str, a: float, b: float) -> float:
    """Demonstrates updating event based on runtime conditions."""
    
    # Get current event ID
    event_id = get_decorator_event()
    print(f"Event ID: {event_id}")
    
    # Update based on the operation type
    if operation == "divide" and b == 0:
        # Update event for error case
        lai.update_event(
            event_id=event_id,
            description="Division by zero attempted",
            result="Error: Cannot divide by zero",
            model="error-handler"
        )
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
        lai.update_event(
            event_id=event_id,
            description=f"Unknown operation: {operation}",
            result="Error: Operation not supported"
        )
        raise ValueError(f"Unknown operation: {operation}")
    
    # Complex operations cost more
    if operation in ["power", "divide"]:
        lai.update_event(
            event_id=event_id,
            description=f"Complex calculation: {operation}",
            model="calculator-v2-advanced",
            cost_added=0.005  # Higher cost for complex operations
        )
    
    result = operations[operation](a, b)
    
    # Custom result formatting based on operation
    lai.update_event(
        event_id=event_id,
        result=f"{a} {operation} {b} = {result}"
    )
    
    return result


@lai.step(state="Data validation", action="Validate and process")
def validate_and_process(data: dict) -> dict:
    """Demonstrates nested decorators with manual updates."""
    
    step_id = get_decorator_step()
    
    # Validation phase
    @lai.event(description="Validating input data")
    def validate(input_data: dict) -> bool:
        event_id = get_decorator_event()
        
        required_fields = ["name", "value", "type"]
        missing = [f for f in required_fields if f not in input_data]
        
        if missing:
            lai.update_event(
                event_id=event_id,
                description="Validation failed",
                result=f"Missing required fields: {missing}"
            )
            return False
        
        lai.update_event(
            event_id=event_id,
            description="Validation successful",
            result="All required fields present"
        )
        return True
    
    # Processing phase
    @lai.event(model="processor-v1")
    def process(input_data: dict) -> dict:
        event_id = get_decorator_event()
        
        # Update based on data type
        data_type = input_data.get("type", "unknown")
        lai.update_event(
            event_id=event_id,
            description=f"Processing {data_type} data",
            model=f"processor-{data_type}"
        )
        
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
        lai.update_step(
            step_id=step_id,
            state="Validation failed",
            eval_score=0.0,
            eval_description="Input data did not pass validation"
        )
        return {"status": "failed", "reason": "validation_error"}
    
    # Update step after successful validation
    lai.update_step(
        step_id=step_id,
        state="Validation passed, processing data",
        eval_score=0.5
    )
    
    # Process the data
    result = process(data)
    
    # Final step update
    lai.update_step(
        step_id=step_id,
        state="Processing complete",
        eval_score=1.0,
        eval_description="Data validated and processed successfully"
    )
    
    return {"status": "success", "data": result}


def main():
    """Run demonstrations of manual updates."""
    
    print("=== Manual Updates Demo ===\n")
    
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
    lai.end_session(
        is_successful=True,
        session_eval=0.95,
        session_eval_reason="Successfully demonstrated manual updates"
    )
    
    print("\n=== Demo Completed ===")
    print("\nKey takeaways:")
    print("- Use get_decorator_step() and get_decorator_event() to access IDs")
    print("- Manual updates override decorator parameters")
    print("- Updates can be conditional based on runtime logic")
    print("- Perfect for progress tracking and adaptive behavior")


if __name__ == "__main__":
    main()