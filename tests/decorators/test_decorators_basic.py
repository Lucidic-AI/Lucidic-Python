"""Basic example of using Lucidic decorators for tracking AI workflows."""
import os
from dotenv import load_dotenv
import lucidicai as lai
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Lucidic SDK
lai.init(
    session_name="Decorator Demo - Customer Support Bot",
    providers=["openai"],
    task="Demonstrate decorator usage in a customer support scenario"
)

# Example 1: Using step decorator for high-level workflow tracking
@lai.step(
    state="Received customer inquiry",
    action="Analyze and categorize issue",
    goal="Identify the type of support needed"
)
def categorize_customer_issue(customer_message: str) -> dict:
    """Categorize the customer's issue to route to appropriate handler."""
    
    # In a real scenario, this might use NLP or keyword matching
    issue_keywords = {
        'billing': ['bill', 'payment', 'charge', 'invoice', 'refund'],
        'technical': ['error', 'bug', 'crash', 'not working', 'broken'],
        'account': ['password', 'login', 'account', 'access', 'locked']
    }
    
    message_lower = customer_message.lower()
    
    for category, keywords in issue_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return {
                'category': category,
                'confidence': 0.85,
                'original_message': customer_message
            }
    
    return {
        'category': 'general',
        'confidence': 0.5,
        'original_message': customer_message
    }


# Example 2: Using event decorator for specific operations
@lai.event(
    description="Generate automated response",
    model="response-generator",
    cost_added=0.001
)
def generate_initial_response(issue_data: dict) -> str:
    """Generate an initial response based on issue category."""
    
    responses = {
        'billing': "I understand you're having a billing issue. Let me help you with that.",
        'technical': "I see you're experiencing technical difficulties. I'll help troubleshoot.",
        'account': "I understand you need help with your account. Let me assist you.",
        'general': "Thank you for contacting support. How can I help you today?"
    }
    
    return responses.get(issue_data['category'], responses['general'])


# Example 3: Nested decorators - step with events inside
@lai.step(
    state="Processing support ticket",
    action="Generate complete support response",
    goal="Resolve customer issue"
)
def handle_support_ticket(customer_message: str) -> dict:
    """Complete workflow for handling a support ticket."""
    
    # This event will be tracked within the step
    @lai.event(description="Validate customer message")
    def validate_message(message: str) -> bool:
        """Check if message is valid and not spam."""
        return len(message) > 5 and len(message) < 1000
    
    # Validate the message
    is_valid = validate_message(customer_message)
    if not is_valid:
        return {'status': 'rejected', 'reason': 'Invalid message format'}
    
    # Categorize the issue (uses step decorator)
    issue_data = categorize_customer_issue(customer_message)
    
    # Generate response (uses event decorator)
    initial_response = generate_initial_response(issue_data)
    
    # Create OpenAI client for detailed response
    client = OpenAI()
    
    # This LLM call will automatically create its own event
    detailed_response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"You are a helpful customer support agent handling {issue_data['category']} issues."},
            {"role": "user", "content": customer_message}
        ],
        max_tokens=150
    )
    
    return {
        'status': 'resolved',
        'category': issue_data['category'],
        'initial_response': initial_response,
        'detailed_response': detailed_response.choices[0].message.content,
        'confidence': issue_data['confidence']
    }


# Example 4: Event decorator with custom description and result
@lai.event(
    description="Log support interaction to database",
    result="Successfully logged to database"
)
def log_interaction(ticket_data: dict) -> bool:
    """Log the support interaction for analytics."""
    # In a real app, this would save to a database
    print(f"Logging interaction: {ticket_data['category']} - {ticket_data['status']}")
    return True


def main():
    """Run the customer support bot demo."""
    
    # Test customer messages
    test_messages = [
        "I was charged twice for my subscription last month!",
        "The app keeps crashing when I try to upload files",
        "I forgot my password and can't log in",
        "How do I upgrade my plan?"
    ]
    
    print("=== Customer Support Bot Demo ===\n")
    
    for message in test_messages:
        print(f"Customer: {message}")
        
        # Handle the support ticket
        result = handle_support_ticket(message)
        
        # Log the interaction
        log_interaction(result)
        
        print(f"Bot: {result['initial_response']}")
        print(f"Category: {result['category']} (confidence: {result['confidence']})")
        print(f"Status: {result['status']}")
        print("-" * 50 + "\n")
    
    # End the session
    lai.end_session(
        is_successful=True,
        session_eval=0.95,
        session_eval_reason="Successfully handled all test support tickets"
    )
    
    print("Session completed successfully!")


if __name__ == "__main__":
    main()