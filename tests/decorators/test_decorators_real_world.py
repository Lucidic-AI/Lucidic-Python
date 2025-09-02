"""Real-world example: AI-powered document processing pipeline with decorators."""
import os
from dotenv import load_dotenv
import lucidicai as lai
from openai import OpenAI
import json
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI()


@lai.event(
    description="Document processing pipeline started: Extract and analyze content (Generate structured insights)"
)
def process_document_batch(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Process a batch of documents through an AI pipeline."""
    
    processed_docs = []
    
    for doc in documents:
        # Process each document
        result = process_single_document(doc)
        processed_docs.append(result)
    
    # Generate batch summary
    summary = generate_batch_summary(processed_docs)
    
    return {
        'documents_processed': len(documents),
        'individual_results': processed_docs,
        'batch_summary': summary
    }


@lai.event(
    description="Processing individual document: Extract key information (Create structured summary)"
)
def process_single_document(document: Dict[str, str]) -> Dict[str, Any]:
    """Process a single document through multiple AI-powered steps."""
    
    doc_id = document.get('id', 'unknown')
    content = document.get('content', '')
    doc_type = document.get('type', 'general')
    
    # Step 1: Extract entities
    entities = extract_entities(content)
    
    # Step 2: Classify document
    classification = classify_document(content, doc_type)
    
    # Step 3: Extract key points
    key_points = extract_key_points(content, classification)
    
    # Step 4: Generate summary
    summary = generate_summary(content, key_points)
    
    return {
        'document_id': doc_id,
        'type': classification['category'],
        'entities': entities,
        'key_points': key_points,
        'summary': summary,
        'confidence': classification['confidence']
    }


@lai.event(
    description="Extract named entities from text",
    model="gpt-3.5-turbo",
    cost_added=0.002
)
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities using OpenAI."""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract named entities from the text. Return as JSON with categories: people, organizations, locations, dates."
            },
            {"role": "user", "content": text[:1000]}  # Limit text length
        ],
        response_format={"type": "json_object"},
        max_tokens=200
    )
    
    try:
        entities = json.loads(response.choices[0].message.content)
    except:
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
    
    return entities


@lai.event(
    description="Classify document type",
    model="gpt-3.5-turbo"
)
def classify_document(text: str, hint: str) -> Dict[str, Any]:
    """Classify the document into categories."""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"Classify this document. Hint: {hint}. Categories: legal, financial, technical, marketing, other."
            },
            {"role": "user", "content": text[:500]}
        ],
        max_tokens=50
    )
    
    content = response.choices[0].message.content.lower()
    
    # Parse classification
    categories = ['legal', 'financial', 'technical', 'marketing', 'other']
    category = 'other'
    confidence = 0.5
    
    for cat in categories:
        if cat in content:
            category = cat
            confidence = 0.9
            break
    
    return {'category': category, 'confidence': confidence}


@lai.event(
    description="Extract key points from document",
    model="gpt-4"
)
def extract_key_points(text: str, classification: Dict[str, Any]) -> List[str]:
    """Extract key points based on document type."""
    
    doc_type = classification['category']
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"Extract 3-5 key points from this {doc_type} document. Be concise."
            },
            {"role": "user", "content": text[:1500]}
        ],
        max_tokens=200
    )
    
    # Parse key points
    content = response.choices[0].message.content
    key_points = [point.strip() for point in content.split('\n') if point.strip()]
    
    return key_points[:5]  # Limit to 5 points


@lai.event(
    description="Generate document summary",
    result="Summary generated successfully"
)
def generate_summary(text: str, key_points: List[str]) -> str:
    """Generate a concise summary of the document."""
    
    key_points_text = "\n".join(f"- {point}" for point in key_points)
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Generate a 2-3 sentence summary based on the text and key points."
            },
            {
                "role": "user",
                "content": f"Text excerpt: {text[:500]}\n\nKey points:\n{key_points_text}"
            }
        ],
        max_tokens=100
    )
    
    return response.choices[0].message.content


@lai.event(
    description="Generate batch processing summary",
    model="gpt-4",
    cost_added=0.01
)
def generate_batch_summary(processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate insights from the batch of processed documents."""
    
    # Aggregate statistics
    doc_types = {}
    all_entities = {'people': set(), 'organizations': set(), 'locations': set()}
    
    for doc in processed_docs:
        # Count document types
        doc_type = doc['type']
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Collect unique entities
        for category, items in doc['entities'].items():
            if category in all_entities:
                all_entities[category].update(items)
    
    # Convert sets to lists for JSON serialization
    unique_entities = {k: list(v) for k, v in all_entities.items()}
    
    # Generate insights using AI
    stats_summary = f"""
    Documents processed: {len(processed_docs)}
    Document types: {json.dumps(doc_types)}
    Unique entities found: {sum(len(v) for v in unique_entities.values())}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Generate executive insights from this document batch analysis."
            },
            {"role": "user", "content": stats_summary}
        ],
        max_tokens=200
    )
    
    return {
        'total_documents': len(processed_docs),
        'document_types': doc_types,
        'unique_entities': unique_entities,
        'executive_insights': response.choices[0].message.content
    }


def main():
    """Run the document processing pipeline demo."""
    
    # Initialize Lucidic session
    lai.init(
        session_name="Document Processing Pipeline Demo",
        providers=["openai"],
        task="Process and analyze multiple documents using AI",
        tags=["document-processing", "nlp", "production-example"]
    )
    
    # Sample documents to process
    sample_documents = [
        {
            'id': 'doc001',
            'type': 'legal',
            'content': """
            CONFIDENTIALITY AGREEMENT
            
            This Agreement is entered into as of January 15, 2024, between TechCorp Inc., 
            a Delaware corporation ("Company"), and Jane Smith, an individual ("Recipient").
            
            The Company agrees to disclose certain confidential information to the Recipient 
            for the purpose of evaluating a potential business relationship. The Recipient 
            agrees to maintain the confidentiality of such information and not to disclose 
            it to any third parties without the prior written consent of the Company.
            
            This Agreement shall be governed by the laws of the State of California.
            """
        },
        {
            'id': 'doc002',
            'type': 'financial',
            'content': """
            Q4 2023 Financial Report - TechCorp Inc.
            
            Revenue: $45.2M (up 23% YoY)
            Operating Income: $12.1M (up 31% YoY)
            Net Income: $9.8M (up 28% YoY)
            
            Key Highlights:
            - Strong growth in cloud services division
            - Successful launch of new AI product line
            - Expanded operations to European markets
            - Increased R&D investment by 40%
            
            CEO John Doe commented: "We're pleased with our strong finish to 2023 
            and remain optimistic about our growth trajectory in 2024."
            """
        },
        {
            'id': 'doc003',
            'type': 'technical',
            'content': """
            Technical Specification: AI Model Architecture v2.0
            
            Model: Transformer-based architecture with 175B parameters
            Training Data: 1TB of curated text from multiple domains
            Hardware: Trained on 1000 NVIDIA A100 GPUs
            
            Key Features:
            - Multi-modal input support (text, image, audio)
            - Context window: 32K tokens
            - Inference speed: 50ms average latency
            - Supports 95 languages
            
            Performance Metrics:
            - Accuracy: 94.3% on benchmark tasks
            - F1 Score: 0.92
            - Model size: 350GB (quantized: 87GB)
            """
        }
    ]
    
    print("=== Document Processing Pipeline Demo ===\n")
    print(f"Processing {len(sample_documents)} documents...\n")
    
    try:
        # Process the batch of documents
        results = process_document_batch(sample_documents)
        
        # Display results
        print("\n=== Processing Complete ===\n")
        print(f"Documents processed: {results['documents_processed']}")
        
        print("\n--- Document Summaries ---")
        for doc_result in results['individual_results']:
            print(f"\nDocument ID: {doc_result['document_id']}")
            print(f"Type: {doc_result['type']} (confidence: {doc_result['confidence']:.2f})")
            print(f"Summary: {doc_result['summary']}")
            print(f"Entities found: {sum(len(v) for v in doc_result['entities'].values())}")
        
        print("\n--- Batch Insights ---")
        batch_summary = results['batch_summary']
        print(f"Document type distribution: {batch_summary['document_types']}")
        print(f"Total unique entities: {sum(len(v) for v in batch_summary['unique_entities'].values())}")
        print(f"\nExecutive Insights:\n{batch_summary['executive_insights']}")
        
        # End session successfully
        lai.end_session(
            is_successful=True,
            session_eval=0.98,
            session_eval_reason="Successfully processed all documents with high-quality insights"
        )
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        lai.end_session(
            is_successful=False,
            session_eval=0.0,
            session_eval_reason=f"Pipeline failed: {str(e)}"
        )
    
    print("\n=== Session Completed ===")


if __name__ == "__main__":
    main()