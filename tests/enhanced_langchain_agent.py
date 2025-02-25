from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI  # Updated import
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import os

class LangchainAgent:
    """Agent to test various Langchain capabilities with Lucidic tracking"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Set OpenAI API key explicitly
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Initialize different models for tracking different event types
        self.chat_model = ChatOpenAI(
            model="gpt-4", 
            temperature=0,
            openai_api_key=self.api_key
        )
        
        self.analysis_model = ChatOpenAI(
            model="gpt-4", 
            temperature=0,
            openai_api_key=self.api_key
        )
        
        self.summary_model = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0.2,
            openai_api_key=self.api_key
        )
        
        # Initialize memory for conversation tracking
        self.message_history = ChatMessageHistory()
        
        # Initialize chains
        self.setup_chains()
        
    def setup_chains(self):
        """Set up various chain types for testing using modern Langchain pattern"""
        # Analysis chain as a runnable sequence
        analysis_prompt = PromptTemplate.from_template(
            "Provide a detailed analysis of {topic}, including key points and implications."
        )
        self.analysis_chain = analysis_prompt | self.analysis_model | StrOutputParser()
        
        # Summary chain as a runnable sequence
        summary_prompt = PromptTemplate.from_template(
            "Summarize the main points of this text:\n\n{text}"
        )
        self.summary_chain = summary_prompt | self.summary_model | StrOutputParser()
    
    def test_chat(self, message: str) -> Dict:
        """Test basic chat functionality"""
        try:
            messages = [
                SystemMessage(content="You are a helpful AI assistant."),
                HumanMessage(content=message)
            ]
            
            response = self.chat_model.invoke(messages)
            return {
                "success": True,
                "response": response.content,
                "type": "chat"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "type": "chat"
            }
    
    def test_analysis(self, topic: str) -> Dict:
        """Test chain-based analysis"""
        try:
            # Use modern chain pattern (Runnable)
            result = self.analysis_chain.invoke({"topic": topic})
            return {
                "success": True,
                "response": result,
                "type": "analysis"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "type": "analysis"
            }
    
    def test_memory(self, messages: List[str]) -> List[Dict]:
        """Test conversation memory"""
        results = []
        for message in messages:
            try:
                # Add the message to memory
                self.message_history.add_user_message(message)
                
                # Create message list with history
                all_messages = [
                    SystemMessage(content="You are a helpful AI assistant.")
                ] + self.message_history.messages
                
                # Get response
                response = self.chat_model.invoke(all_messages)
                
                # Add response to memory
                self.message_history.add_ai_message(response.content)
                
                results.append({
                    "success": True,
                    "message": message,
                    "response": response.content,
                    "type": "memory"
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "message": message,
                    "error": str(e),
                    "type": "memory"
                })
        
        return results
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.message_history.clear()