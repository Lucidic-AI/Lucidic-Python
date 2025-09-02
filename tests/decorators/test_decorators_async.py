"""Example of using Lucidic decorators with async functions."""
import asyncio
import os
from dotenv import load_dotenv
import lucidicai as lai
from openai import AsyncOpenAI
import aiohttp

# Load environment variables
load_dotenv()

# Example of async event decorator
@lai.event(
    description="Fetching data from multiple sources: Aggregate external API data (Compile comprehensive market analysis)"
)
async def fetch_market_data(symbols: list[str]) -> dict:
    """Fetch market data from multiple sources concurrently."""
    
    @lai.event(
        description="Fetch stock price",
        model="market-data-api",
        cost_added=0.0001
    )
    async def fetch_stock_price(symbol: str) -> dict:
        """Simulate fetching stock price from API."""
        # In real scenario, this would call an actual API
        await asyncio.sleep(0.5)  # Simulate API delay
        import random
        return {
            'symbol': symbol,
            'price': round(random.uniform(50, 500), 2),
            'change': round(random.uniform(-5, 5), 2)
        }
    
    @lai.event(
        description="Fetch company news",
        model="news-api"
    )
    async def fetch_company_news(symbol: str) -> list:
        """Simulate fetching latest news for a company."""
        await asyncio.sleep(0.3)  # Simulate API delay
        return [
            f"Breaking: {symbol} announces Q4 earnings",
            f"{symbol} expands into new markets",
            f"Analysts upgrade {symbol} rating"
        ][:2]  # Return top 2 news items
    
    # Fetch data concurrently for all symbols
    price_tasks = [fetch_stock_price(symbol) for symbol in symbols]
    news_tasks = [fetch_company_news(symbol) for symbol in symbols]
    
    prices = await asyncio.gather(*price_tasks)
    news = await asyncio.gather(*news_tasks)
    
    # Combine results
    market_data = {}
    for i, symbol in enumerate(symbols):
        market_data[symbol] = {
            'price_data': prices[i],
            'news': news[i]
        }
    
    return market_data


# Example of async event decorator with OpenAI
@lai.event(
    description="Generate market analysis using AI",
    model="gpt-4",
    cost_added=0.03
)
async def generate_market_analysis(market_data: dict) -> str:
    """Use OpenAI to generate a market analysis report."""
    
    # Create async OpenAI client
    client = AsyncOpenAI()
    
    # Prepare market summary
    summary_parts = []
    for symbol, data in market_data.items():
        price_info = data['price_data']
        summary_parts.append(
            f"{symbol}: ${price_info['price']} ({price_info['change']:+.2f}%)"
        )
    
    market_summary = "\n".join(summary_parts)
    
    # Generate analysis using OpenAI
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system", 
                "content": "You are a financial analyst providing market insights."
            },
            {
                "role": "user", 
                "content": f"Analyze this market data and provide a brief insight:\n\n{market_summary}"
            }
        ],
        max_tokens=200
    )
    
    return response.choices[0].message.content


# Main async workflow
@lai.event(
    description="Running complete market analysis workflow: Fetch data and generate insights (Provide comprehensive market report)"
)
async def complete_market_analysis(symbols: list[str]) -> dict:
    """Complete workflow for market analysis."""
    
    print(f"Starting analysis for: {', '.join(symbols)}")
    
    # Fetch market data (contains nested events)
    market_data = await fetch_market_data(symbols)
    
    # Generate AI analysis
    analysis = await generate_market_analysis(market_data)
    
    # Create final report
    @lai.event(
        description="Compile final report",
        result="Report compiled successfully"
    )
    async def compile_report(data: dict, analysis: str) -> dict:
        """Compile all data into final report."""
        await asyncio.sleep(0.1)  # Simulate processing
        
        return {
            'timestamp': asyncio.get_event_loop().time(),
            'symbols_analyzed': list(data.keys()),
            'market_data': data,
            'ai_analysis': analysis,
            'report_status': 'complete'
        }
    
    final_report = await compile_report(market_data, analysis)
    
    return final_report


async def main():
    """Run the async market analysis demo."""
    
    # Initialize Lucidic session
    lai.init(
        session_name="Async Market Analysis Demo",
        providers=["openai"],
        task="Demonstrate async decorator usage with market data"
    )
    
    # Symbols to analyze
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    
    print("=== Async Market Analysis Demo ===\n")
    
    try:
        # Run the complete analysis
        report = await complete_market_analysis(symbols)
        
        # Display results
        print("\nMarket Analysis Complete!")
        print(f"Symbols analyzed: {', '.join(report['symbols_analyzed'])}")
        print("\nMarket Data:")
        for symbol, data in report['market_data'].items():
            price_data = data['price_data']
            print(f"  {symbol}: ${price_data['price']} ({price_data['change']:+.2f}%)")
        
        print("\nAI Analysis:")
        print(report['ai_analysis'])
        
        # End session successfully
        lai.end_session(
            is_successful=True,
            session_eval=1.0,
            session_eval_reason="Successfully completed async market analysis"
        )
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        lai.end_session(
            is_successful=False,
            session_eval=0.0,
            session_eval_reason=f"Analysis failed: {str(e)}"
        )
    
    print("\nSession completed!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())