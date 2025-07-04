from setuptools import setup, find_packages

setup(
    name="lucidicai",
    version="1.2.16",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.1",
        "urllib3",
        "boto3",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "openai>=1.3.0",
        "pillow",
        "anthropic",
        # "pydantic_ai",
    ],
    author="Andy Liang",
    author_email="andy@lucidic.ai",
    description="Lucidic AI Python SDK",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
