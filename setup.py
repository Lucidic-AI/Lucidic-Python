from setuptools import setup, find_packages

setup(
    name="lucidicai",
    version="3.7.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.1",
        "urllib3",
        "httpx>=0.27.0",
        "boto3",
        "python-dotenv",
        "langchain",
        "langchain-community",
        "langchain-core",
        "openai>=1.3.0",
        "pillow",
        "anthropic",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-instrumentation",
        # LUC-667: floor pins on the openllmetry instrumentation packages. these control the
        # span shape Lucidic consumes; <0.53.4 predates fixes we rely on. the extractor reads
        # both the legacy flat and the new gen_ai.input/output.messages shapes, so newer
        # versions are supported without a ceiling.
        "opentelemetry-instrumentation-openai>=0.53.4",
        "opentelemetry-instrumentation-anthropic>=0.53.4",
        "opentelemetry-instrumentation-langchain>=0.53.4",
        "opentelemetry-semantic-conventions-ai",
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
