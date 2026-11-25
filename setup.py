from pathlib import Path

from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="lucidicai",
    version="4.1.0",
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
        # Floor pins on the openllmetry instrumentation packages. these control the
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
