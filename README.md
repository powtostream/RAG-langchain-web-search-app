# RAG-langchain-web-search-app

This rag app is answering questions about recent news using Ollama LLM and DuckDuckGo web search.
Stages:
- Get user query
- Translate it to several queries suitable for web search
- Get html pages using ddgs for every query
- Parse, chunk, embed and put to vector storage
- Use llm query translation to rephrase the initial query
- Retrieve chunks based on vector similarity
- Use lln query expansion to generate key phrases
- Perform fused semantic + lexicographic search to rank retrieved chunks
- Put top n best chunks to prompt as a context and get Ollama llm response
