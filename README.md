# Structured Chat Agent with Tool Integration

## Project Overview
This project implements a conversational AI agent using the LangChain framework, designed to answer user queries by leveraging three tools: Wikipedia, Arxiv, and a custom LangSmith retriever. The agent uses the `llama2` model from Ollama, configured to output structured JSON responses, and is built with `create_structured_chat_agent` to process queries in a structured format. The agent fetches factual information, academic paper summaries, and LangSmith documentation to provide accurate responses to queries like "What is OpenAI."

The agent follows a structured workflow:
- Receives a user query.
- Outputs a JSON blob with an `action` (tool name or "Final Answer") and `action_input` (tool input or final response).
- Maintains conversation history via `agent_scratchpad` to track tool calls and observations.
- Iterates up to 5 times to gather information before delivering a final answer.

The project is implemented in a Jupyter Notebook (`agents.ipynb`) and successfully handles queries like "What is OpenAI" but faced challenges with "Tell me about India" due to parsing errors.

## Setup and Installation

### Prerequisites
- **Python**: 3.12.0
- **Ollama**: Install and run the `llama2` model:
  ```bash
  ollama pull llama2
  ollama serve
  ```
- **Dependencies**:
  ```bash
  pip install langchain langchain-community langchain-ollama faiss-cpu wikipedia arxiv
  ```

### Dependencies
- `langchain`: Core framework for building LLM-powered applications.
- `langchain-community`: Provides Wikipedia and Arxiv tools.
- `langchain-ollama`: Integration for Ollama LLMs.
- `faiss-cpu`: Vector store for LangSmith document retrieval.
- `wikipedia`: Wikipedia API wrapper.
- `arxiv`: Arxiv API wrapper.

### Environment Notes
- Set the `USER_AGENT` environment variable to avoid Wikipedia API warnings:
  ```bash
  export USER_AGENT="YourAppName (your.email@example.com)"
  ```
- Use LangChain version 0.3.x or later to avoid compatibility issues:
  ```bash
  pip install --upgrade langchain langchain-community langchain-ollama
  ```
- Note: The `OllamaEmbeddings` class is deprecated in LangChain 0.3.1 and should be imported from `langchain_ollama` in future updates:
  ```python
  from langchain_ollama import OllamaEmbeddings
  ```

## Project Structure
The project is implemented in `agents.ipynb` with the following components:

1. **Tool Definitions**:
   - **Wikipedia Tool**: Fetches concise summaries (max 200 characters) from Wikipedia for general knowledge queries.
     ```python
     from langchain_community.tools import WikipediaQueryRun
     from langchain_community.utilities import WikipediaAPIWrapper
     wiki_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
     wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api, name="wikipedia", description="Search Wikipedia for factual information about general topics, such as countries, historical events, or public figures.")
     ```
   - **Arxiv Tool**: Retrieves academic paper summaries (max 200 characters) from Arxiv for research-related queries.
     ```python
     from langchain_community.utilities import ArxivAPIWrapper
     from langchain_community.tools import ArxivQueryRun
     arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
     arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api, name="arxiv", description="Search Arxiv for academic papers and research-related information.")
     ```
   - **LangSmith Retrieval Tool**: Searches LangSmith documentation using a FAISS vector store with `llama2` embeddings.
     ```python
     from langchain_community.document_loaders import WebBaseLoader
     from langchain_community.vectorstores import FAISS
     from langchain_community.embeddings import OllamaEmbeddings
     from langchain_text_splitters import RecursiveCharacterTextSplitter
     from langchain.tools.retriever import create_retriever_tool
     loader = WebBaseLoader(web_path="https://docs.smith.langchain.com/")
     docs = loader.load()
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
     documents = text_splitter.split_documents(docs)
     db = FAISS.from_documents(documents, OllamaEmbeddings(model="llama2"))
     retriever = db.as_retriever()
     retrieval_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith. For any questions about LangSmith, you must use this tool")
     ```
   - Combined tools:
     ```python
     tools = [wiki_tool, arxiv_tool, retrieval_tool]
     ```

2. **LLM Setup**:
   - Uses `llama2` with `format="json"` to ensure structured JSON output:
     ```python
     from langchain_ollama import ChatOllama
     llm = ChatOllama(model="llama2", temperature=0, format="json")
     ```

3. **Prompt**:
   - Pulled from LangChain Hub (`hwchase17/structured-chat-agent`):
     ```python
     from langchain import hub
     prompt = hub.pull("hwchase17/structured-chat-agent")
     ```
   - The prompt instructs the LLM to:
     - Respond with a single JSON blob containing `action` (tool name or "Final Answer") and `action_input`.
     - Use a structure: `Question`, `Thought`, `Action`, `Observation`, repeating until `Final Answer`.
     - Include `MessagesPlaceholder("agent_scratchpad")` to track tool calls and observations.

4. **Agent and Executor**:
   - Creates a structured chat agent:
     ```python
     from langchain.agents import create_structured_chat_agent
     agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
     ```
   - Uses `AgentExecutor` to manage tool calls and iterations:
     ```python
     from langchain.agents import AgentExecutor
     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)
     ```

5. **Execution**:
   - Processes queries by invoking the executor:
     ```python
     response = agent_executor.invoke({"input": "What is OpenAI"})
     print(response)
     ```
   - Output for "What is OpenAI":
     ```
     > Entering new AgentExecutor chain...
     > Action: {"action": "wikipedia", "action_input": {"query": "OpenAI"}}
     > Observation: Page: OpenAI Summary: OpenAI, Inc. is an American artificial intelligence...
     > Action: {"action": "arxiv", "action_input": {"query": "OpenAI"}}
     > Observation: Published: 2016-06-05 Title: OpenAI Gym Authors: Greg Brockman...
     > Action: {"action": "langsmith_search", "action_input": {"query": "OpenAI"}}
     > Observation: LangSmith is a platform for building production-grade LLM applications...
     > Action: {"action": "Final Answer", "action_input": "OpenAI is an American artificial intelligence..."}
     > Finished chain.
     {'input': 'What is OpenAI', 
      'output': 'OpenAI is an American artificial intelligence organization headquartered in San Francisco, California. It aims to develop "safe and beneficial" artificial general intelligence.'}
     ```

## Challenges Faced
The project faced significant challenges when processing queries like "Tell me about India":

1. **ValueError: variable agent_scratchpad should be a list of base messages, got of type <class 'str'>**:
   - **Cause**: The LLM (`llama2` or `llama3`) output a JSON string (e.g., `'{"action": "wikipedia", "action_input": "India"}'`) instead of a parsed JSON object, which was passed to `agent_scratchpad`, expecting a list of `BaseMessage` objects.
   - **Attempts**:
     - Switched to `llama3` for better JSON handling, but the issue persisted.
     - Added `format="json"` to `ChatOllama`, expecting a parsed JSON object, but the LLM still output a string.
     - Tried a custom parser to convert the JSON string to `AgentAction` or `AgentFinish`, but it was ignored by `create_react_agent`.
     - Used `create_structured_chat_agent` instead of `create_react_agent`, which resolved the issue for some queries (e.g., "What is OpenAI").
   - **Resolution**: The final code used `create_structured_chat_agent` with `llama2` and `format="json"`, which worked for "What is OpenAI" but not consistently for "Tell me about India".

2. **KeyError: 'intermediate_steps'**:
   - **Cause**: A custom `debug_scratchpad` function didn’t include `intermediate_steps` in the input dictionary, causing a `KeyError`.
   - **Resolution**: Initialized `intermediate_steps` as an empty list, but ultimately removed the debug function.

3. **RecursionError: maximum recursion depth exceeded**:
   - **Cause**: The `debug_scratchpad` function recursively called `agent_executor._call`, creating an infinite loop.
   - **Resolution**: Replaced with a non-recursive `debug_invoke` wrapper.

## Lessons Learned
1. **LLM Output Handling**:
   - `llama2` and `llama3` may output JSON strings instead of parsed objects, even with `format="json"`. This requires careful handling to align with LangChain’s expectations.
   - `create_structured_chat_agent` is more reliable than `create_react_agent` for structured JSON output.

2. **Agent Configuration**:
   - `MessagesPlaceholder("agent_scratchpad")` is critical for maintaining conversation history, enabling the agent to track tool calls and observations.
   - `max_iterations=5` allows the agent to query multiple tools before finalizing a response, improving answer quality.

3. **Debugging**:
   - Using `verbose=True` and a debug wrapper (`debug_invoke`) helped identify issues like stringified JSON outputs and empty `intermediate_steps`.
   - Testing LLM output directly (e.g., `llm.invoke(...)`) was key to diagnosing parsing issues.

4. **Tool Integration**:
   - Tools like `WikipediaQueryRun` and `ArxivQueryRun` are easy to integrate, but custom retrievers (e.g., `langsmith_search`) require proper vector store setup.
   - Limiting output length (`doc_content_chars_max=200`) ensures concise responses.

## Project Importance
This project is significant for several reasons:
- **Versatile Query Handling**: Integrates multiple tools (Wikipedia, Arxiv, LangSmith) to answer diverse queries, from general knowledge to academic research and technical documentation.
- **Structured AI Workflow**: Uses a structured JSON-based workflow to ensure reliable tool selection and response generation, suitable for production-grade applications.
- **Learning Experience**: Overcoming errors like `ValueError`, `KeyError`, and `RecursionError` provided deep insights into LangChain’s agent framework, LLM output parsing, and debugging complex AI systems.
- **Open-Source Accessibility**: Leverages open-source tools (LangChain, Ollama, FAISS) and free APIs (Wikipedia, Arxiv), making it accessible for developers to build upon.
- **Scalability**: The modular tool setup allows easy addition of new tools, enhancing the agent’s capabilities for various domains.

## Running the Project
1. Clone the repository or copy `agents.ipynb`.
2. Install dependencies (see above).
3. Ensure Ollama is running with `llama2`.
4. Run the notebook cells sequentially.
5. Test queries:
   ```python
   response = agent_executor.invoke({"input": "What is OpenAI"})
   print(response)
   ```

## Example Query
```python
response = agent_executor.invoke({"input": "What is OpenAI"})
```
**Output**:
```
{'input': 'What is OpenAI', 
 'output': 'OpenAI is an American artificial intelligence organization headquartered in San Francisco, California. It aims to develop "safe and beneficial" artificial general intelligence.'}
```

## Future Improvements
- **Resolve India Query Issue**: Investigate why "Tell me about India" fails, possibly due to query complexity or tool output parsing. Test with `llama3` or a custom parser.
- **Add More Tools**: Integrate web search or additional APIs for broader query coverage.
- **Optimize Prompt**: Refine the prompt to improve tool selection logic or reduce iterations.
- **Update Embeddings**: Use `langchain_ollama.OllamaEmbeddings` to address deprecation warnings.

## Troubleshooting
If errors occur (e.g., for "Tell me about India"):
1. **Check LLM Output**:
   ```python
   print(llm.invoke("Return a JSON object: {'action': 'Final Answer', 'action_input': 'Test'}"))
   ```
   Ensure it’s a parsed JSON object, not a string.
2. **Verify LangChain Version**:
   ```bash
   pip show langchain langchain-community langchain-ollama
   ```
   Upgrade to 0.3.x:
   ```bash
   pip install --upgrade langchain langchain-community langchain-ollama
   ```
3. **Test Tools**:
   ```python
   print(wiki_tool.run("India"))
   print([t.name for t in tools])
   ```

## Conclusion
This project successfully builds a structured chat agent with LangChain, integrating Wikipedia, Arxiv, and LangSmith tools to answer queries like "What is OpenAI." Despite challenges with parsing errors for queries like "Tell me about India," the use of `create_structured_chat_agent` and `llama2` with `format="json"` delivers reliable performance for most queries. The journey highlighted the importance of structured output, robust error handling, and debugging in building AI agents, making this a valuable foundation for future enhancements.