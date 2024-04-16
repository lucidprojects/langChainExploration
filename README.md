# Project Title

Simple LangServe Test Project

## Description

Simple LangServe Test Project using Redis for temporay chat history.

## Getting Started

### Dependencies

- Run requrirements - (ideally in a venv)

```
pip install -r requirements.txt
```

### Installing

- Install Redis for chat history
  on mac os

```
brew install redis-stack
```

- Set API key environment variables

```
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="..."
```

### Executing program

Start redis server

```
redis-stack-server
```

\*note - confirm redis address matches whats in serveRedisHistory.py
line 67

```
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id=sessionId
)
```

- Run LangServe server

```
python serveRedisHistory.py
```

- Run the client

```
python clientRedis.py
```
