# MCP Tool Usage Guide for Solo Dev Project

## Tool Hierarchy and Fallbacks

### Web Research
1. **WebSearch** (primary)
2. **mcp__searxng** (fallback when WebSearch fails)
3. **mcp__crawl4ai** (deep dive for specific pages)

### Documentation
1. **mcp__context7** (ALWAYS for official docs - FastAPI, Railway, Fly, Docker)
2. **WebSearch → SearxNG → Crawl4AI** (for tutorials/examples)

### Code Analysis
1. **mcp__serena__find_symbol** (better than Read for code)
2. **mcp__serena__search_for_pattern** (for finding code patterns)
3. **mcp__serena__initial_instructions** (MUST run at session start)

### Complex Tasks
1. **mcp__sequential-thinking** (for planning and analysis)
2. **Task tool** (for parallel agent work)

### Container Management (NEW)
1. **docker** MCP server (container operations)
   - `create-container` - Create and run containers
   - `deploy-compose` - Deploy compose stacks
   - `list-containers` - List all containers
   - `get-logs` - Get container logs

## Required Usage Patterns

### Every Session Start
```
1. mcp__serena__initial_instructions()
2. mcp__serena__list_memories()
3. mcp__sequential-thinking("plan session")
```

### Every Feature Implementation
```
1. Sequential thinking (plan)
2. Context7 (check docs)
3. Serena (understand code)
4. Task (parallel implementation)
5. Docker (test locally before deploy)
```

### Every Debug Session
```
1. Sequential thinking (analyze)
2. Serena search (find code)
3. Docker get-logs (container errors)
4. WebSearch → SearxNG (research error)
```

## Common Mistakes to Avoid
- ❌ Using Fetch without Crawl4AI fallback
- ❌ Skipping Context7 for library docs
- ❌ Not using Sequential thinking for complex tasks
- ❌ Working sequentially instead of using Task tool
- ❌ Not checking Serena memories

## Tool Performance Notes
- WebFetch often fails - always use Crawl4AI as backup
- Context7 is fastest for official docs
- Task tool can run 10 agents in parallel
- Serena memories persist across sessions
