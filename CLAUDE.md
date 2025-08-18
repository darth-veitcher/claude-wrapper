# CLAUDE.md - Project Context for Claude Code

## 🚨 SESSION START CHECKLIST (DO THIS FIRST!)

When starting any new session on this project:

1. **Initialize Serena Context**

   ```
   mcp__serena__initial_instructions()
   mcp__serena__list_memories()
   # Read any relevant memories
   ```

2. **Check Project Status**

   - Review git status and recent commits
   - Check for any failing tests or CI/CD issues
   - Identify current development priorities

3. **Plan Session with Sequential Thinking**

   ```
   mcp__sequential-thinking("What should I work on this session based on the project state?")
   ```

4. **Set Up Tool Fallbacks**
   - WebSearch → mcp**searxng → mcp**crawl4ai
   - Fetch → mcp\_\_crawl4ai
   - Read → mcp**serena**find_symbol

## 📝 MEMORY CREATION TRIGGERS

**ALWAYS create a Serena memory when:**

1. **Starting a new project or major feature** - Document goals, architecture, and approach
2. **Making architectural decisions** - Record the decision, rationale, and alternatives considered
3. **Completing major milestones** - Capture what was accomplished and lessons learned
4. **Discovering important patterns or solutions** - Save for future reference
5. **Establishing project conventions** - Document agreed-upon standards and practices

**Memory Naming Convention**: Use descriptive names like `project_name_topic` (e.g., `claude_memory_project_overview`, `auth_system_architecture`)

**What to Include**: Summary, key decisions, technical details, file locations, current status, next steps

## 🏁 SESSION END CHECKLIST

Before ending a session, consider:

1. **Document significant work** - Did you complete something memory-worthy? (See triggers above)
2. **Update project status** - Does the "Current Phase" in CLAUDE.md need updating?
3. **Commit changes** - Are there uncommitted changes that should be saved?
4. **Note blockers** - Document any blockers or issues for the next session

## Documentation Structure

### Project Management (`/project-management/`)

Check this directory for:

- Project requirements and specifications
- Implementation plans and roadmaps
- Technical architecture decisions
- Feature specifications

Use these documents as canonical sources of intent and background context for all development decisions.

## MCP Tool Usage Guidelines

### Tool Selection and Fallback Strategies

#### Web Research Hierarchy

1. **Primary**: `WebSearch` for general queries
2. **Fallback**: `mcp__searxng` when WebSearch fails or returns limited results
3. **Deep Dive**: `mcp__crawl4ai` for specific page content when Fetch fails

#### Documentation and Learning

1. **Always Start With**: `mcp__context7` for official library documentation
   - Use for: FastAPI, Docker, Railway, Fly.io, Stripe, etc.
   - Command: First `resolve-library-id` then `get-library-docs`
2. **Fallback**: WebSearch → SearxNG → Crawl4AI

#### Complex Problem Solving

1. **Use Sequential Thinking** for:
   - Architecture decisions
   - Multi-step implementations
   - Debugging complex issues
   - Planning agent interactions
2. **Minimum Usage**: At least once per complex task

#### Code Analysis (Serena)

1. **Session Start**: ALWAYS run `mcp__serena__initial_instructions`
2. **Memory Usage**:
   - Check memories with `mcp__serena__list_memories`
   - Read relevant memories for context
   - Write new memories for important decisions
3. **Code Navigation**:
   - Use `mcp__serena__find_symbol` over basic Read
   - Use `mcp__serena__search_for_pattern` for code searches

### MCP Tool Best Practices

#### Research Flow

```
1. mcp__serena__initial_instructions (get project context)
2. mcp__context7 (check official docs)
3. WebSearch (general research)
4. mcp__searxng (if WebSearch fails)
5. mcp__crawl4ai (for specific pages)
6. mcp__sequential-thinking (synthesize findings)
```

#### Implementation Flow

```
1. mcp__sequential-thinking (plan approach)
2. mcp__serena__find_symbol (understand existing code)
3. Task tool (spawn parallel agents if needed)
4. Write/Edit (implement changes)
5. mcp__serena__write_memory (document decisions)
```

#### Debugging Flow

```
1. mcp__sequential-thinking (analyze problem)
2. mcp__serena__search_for_pattern (find related code)
3. mcp__playwright (if UI testing needed)
4. Fix issue
5. mcp__serena__write_memory (document solution)
```

### Required Tool Usage Per Task Type

#### Feature Implementation

- ✅ Sequential thinking for planning
- ✅ Context7 for framework patterns
- ✅ Serena for code structure understanding
- ✅ Task tool for parallel work

#### Debugging

- ✅ Sequential thinking for root cause analysis
- ✅ Serena pattern search
- ✅ WebSearch → SearxNG for error research

#### Documentation

- ✅ Context7 for best practices
- ✅ Serena memories for project decisions
- ✅ Sequential thinking for structure

### Session Initialization Checklist

Every new session MUST:

1. Run `mcp__serena__initial_instructions`
2. Check `mcp__serena__list_memories`
3. Read CLAUDE.md (this file)
4. Review current project state
5. Use `mcp__sequential-thinking` to plan session

## MCP Tool Usage Patterns

### Research Pattern (Use for all new features)

```python
# 1. Plan with sequential thinking
mcp__sequential-thinking("How should I implement [feature]?")

# 2. Check official docs
mcp__context7__resolve-library-id("library-name")
mcp__context7__get-library-docs(library_id, topic="relevant-topic")

# 3. Search for examples
WebSearch("library feature example 2025")
# If limited results:
mcp__searxng("library tutorial implementation")

# 4. Deep dive specific pages
mcp__crawl4ai("https://docs.example.com/feature")
```

### Code Analysis Pattern (Use before any edits)

```python
# 1. Get Serena context
mcp__serena__initial_instructions()

# 2. Find relevant code
mcp__serena__find_symbol("ClassName")
mcp__serena__search_for_pattern("pattern")

# 3. Understand relationships
mcp__serena__find_referencing_symbols("function_name")
```

### Parallel Implementation Pattern

```python
# Use Task tool for parallel work
Task("Component A", "Implement feature A")
Task("Component B", "Implement feature B")
Task("Tests", "Create test suite")
```

### Common Pitfalls to Avoid

#### ❌ Bad Pattern: Direct implementation without research

```python
# DON'T DO THIS
Write("config.py", "some config I think might work")
```

#### ✅ Good Pattern: Research → Plan → Implement

```python
# DO THIS
mcp__sequential-thinking("Plan configuration approach")
mcp__context7("framework configuration docs")
mcp__serena__search_for_pattern("existing config")
Write("config.py", "validated config from docs")
```

#### ❌ Bad Pattern: Using Fetch without fallback

```python
# DON'T DO THIS
Fetch("https://some-site.com") # Often fails
```

#### ✅ Good Pattern: Fetch → Crawl4AI fallback

```python
# DO THIS
try:
    WebFetch("https://some-site.com")
except:
    mcp__crawl4ai("https://some-site.com")
```

#### ❌ Bad Pattern: Sequential work that could be parallel

```python
# DON'T DO THIS
implement_feature_a()
implement_feature_b()
implement_feature_c()
```

#### ✅ Good Pattern: Parallel agents for independent tasks

```python
# DO THIS
Task("Feature A", "Implement feature A")
Task("Feature B", "Implement feature B")
Task("Feature C", "Implement feature C")
```

## Agent Development Guidelines

### Creating New Agents

When creating AI agents for this project:

1. Check `.claude/agents/` for existing patterns
2. Use YAML configuration for agent definitions
3. Include clear triggers and prompts
4. Specify required tools explicitly
5. Document agent interactions and dependencies

### Parallel Execution

- Use Task tool to spawn sub-agents
- Maximum 10 agents running simultaneously
- Each agent has independent context window
- Coordinate through shared file system

## Testing Approach

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **E2E Tests**: Test complete user workflows
4. **Performance Tests**: Validate speed requirements

## Important Context

### Development Philosophy

- **Simplicity First**: If it requires extensive documentation, it's too complex
- **Speed Matters**: Optimize for rapid iteration and deployment
- **Automation**: Prefer automated solutions over manual processes
- **User Experience**: Development should be enjoyable and productive

### Code Quality Standards

- Follow existing patterns and conventions
- Write tests for new functionality
- Document complex logic and decisions
- Use type hints where applicable
- Keep functions focused and small

## Your Role

You are an AI assistant helping to develop and maintain this project. Focus on:

1. **Code Quality** - Write clean, maintainable code
2. **Best Practices** - Follow framework and language conventions
3. **Performance** - Consider efficiency in all implementations
4. **User Experience** - Make the developer experience delightful

When in doubt, use `mcp__sequential-thinking` to reason through decisions and document important choices with `mcp__serena__write_memory`.

---

_Remember: Always start with the session checklist and use the appropriate MCP tools for each task._
