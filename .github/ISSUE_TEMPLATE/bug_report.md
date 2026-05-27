---
name: Bug report
about: Something in Flux behaves incorrectly
title: "[BUG] "
labels: bug
---

**What happened**
A clear description of the bug.

**Reproduction**
The smallest snippet that triggers it:

```python
# make_flux(...) / RoutingRequest(...) call, prompt, routing_priority, etc.
```

**Expected vs. actual**
What you expected to happen, and what happened instead. Include the routing
decision or error message. For routing questions, run with `verbose=True` and
paste the explanation.

**Environment**
- Flux version:
- Python version (3.10+):
- OS:

**Logs**
Relevant structlog output. Do NOT paste API keys.
