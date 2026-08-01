"""
evals/ — standalone cost-vs-quality harness for routing_priority tuning.

Not part of the router library: it imports router/ as a consumer (RoutingEngine,
RoutingRequest, provider_caller) exactly like any other caller would, and
touches no core router code. See EVALS_ROUTING_PRIORITY.md for usage.

This is a different, narrower harness than router/evals/ (which grades
GSM8K/MMLU/HumanEval/MT-Bench across "strategies" like flux/premium/cheapest).
This one asks a specific question: for a fixed, hand-built set of agent-style
steps (plan / tool_select / tool_result_summarize / final_answer), how much
quality does routing_priority="cascade" or "cost_optimized" actually give up
relative to "quality_max"?
"""
