"""Run the agent from SearchEngine/app.py over the dataset and score tool selection.
Usage: GROQ_API_KEY=... python evaluate.py
"""
import json, os, time
from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.callbacks import BaseCallbackHandler

class ToolLogger(BaseCallbackHandler):
    def __init__(self): self.calls = []
    def on_tool_start(self, serialized, input_str, **kw):
        self.calls.append(serialized.get("name"))

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250))
search = DuckDuckGoSearchRun(name="Search")
llm = ChatGroq(groq_api_key=os.environ["GROQ_API_KEY"], model_name="llama-3.3-70b-versatile")
agent = initialize_agent([search, arxiv, wiki], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         handle_parsing_errors=True, max_iterations=6)

data = json.load(open("tool_selection_dataset.json"))
results, by_cat = [], defaultdict(list)
for d in data:
    cb = ToolLogger()
    try: agent.run(d["query"], callbacks=[cb])
    except Exception as e: print("error", d["id"], e)
    called, exp = list(dict.fromkeys(cb.calls)), d["expected_tools"]  # de-dup, keep order
    exact = set(called) == set(exp)                       # right set of tools, nothing extra
    recall = (len(set(called) & set(exp)) / len(exp)) if exp else float(not called)
    precision = (len(set(called) & set(exp)) / len(called)) if called else float(not exp)
    results.append({**d, "called": called, "exact_match": exact, "recall": recall, "precision": precision})
    by_cat[d["category"]].append(exact)
    print(d["id"], "OK " if exact else "XX ", exp, "->", called)
    time.sleep(2)  # avoid Groq rate limits

json.dump(results, open("results.json", "w"), indent=2)
print("\nExact-match accuracy overall: %.1f%%" % (100 * sum(r["exact_match"] for r in results) / len(results)))
for c, v in by_cat.items(): print(f"  {c:16s} {100*sum(v)/len(v):5.1f}%  (n={len(v)})")
print("Mean recall %.2f | mean precision %.2f" % (
    sum(r["recall"] for r in results)/len(results), sum(r["precision"] for r in results)/len(results)))
