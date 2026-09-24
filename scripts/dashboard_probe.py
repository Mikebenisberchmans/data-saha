import sys
from app.dashboard.specification import generate_dashboard

question = " ".join(sys.argv[1:]) or "Show me a dashboard of orders revenue by status"
result = generate_dashboard(question)

print("=== SPECIFICATION ===")
print(result.specification.model_dump_json(indent=2))
print()
print("=== DATASETS ===")
for d in result.datasets:
    print(d.model_dump_json(indent=2))
