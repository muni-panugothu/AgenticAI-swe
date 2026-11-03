

from app.tool_registry import tools

print("Available Tools:")
print("=" * 30)


for tool_name in tools:
    print(tool_name)

print("\nTesting Calculator Through Registry")
print("=" * 30)

result = tools["calculator"]("50 * 10")

print(result)
