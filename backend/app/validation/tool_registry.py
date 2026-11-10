from app.validation.tool_models import (
    CalculatorInput,
    ConfirmMemoryUpdateInput,
    CurrentTimeInput,
    DetectMemoryConflictInput,
    ExpandMemorySummaryInput,
    ForgetMemoryInput,
    InspectMemoryMetadataInput,
    ReadFileInput,
    ReinforceMemoryInput,
    RequestMemoryUpdateInput,
    SaveMemoryInput,
    SearchCodeInput,
    SearchMemoryInput,
    SearchMemorySummariesInput,
    SummarizeMemoriesInput,
    WebSearchInput,
)


tool_input_models = {
    "calculator": CalculatorInput,
    "get_current_time": CurrentTimeInput,
    "read_file": ReadFileInput,
    "search_code": SearchCodeInput,
    "web_search": WebSearchInput,
    "search_memory": SearchMemoryInput,
    "save_memory": SaveMemoryInput,
    "detect_memory_conflict": DetectMemoryConflictInput,
    "request_memory_update": RequestMemoryUpdateInput,
    "confirm_memory_update": ConfirmMemoryUpdateInput,
    "reinforce_memory": ReinforceMemoryInput,
    "forget_memory": ForgetMemoryInput,
    "inspect_memory_metadata": InspectMemoryMetadataInput,
    "summarize_memories": SummarizeMemoriesInput,
    "search_memory_summaries": SearchMemorySummariesInput,
    "expand_memory_summary": ExpandMemorySummaryInput,
}

# Tools that must receive authenticated user_id from the orchestrator.
MEMORY_TOOL_NAMES = {
    "search_memory",
    "save_memory",
    "detect_memory_conflict",
    "request_memory_update",
    "confirm_memory_update",
    "reinforce_memory",
    "forget_memory",
    "inspect_memory_metadata",
    "summarize_memories",
    "search_memory_summaries",
    "expand_memory_summary",
}
