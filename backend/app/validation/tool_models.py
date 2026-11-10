from pydantic import BaseModel, Field, ConfigDict


class StrictToolInput(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


class CalculatorInput(StrictToolInput):

    expression: str = Field(
        description="Mathematical expression to calculate"
    )


class CurrentTimeInput(StrictToolInput):

    pass


class ReadFileInput(StrictToolInput):

    file_path: str = Field(
        description="Path to the file to read"
    )

    line_start: int | None = Field(
        default=None,
        description="First line of one function or a small range"
    )

    line_end: int | None = Field(
        default=None,
        description="Last line of that function or small range"
    )


class SearchCodeInput(StrictToolInput):

    search_text: str = Field(
        description="Text or function name to search for"
    )

    folder_path: str = Field(
        default="app",
        description="Folder to search"
    )


class WebSearchInput(StrictToolInput):

    query: str = Field(
        description="Search query"
    )


class SearchMemoryInput(StrictToolInput):

    query: str = Field(description="What to search for in long-term memory")
    limit: int = Field(default=5, ge=1, le=10)


class SaveMemoryInput(StrictToolInput):

    memory: str = Field(min_length=1, description="Canonical fact to store")


class DetectMemoryConflictInput(StrictToolInput):

    memory: str = Field(min_length=1, description="Candidate fact to check")


class RequestMemoryUpdateInput(StrictToolInput):

    memory_id: str = Field(min_length=1)
    new_memory: str = Field(min_length=1)


class ConfirmMemoryUpdateInput(StrictToolInput):

    memory_id: str = Field(min_length=1)
    new_memory: str = Field(min_length=1)


class ReinforceMemoryInput(StrictToolInput):

    memory_id: str = Field(min_length=1)


class ForgetMemoryInput(StrictToolInput):

    memory_id: str = Field(min_length=1)


class InspectMemoryMetadataInput(StrictToolInput):

    fact_key: str = Field(
        min_length=1,
        description=(
            "Canonical fact key to inspect, e.g. preferred_database. "
            "Read-only; returns actual stored metadata."
        ),
    )


class SummarizeMemoriesInput(StrictToolInput):

    summary_type: str | None = Field(
        default=None,
        description=(
            "Optional bucket: user, project, preferences, or general. "
            "Omit to refresh all buckets."
        ),
    )
    scope_key: str | None = Field(
        default=None,
        description=(
            "Optional topic/project scope such as listifys or primary. "
            "Omit to refresh every discovered scope."
        ),
    )


class SearchMemorySummariesInput(StrictToolInput):

    query: str = Field(description="What to look for in memory summaries")
    limit: int = Field(default=5, ge=1, le=10)


class ExpandMemorySummaryInput(StrictToolInput):

    summary_id: str = Field(
        min_length=1,
        description="Summary id whose original source memories should be loaded",
    )
