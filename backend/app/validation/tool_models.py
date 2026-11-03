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