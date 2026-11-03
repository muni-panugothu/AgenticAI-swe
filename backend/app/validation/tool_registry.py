from app.validation.tool_models import (
    CalculatorInput,
    CurrentTimeInput,
    ReadFileInput,
    SearchCodeInput,
    WebSearchInput
)


tool_input_models = {

    "calculator": CalculatorInput,

    "get_current_time": CurrentTimeInput,

    "read_file": ReadFileInput,

    "search_code": SearchCodeInput,

    "web_search": WebSearchInput

}