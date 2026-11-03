from app.tools.calculator import calculator
from app.tools.time_tool import get_current_time
from app.tools.file_reader import read_file
from app.tools.code_search import search_code
from app.tools.web_search import web_search


# --------------------------------
# Actual Python tools
# --------------------------------

tools = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "read_file": read_file,
    "search_code": search_code,
    "web_search": web_search
}


# --------------------------------
# Tool definitions for the LLM
# --------------------------------

tool_definitions = [

    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate."
                    }
                },
                "required": ["expression"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },

  {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read one function or a small line range from a project file by its real path, including directories, such as app/services/temporary_memory_service.py. A bare file name is not a path. If the user only named the file, call search_code first and use the path it returns. Always pass line_start and line_end. Do not request the entire file. The content after the content: label is exact source text: spaces, blank lines, and indentation are preserved. If truncated is true, one more read may start at next_line and cover only the rest of that function. line_end is the last line actually returned. Then answer.",
        "parameters": {
            "type": "object",
            "properties": {

                "file_path": {
                    "type": "string",
                    "description": "Directory path from the user or from search_code, such as app/services/temporary_memory_service.py. Never pass only the file name."
                },

                "line_start": {
                    "type": "integer",
                    "description": "First line of one function or a small range. Required. Do not omit this to read the whole file."
                },

                "line_end": {
                    "type": "integer",
                    "description": "Last line of that function or small range, at most about 70 lines after line_start. Required."
                }

            },

            "required": ["file_path"]
        }
    }
},

   
 {
    "type": "function",
    "function": {
        "name": "search_code",

        "description": (
            "Search the project before reading a file when "
            "you only know a file name or a symbol. Returns "
            "the real file path, line number, and matching "
            "line. Search once for the exact symbol. The "
            "matches already include the assignment and "
            "every use, including expire calls. Then call "
            "read_file once on that path for the function "
            "only, about 70 lines, and answer. Do not read "
            "the entire file. Do not search again for the "
            "file name or for calls that already appeared."
        ),

        "parameters": {
            "type": "object",

            "properties": {

                "search_text": {
                    "type": "string",
                    "description": (
                        "Text or function name to search for."
                    )
                },

                "folder_path": {
                    "type": "string",
                    "description": (
                        "Optional project folder to search. "
                        "Defaults to app."
                    )
                }
            },

            "required": [
                "search_text"
            ]
        }
    }
},

   {
    "type": "function",
    "function": {
        "name": "web_search",

        "description": (
            "Search the internet for current, "
            "recent, factual, or external information. "
            "Use this tool when the user explicitly asks "
            "to search the web, asks for latest/current/news "
            "information, or asks about information that may "
            "have changed recently."
        ),

        "parameters": {
            "type": "object",

            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise search query."
                    )
                }
            },

            "required": ["query"]
        }
    }
}
]