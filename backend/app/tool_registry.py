from app.tools.calculator import calculator
from app.tools.time_tool import get_current_time
from app.tools.file_reader import read_file
from app.tools.code_search import search_code
from app.tools.web_search import web_search
from app.tools.memory_tools import (
    confirm_memory_update,
    detect_memory_conflict_tool,
    expand_memory_summary_tool,
    forget_memory,
    inspect_memory_metadata_tool,
    reinforce_memory_tool,
    request_memory_update,
    save_memory_tool,
    search_memory,
    search_memory_summaries_tool,
    summarize_memories_tool,
)


# --------------------------------
# Actual Python tools
# --------------------------------

tools = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "read_file": read_file,
    "search_code": search_code,
    "web_search": web_search,
    "search_memory": search_memory,
    "save_memory": save_memory_tool,
    "detect_memory_conflict": detect_memory_conflict_tool,
    "request_memory_update": request_memory_update,
    "confirm_memory_update": confirm_memory_update,
    "reinforce_memory": reinforce_memory_tool,
    "forget_memory": forget_memory,
    "inspect_memory_metadata": inspect_memory_metadata_tool,
    "summarize_memories": summarize_memories_tool,
    "search_memory_summaries": search_memory_summaries_tool,
    "expand_memory_summary": expand_memory_summary_tool,
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
            "Search the internet for current, recent, or "
            "externally verifiable information. Use for live "
            "facts that change over time: officeholders "
            "(e.g. chief minister), prices/exchange rates, "
            "latest news or developments, weather, sports "
            "results, and software release versions. Do not "
            "use for stable conceptual explanations "
            "(e.g. 'What is Python?') or for writing code."
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
},

    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the authenticated user's long-term memories with "
                "freshness-aware ranking. Use when personal facts may help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query for stored user facts.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max memories to return (1-10).",
                    },
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "inspect_memory_metadata",
            "description": (
                "Read-only inspection of stored memory metadata for one "
                "canonical fact key (memory_id, status, fact_key, "
                "normalized_value, conflict_group_id, user_id). "
                "Use ONLY when the user explicitly asks to inspect or verify "
                "metadata / conflict_group_id. Never modify records. "
                "Do not use for ordinary preference questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_key": {
                        "type": "string",
                        "description": (
                            "Canonical fact key such as preferred_database."
                        ),
                    },
                },
                "required": ["fact_key"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "summarize_memories",
            "description": (
                "Generate or refresh conflict-aware digests of the user's "
                "long-term memories into the memory_summaries collection. "
                "Never invents facts; unresolved conflicts stay unresolved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_type": {
                        "type": "string",
                        "description": (
                            "Optional: user, project, preferences, or general."
                        ),
                    },
                    "scope_key": {
                        "type": "string",
                        "description": (
                            "Optional project/topic scope, e.g. listifys or primary."
                        ),
                    },
                },
                "required": [],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "search_memory_summaries",
            "description": (
                "Search active memory summaries for the authenticated user. "
                "Outdated summaries are excluded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic to find in summaries.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max summaries to return (1-10).",
                    },
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "expand_memory_summary",
            "description": (
                "Load the original source memories for one summary_id "
                "(ownership-checked). Use when detail beyond the digest is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_id": {
                        "type": "string",
                        "description": "Summary document id.",
                    },
                },
                "required": ["summary_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Save an explicit user fact to long-term memory. "
                "Conflicts are staged for confirmation and never silently overwrite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "Canonical fact text to store.",
                    }
                },
                "required": ["memory"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "detect_memory_conflict",
            "description": (
                "Check whether a candidate fact contradicts an existing "
                "exclusive memory for this user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "Candidate fact to evaluate.",
                    }
                },
                "required": ["memory"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "request_memory_update",
            "description": (
                "Stage a proposed replacement for an existing memory. "
                "Does not apply the change until confirm_memory_update."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Existing active memory id to replace.",
                    },
                    "new_memory": {
                        "type": "string",
                        "description": "Proposed replacement fact.",
                    },
                },
                "required": ["memory_id", "new_memory"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "confirm_memory_update",
            "description": (
                "Apply a user-confirmed factual update using versioned "
                "supersede. Never overwrite history in place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Active memory id being replaced.",
                    },
                    "new_memory": {
                        "type": "string",
                        "description": "Confirmed replacement fact.",
                    },
                },
                "required": ["memory_id", "new_memory"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "reinforce_memory",
            "description": (
                "Explicitly reaffirm an existing memory to refresh freshness "
                "without changing its text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Active memory id to reinforce.",
                    }
                },
                "required": ["memory_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "forget_memory",
            "description": (
                "Soft-forget a memory for the authenticated user. "
                "The fact is marked forgotten, not hard-deleted by default."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "Memory id to forget.",
                    }
                },
                "required": ["memory_id"],
            },
        },
    },
]
