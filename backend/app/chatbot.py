import os
import json
import re
from uuid import uuid4

from dotenv import load_dotenv
from groq import Groq

from app.tool_registry import (
    tools,
    tool_definitions
)

# Pydantic Validation
from pydantic import ValidationError
from app.validation.tool_registry import (
    tool_input_models
)

# Long-Term Memory
from app.services.memory_service import (
    save_memory,
    semantic_search_memories,
    check_duplicate_memory,
    update_memory,
    update_memory_importance,
    delete_memory,
    supersede_memory,
)

from app.services.memory_conflict_service import (
    extract_exclusive_fact,
    find_conflicting_memories,
)

from app.services.temporary_memory_service import (
    add_task_context,
    add_task_context_once,
    get_task_context,
    clear_task_context,
)


# Embedding service for long-term memory
from app.services.embedding_service import (
    create_embedding
)



# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

MAX_AGENT_STEPS = 15

MAX_SAME_TOOL_FAILURES = 3

MAX_TOOL_RESULT_CHARS = 4000

# Conversation memory limit
MAX_CONVERSATION_MESSAGES = 10

# Long-term memory limit
MAX_LONG_TERM_MEMORIES = 10

# Temporary user ID
# Later this should come from your authenticated user
USER_ID = "muni"


# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_history = []


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an AI Software Engineer assistant.

Your job is to help the user investigate software projects,
understand code, debug problems, perform calculations,
retrieve current information, and reason about technical
issues.

You have access to these tools:

1. calculator
2. get_current_time
3. read_file
4. search_code
5. web_search


============================================================
GENERAL TOOL RULES
============================================================

Use a tool when the user's request requires information
or an action that the tool can provide.

Do not call a tool when you already have enough information
to answer the user's question.

Do not invent tool results.

Do not claim that a tool was used when it was not used.

After receiving a tool result:

- Analyze the result.
- Use the result to continue solving the user's request.
- Do not expose internal tool-selection reasoning.
- Do not expose unnecessary raw tool output.


============================================================
1. CALCULATOR
============================================================

Use calculator for mathematical calculations.

Examples:

- 25 * 40
- 1000 / 8
- 80 * 136 + 22 * 10

Do not manually perform complex calculations when the
calculator tool can provide the result.


============================================================
2. CURRENT TIME
============================================================

Use get_current_time when the user asks for:

- Current time
- Current date
- Today's date
- Time in a specific supported context


============================================================
3. FILE READER
============================================================

Use read_file only after you have a real project path.

A file name is not a path. temporary_memory_service.py
is not app/services/temporary_memory_service.py.

If the user names a file but does not include its
directories, do not call read_file yet. Search the
project first and use the path from that result.

Call read_file directly only when the user already
gave a path that includes directories, such as
app/main.py or app/services/temporary_memory_service.py.

Never invent file contents.

If the requested file cannot be read, do not guess
another path. Search for the file or symbol, then
read the path that search returns.

Do not request an entire source file. Always pass line_start
and line_end for one function, or a small range of about
40 to 70 lines. A path with no line range is not a read.

line_end is the last line actually returned.
If truncated is true, you have not seen past line_end.
One more read may start at next_line and stay the same
small size, only to finish that function. Then answer.
Do not keep reading the rest of the file.

The source is the text after the content: label.
That text is exact. Keep its spaces and indentation.
Do not fill unread gaps with reconstructed lines and
present the result as the file.


============================================================
4. CODE SEARCH
============================================================

Use search_code before read_file whenever you do not
already have a directory path.

Search once for the exact symbol the user asked about,
such as TASK_CONTEXT_TTL. Do not search for the file
name, and do not read a guessed file name first.

search_code returns the real file path, line number,
and the matching line. Those matches already include
the assignment and every use, including redis.expire
and ex=.

After that one successful search:

1. Call read_file once on the returned path.
2. Set line_start a few lines before the function
   or assignment.
3. Set line_end at the end of that function, or about
   70 lines later. Do not set line_end at the end of
   the file.
4. Answer from that read.

Do not search again for expire, redis.expire, the
file name, or any other text that already appeared
in the matches. Read the same file again only when
the first read was truncated before the function ended,
and keep that second read to the same small size.

A question about a value, a unit conversion, and what
happens when it expires is three steps:

1. search_code for the symbol.
2. read_file once across the definition and the
   expiration lines.
3. Answer.

Convert seconds to minutes in the answer. If the code
calls redis.expire or sets ex= again when a message
is appended, the TTL is refreshed on that write and
the key expires after that period of inactivity.

When investigating any other project question:

1. Search once for the function or phrase you need.
2. Read a small line range around that function once.
3. Answer once those lines are visible.
4. Never invent source-code contents.
5. Never read the entire file.


============================================================
5. WEB SEARCH
============================================================

Use web_search when the user requests information that
requires current or external information.

Use web_search when:

- The user explicitly says "search the web"
- The user says "search online"
- The user asks for latest information
- The user asks for current information
- The user asks for today's news
- The user asks about recent releases
- The user asks about current versions
- The user asks about current documentation
- The answer depends on information that may have changed
- The user asks about a current library, framework, API,
  service, or technology version

If the user explicitly asks you to search the web,
you MUST call web_search.

Do not answer a web-search request using internal knowledge
alone.

After receiving search results:

- Analyze the results.
- Do not expose raw HTML.
- Do not invent facts.
- Use the most relevant information.
- If the results are insufficient, perform another useful
  search with a meaningfully different query.
- Do not repeatedly search using the same or nearly
  identical query.
- If web_search repeatedly fails, stop trying and explain
  that the search tool could not retrieve the required
  information.
- Then provide the best possible final answer based on
  the available information.


============================================================
CONVERSATION MEMORY
============================================================

You can see previous messages from the current conversation.

Use previous conversation messages when they are relevant
to the user's current question.

Do not assume information that is not present in the
conversation.

Conversation memory only lasts while this application is
running.


============================================================
LONG-TERM MEMORY
============================================================

The application may provide relevant persistent memories
about the user.

Use long-term memory only when it is relevant to the
current request.

Do not invent information that is not present in memory.

Do not treat an irrelevant retrieved memory as fact.

The application performs semantic retrieval and memory
quality filtering before providing long-term memories.

Long-term memory persists after the application is
restarted.


============================================================
MEMORY WRITE RULE
============================================================

When the user explicitly says:

"Remember that ..."

the application handles long-term memory saving
automatically.

The application performs:

1. Memory extraction
2. Embedding creation
3. Duplicate detection
4. Memory saving when appropriate

Do NOT call a tool named save_memory.

Do NOT attempt to save memories through tool calling.

Do NOT automatically save every normal conversation message.

Only explicit "remember that" requests are saved at this
stage.

After the application processes a memory request, simply
acknowledge it naturally.


============================================================
MEMORY READ RULE
============================================================

Long-term memory may be provided to you as system context.

Use it when it is relevant to the user's current request.

Do not force unrelated memories into your answer.

If the required information is not present in the available
memory, say that you do not have that information.

Never invent personal information about the user.


============================================================
AGENT BEHAVIOR
============================================================

Think about what is required to answer the user's request.

If a tool is necessary:

1. Select the appropriate tool.
2. Use the tool with valid arguments.
3. Analyze the result.
4. Continue if another tool is required.
5. Provide the final answer.

If multiple tools are required, use them as needed.

Do not stop after the first tool if the user's request
requires additional investigation.

Do not call unnecessary tools.

A bare file name is not enough to call read_file.
Search first, read the returned path once, then answer.

You have a limited number of steps. Once one search and
one read show the value and the code that uses it, give
the final answer. Do not keep searching after that.


============================================================
TOOL FAILURE
============================================================

If a tool fails:

- Understand the error.
- Do not invent a successful result.
- If another approach is available, try it.
- If the tool repeatedly fails, stop and explain the
  limitation clearly.


============================================================
FINAL ANSWER
============================================================

Always provide a final natural-language answer after
tool execution.

Be clear and concise.

Do not expose internal tool-selection reasoning.

Do not expose unnecessary raw tool output.

Do not claim actions were completed when they were not.

When investigating code, explain the actual evidence found
in the project.

Quote source only from text a tool returned. The lines
after the content: label are the file. If you did not
read a line, do not display it as source. A snippet you
assemble from memory or from an explanation is a
reconstruction. Call it an illustration. Do not call it
the function, the implementation, or the exact code.

Redis Lua atomicity is not rollback.

Atomic execution means other clients cannot interleave
commands while a script runs.

Rollback means earlier writes are undone when a later
command fails. Lua scripting provides the first
guarantee, not the second. A successful command inside
a script remains applied if a later command fails.
A Redis transaction has the same limit: MULTI/EXEC does
not undo an earlier command when a later one fails.

When you need add_task_context_once, read that function
only: a small line range around its definition. Do not
read the entire source file. That function sends one EVAL
script, and that script is the implementation. Do not
replace it with a shorter example and present that example
as the source.

The script claims the deduplication key with SET NX and
the task-context TTL, appends the JSON message with RPUSH
(including request_id), trims the list to the last 10
messages with LTRIM, and applies the same TTL to the list
with EXPIRE.

These client outcomes follow from that script:

- Python crashes before EVAL is sent: Redis is unchanged.
- Redis finishes the script and the reply is lost: the
  dedupe key and the message can both already be stored.
  Do not describe that as a failed write, and do not say
  the handler deletes the dedupe key. A retry must use the
  same deduplication key. SET NX then rejects a second append.
- RPUSH fails after SET: the script deletes the dedupe key
  so a retry can claim it. That delete is explicit
  compensation, not automatic rollback. If the delete also
  fails, the key remains until its TTL and retries are
  rejected as duplicates.
- LTRIM or EXPIRE fails after RPUSH: the message stays in
  the list and the dedupe key stays. Those writes are not
  undone.

This is not exactly-once delivery. A lost reply is still
ambiguous. The idempotent key makes a completed script
safe to retry.

Once that function has been read, answer from it. Do not
search again for a rollback the script does not perform.


============================================================
CURRENT PROJECT ROLE
============================================================

You are being developed as an AI Software Engineer /
Bug Investigation Agent.

The current project focuses on:

- Tool calling
- Agent loops
- Tool registry
- Pydantic validation
- Error handling
- Conversation memory
- Long-term memory
- Embeddings
- Semantic retrieval
- Memory quality
- Memory deduplication

The toolset will evolve as the project becomes more
advanced.

Use only the tools currently provided by the application.
Do not assume future tools exist.
"""


# ============================================================
# LIMIT TOOL RESULT
# ============================================================

def limit_tool_result(result):

    if isinstance(result, (dict, list)):

        result = json.dumps(
            result,
            ensure_ascii=False
        )

    else:

        result = str(result)


    if len(result) <= MAX_TOOL_RESULT_CHARS:

        return result


    return (
        result[:MAX_TOOL_RESULT_CHARS]
        + "\n\n"
        "[Tool result truncated because it was too large.]"
    )


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(tool_name, tool_arguments):

    print(
        f"\n[Tool selected: {tool_name}]"
    )

    print(
        f"[Arguments: {tool_arguments}]"
    )


    # ========================================================
    # FIND TOOL
    # ========================================================

    tool_function = tools.get(
        tool_name
    )

    if tool_function is None:

        print(
            f"[Tool not found: {tool_name}]"
        )

        return {
            "success": False,
            "error": f"Tool not found: {tool_name}"
        }


    # ========================================================
    # FIND VALIDATION MODEL
    # ========================================================

    input_model = tool_input_models.get(
        tool_name
    )

    if input_model is None:

        print(
            f"[Validation model not found: {tool_name}]"
        )

        return {
            "success": False,
            "error": (
                f"No validation model "
                f"found for tool: {tool_name}"
            )
        }


    # ========================================================
    # VALIDATE TOOL ARGUMENTS
    # ========================================================

    try:

        validated_arguments = input_model(
            **tool_arguments
        )

        print(
            "[Arguments validated successfully]"
        )

    except ValidationError as e:

        print(
            "[Validation error]"
        )

        print(e)

        return {
            "success": False,
            "error": "Invalid tool arguments",
            "details": e.errors()
        }


    # ========================================================
    # EXECUTE TOOL
    # ========================================================

    try:

        result = tool_function(
            **validated_arguments.model_dump()
        )


        result = limit_tool_result(
            result
        )


        print(
            f"[Tool result size: "
            f"{len(result)} characters]"
        )


        print(
            f"[Tool result preview]: "
            f"{result[:1000]}"
        )


        return {
            "success": True,
            "result": result
        }


    except Exception as e:

        print(
            f"[Tool execution error]: {e}"
        )

        return {
            "success": False,
            "error": "Tool execution error",
            "details": str(e)
        }


# ============================================================
# ADD MESSAGE TO CONVERSATION MEMORY
# ============================================================

def add_to_conversation_memory(
    role,
    content
):

    conversation_history.append({

        "role": role,

        "content": content

    })


    # ========================================================
    # LIMIT MEMORY SIZE
    # ========================================================

    if len(conversation_history) > MAX_CONVERSATION_MESSAGES:

        del conversation_history[
            :-MAX_CONVERSATION_MESSAGES
        ]


# ============================================================
# GET CONVERSATION MEMORY
# ============================================================

def get_conversation_memory():

    return conversation_history.copy()


def restore_task_context_from_redis(user_id):
    """Restore Redis context only when local history is empty."""

    if conversation_history:
        return

    try:
        saved_context = get_task_context(user_id)

        for message in saved_context[-MAX_CONVERSATION_MESSAGES:]:
            if (
                isinstance(message, dict)
                and message.get("role") in ("user", "assistant")
                and isinstance(message.get("content"), str)
            ):
                # Append directly so restored messages aren't
                # written back to Redis a second time.
                conversation_history.append({
                    "role": message["role"],
                    "content": message["content"],
                })

        print(
            f"[Redis context restored: "
            f"{len(conversation_history)} messages]"
        )

    except Exception as e:
        # Redis failure must not stop the existing chatbot.
        print(f"[Redis context restore failed]: {e}")


def save_task_context_to_redis(
    role,
    content,
    user_id,
    request_id=None,
):
    """Save a message, optionally deduplicating by request ID."""
    try:
        if request_id:
            result = add_task_context_once(
                user_id=user_id,
                request_id=request_id,
                role=role,
                content=content,
            )
        else:
            # Preserve the existing behavior for older call sites.
            result = add_task_context(
                user_id=user_id,
                role=role,
                content=content,
            )

        if not result.get("success"):
            print("[Redis context save was unsuccessful]")
        elif result.get("duplicate"):
            print(
                f"[Duplicate Redis message skipped: "
                f"{request_id}:{role}]"
            )

    except Exception as e:
        print(f"[Redis context save failed]: {e}")


# ============================================================
# CLEAR CONVERSATION MEMORY
# ============================================================

def clear_conversation_memory():

    conversation_history.clear()

    try:
        clear_task_context(USER_ID)
        print("[Redis task context cleared]")
    except Exception as e:
        print(f"[Redis context clear failed]: {e}")

    print(
        "[Conversation memory cleared]"
    )


# ============================================================
# LOAD LONG-TERM MEMORY
# ============================================================

def load_long_term_memory(
    user_id,
    query
):

    print(
        "\n[Loading long-term memory...]"
    )

    result = semantic_search_memories(
        user_id=user_id,
        query=query,
        limit=MAX_LONG_TERM_MEMORIES
    )


    if not result.get(
        "success",
        False
    ):

        print(
            "[Long-term memory unavailable]"
        )

        return []


    memories = result.get(
        "memories",
        []
    )


    print(
        f"[Long-term memories loaded: "
        f"{len(memories)}]"
    )


    return memories


# ============================================================
# BUILD LONG-TERM MEMORY CONTEXT
# ============================================================

def build_memory_context(
    memories
):

    if not memories:

        return ""


    memory_lines = []


    for item in memories:

        memory = item.get(
            "memory"
        )

        if memory:

            importance = item.get("importance", 3)
            confidence = item.get("confidence", 0.75)

            memory_lines.append(
                f"- [Importance: {importance}/5] "
                f"[Confidence: {confidence:.2f}] "
                f"{memory}"
            )


    if not memory_lines:

        return ""


    return (
        "Known information about the user "
        "from long-term memory:\n\n"
        + "\n".join(memory_lines)
    )



# ============================================================
# UPDATE LONG-TERM MEMORY
# ============================================================


def try_delete_explicit_memory(user_message, user_id):
    """Delete a matching long-term memory only when the match is clear."""

    patterns = [
        r"^\s*(?:please\s+)?forget(?:\s+that)?\s+(.+?)\s*[.!?]?\s*$",
        r"^\s*(?:please\s+)?delete\s+(?:the\s+)?memory(?:\s+(?:about|for))?\s+(.+?)\s*[.!?]?\s*$",
        r"^\s*(?:please\s+)?delete\s+my\s+(.+?)\s*[.!?]?\s*$",
        r"^\s*(?:please\s+)?remove\s+(?:the\s+)?memory(?:\s+(?:about|for))?\s+(.+?)\s*[.!?]?\s*$",
    ]

    subject = None

    for pattern in patterns:
        match = re.match(pattern, user_message, re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            break

    # This is not a memory deletion request.
    if not subject:
        return None

    # Clean up the requested memory description.
    subject = re.sub(r"^(?:my|the)\s+", "", subject, flags=re.IGNORECASE)
    subject = re.sub(r"\s+memory$", "", subject, flags=re.IGNORECASE)
    subject = subject.strip(" .!?")

    if not subject:
        return None

    print("\n[Memory deletion request detected]")
    print(f"[Memory subject: {subject}]")

    # Search the current user's memories.
    search_result = semantic_search_memories(
        user_id=user_id,
        query=subject,
        limit=MAX_LONG_TERM_MEMORIES,
        min_score=0.0,
    )

    if not search_result.get("success"):
        return {
            "success": False,
            "message": "I couldn't search your memories. Nothing was deleted.",
        }

    # Match the subject against the actual stored memory text.
    stop_words = {"my", "the", "is", "a", "an", "of", "for", "about"}
    subject_words = {
        word.lower()
        for word in re.findall(r"\w+", subject)
        if word.lower() not in stop_words
    }

    if not subject_words:
        return {
            "success": False,
            "message": "Please specify which memory you want to delete. Nothing was deleted.",
        }

    candidates = search_result.get("memories", [])

    matches = [
        item
        for item in candidates
        if subject_words.issubset(
            set(re.findall(r"\w+", item.get("memory", "").lower()))
        )
    ]

    if not matches:
        return {
            "success": False,
            "message": (
                f"I couldn't confidently find a memory about '{subject}'. "
                "Nothing was deleted."
            ),
        }

    # Never choose arbitrarily if multiple memories match.
    if len(matches) > 1:
        return {
            "success": False,
            "message": (
                f"I found multiple memories matching '{subject}'. "
                "Please describe the memory more specifically. Nothing was deleted."
            ),
        }

    target = matches[0]
    memory_id = target.get("memory_id")

    if not memory_id:
        return {
            "success": False,
            "message": "The memory ID is missing. Nothing was deleted.",
        }

    # Delete only this user's memory.
    delete_result = delete_memory(memory_id)

    if not delete_result.get("success"):
        print(f"[Memory deletion failed]: {delete_result.get('error')}")

        return {
            "success": False,
            "message": "I couldn't delete that memory from the database.",
        }

    print(f"[Memory deleted successfully: {memory_id}]")

    return {
        "success": True,
        "message": f"I've deleted the memory matching '{subject}'.",
    }


def try_update_explicit_memory(user_message, user_id):
    pattern = (
        r"^\s*update my (.+?) to (.+?)[.!?]*\s*$"
    )

    match = re.match(
        pattern,
        user_message,
        re.IGNORECASE
    )

    if not match:
        return None

    subject = match.group(1).strip()
    new_value = match.group(2).strip()

    # Build a consistent memory sentence.
    new_memory = f"My {subject} is {new_value}"

    print("\n[Memory update request detected]")
    print(f"[Memory field: {subject}]")
    print(f"[New memory: {new_memory}]")

    # Find existing memories related to this field.
    search_result = semantic_search_memories(
        user_id=user_id,
        query=subject,
        limit=MAX_LONG_TERM_MEMORIES,
        min_score=0.0
    )

    if not search_result.get("success"):
        return {
            "success": False,
            "message": "I couldn't search your existing memories."
        }

    # Only update a memory that explicitly contains
    # the same field wording. Never pick one match arbitrarily.
    subject_words = subject.lower().split()

    candidates = search_result.get("memories", [])

    matches = [
        item for item in candidates
        if all(
            word in item.get("memory", "").lower()
            for word in subject_words
        )
    ]

    if not matches:
        return {
            "success": False,
            "message": (
                "I couldn't find an existing memory "
                f"for '{subject}'. No memory was changed."
            )
        }

    if len(matches) > 1:
        return {
            "success": False,
            "message": (
                f"I found multiple memories matching '{subject}'. "
                "Please describe the memory more specifically. "
                "Nothing was changed."
            ),
        }

    existing = matches[0]

    # Generate a new embedding for the updated text.
    try:
        new_embedding = create_embedding(new_memory)
    except Exception as e:
        print(f"[Memory update embedding error]: {e}")
        return {
            "success": False,
            "message": "I couldn't generate the updated memory embedding."
        }

    fact = extract_exclusive_fact(new_memory)

    result = supersede_memory(
        user_id=user_id,
        memory_id=existing["memory_id"],
        memory=new_memory,
        embedding=new_embedding,
        fact_key=fact[0] if fact else None,
    )

    if not result.get("success"):
        return {
            "success": False,
            "message": "The memory could not be updated in MongoDB."
        }

    print("[Memory update completed successfully]")

    return {
        "success": True,
        "message": f"Updated memory: {new_memory}"
    }





def calculate_memory_importance(memory):
    """Assign a simple importance score from 1 to 5."""

    text = memory.lower()

    # Short-lived or one-time information.
    low_importance_terms = [
        "one-time calculation",
        "current time",
        "what is the weather",
        "today's weather",
        "just for today",
        "temporary task",
    ]

    if any(term in text for term in low_importance_terms):
        return 1

    # Long-term identity, career, and project information.
    high_importance_terms = [
        "career goal",
        "long-term goal",
        "my goal is",
        "i want to become",
        "i am building",
        "i am developing",
        "my main project",
        "my profession",
    ]

    if any(term in text for term in high_importance_terms):
        return 5

    # Ongoing learning, work, and stable preferences.
    medium_high_terms = [
        "i am learning",
        "i am working on",
        "i am studying",
        "i prefer",
        "my preferred",
        "my learning goal",
        "i plan to",
    ]

    if any(term in text for term in medium_high_terms):
        return 4

    # Ordinary preferences and personal facts.
    return 3


def calculate_memory_confidence(memory):
    """Estimate confidence from how the memory is expressed."""
    text = memory.lower()

    uncertain_phrases = [
        "i think",
        "i guess",
        "maybe",
        "might be",
        "not sure",
        "possibly",
        "probably",
        "i am unsure",
    ]

    if any(phrase in text for phrase in uncertain_phrases):
        return 0.60

    direct_phrases = [
        "my favorite",
        "i prefer",
        "i am building",
        "i am learning",
        "i work as",
        "my goal is",
        "remember that",
    ]

    if any(phrase in text for phrase in direct_phrases):
        return 0.95

    return 0.75


# ============================================================
# SAVE EXPLICIT LONG-TERM MEMORY
# ============================================================

def try_save_explicit_memory(
    user_message,
    user_id
):

    text = user_message.strip()

    prefix = "remember that "

    # ========================================================
    # CHECK WHETHER USER EXPLICITLY REQUESTED MEMORY
    # ========================================================

    if not text.lower().startswith(
        prefix
    ):

        return False

    # ========================================================
    # EXTRACT MEMORY
    # ========================================================

    memory = text[
        len(prefix):
    ].strip()

    # Remove common ending punctuation so that: 
    # "Remember that I am learning RAG."
    # and 
    # "Remember that I am learning RAG"
    # are treated as the same memory.
    memory = memory.rstrip(".!?")

    if not memory:

        return False

    print(
        "\n[Explicit memory request detected]"
    )

    print(
        f"[Memory content: {memory}]"
    )

    # ========================================================
    # CREATE EMBEDDING
    # ========================================================

    try:

        print(
            "[Creating memory embedding...]"
        )

        embedding = create_embedding(
            memory
        )

        print(
            "[Memory embedding created successfully]"
        )

    except Exception as e:

        print(
            f"[Memory embedding failed]: {e}"
        )

        return False

    # ========================================================
    # CHECK FOR DUPLICATE MEMORY
    # ========================================================

    print(
        "[Checking for duplicate memory...]"
    )

    duplicate_result = check_duplicate_memory(
        user_id=user_id,
        embedding=embedding,
        duplicate_threshold=0.90
    )

    # ========================================================
    # HANDLE DUPLICATE CHECK FAILURE
    # ========================================================

    if not duplicate_result.get(
        "success",
        False
    ):

        print(
            "[Duplicate check failed]"
        )

        print(
            f"[Error]: {duplicate_result.get('error', 'Unknown error')}"
        )

        return False

    # ========================================================
    # SAME-SUBJECT CONFLICTS BEFORE DUPLICATE SHORT-CIRCUIT
    # ========================================================

    incoming_fact = extract_exclusive_fact(memory)
    conflicts = find_conflicting_memories(
        memory,
        duplicate_result.get("candidates", []),
    )

    if incoming_fact and conflicts:
        conflict = conflicts[0]
        conflict_group_id = (
            conflict.get("conflict_group_id") or uuid4().hex
        )

        importance = calculate_memory_importance(memory)
        confidence = calculate_memory_confidence(memory)

        pending_result = save_memory(
            user_id=user_id,
            memory=memory,
            embedding=embedding,
            importance=importance,
            confidence=confidence,
            status="needs_review",
            fact_key=incoming_fact[0],
            conflict_type="possible_contradiction",
            conflict_group_id=conflict_group_id,
            conflict_with_memory_id=conflict["memory_id"],
            version=1,
        )

        if not pending_result.get("success"):
            print(
                "[Conflict review save failed]: "
                f"{pending_result.get('error', 'Unknown error')}"
            )
            return False

        readable_key = incoming_fact[0].replace("_", " ")
        return {
            "handled": True,
            "status": "needs_review",
            "message": (
                "I found a possible conflict in your saved information.\n\n"
                f"Existing memory: {conflict['memory']}\n"
                f"New statement: {memory}\n\n"
                "I saved the new statement for review and did not replace "
                "the active memory. To confirm that the new value replaces "
                f"the old one, send: `Update my {readable_key} to "
                f"{incoming_fact[1]}`."
            ),
        }

    # ========================================================
    # DUPLICATE MEMORY FOUND
    # ========================================================

    if duplicate_result.get("is_duplicate", False):
        print("[Memory already exists]")

        incoming_importance = calculate_memory_importance(memory)
        existing_importance = duplicate_result.get("importance", 3)

        # Upgrade importance only; never lower an existing score.
        if incoming_importance > existing_importance:
            update_result = update_memory_importance(
                user_id=user_id,
                memory_id=duplicate_result["memory_id"],
                importance=incoming_importance,
            )

            if update_result.get("success"):
                print(
                    f"[Memory importance upgraded]: "
                    f"{existing_importance}/5 -> {incoming_importance}/5"
                )
            else:
                print(
                    f"[Memory importance update failed]: "
                    f"{update_result.get('error', 'Unknown error')}"
                )
        else:
            print(
                f"[Memory importance unchanged]: "
                f"{existing_importance}/5"
            )

        print("[Skipping duplicate memory save]")
        return True

    # ========================================================
    # SAVE NEW MEMORY + EMBEDDING
    # ========================================================

    print(
        "[Memory is unique]"
    )

    importance = calculate_memory_importance(memory)

    print(f"[Calculated memory importance]: {importance}/5")

    confidence = calculate_memory_confidence(memory)
    print(f"[Memory confidence]: {confidence:.2f}")

    fact = extract_exclusive_fact(memory)

    result = save_memory(
        user_id=user_id,
        memory=memory,
        embedding=embedding,
        importance=importance,
        confidence=confidence,
        fact_key=fact[0] if fact else None,
    )

    if result.get(
        "success",
        False
    ):

        print(
            "[Long-term memory saved successfully]"
        )

        return True

    print(
        "[Long-term memory save failed]"
    )

    return False


# ============================================================
# CHAT WITH AI
# ============================================================

def chat_with_ai(user_message, request_id=None):
    if user_message is None or not str(user_message).strip():
        return "Please enter a message."

    if request_id is None:
        request_id = str(uuid4())

    restore_task_context_from_redis(USER_ID)

    # Handle explicit memory deletion first.
    delete_result = try_delete_explicit_memory(
        user_message,
        USER_ID
    )

    if delete_result is not None:
        add_to_conversation_memory("user", user_message)
        save_task_context_to_redis(
            "user",
            user_message,
            USER_ID,
            request_id=request_id,
        )

        answer = delete_result["message"]

        add_to_conversation_memory("assistant", answer)
        save_task_context_to_redis(
            "assistant",
            answer,
            USER_ID,
            request_id=request_id,
        )

        return answer

    # Handle explicit memory updates first.
    update_result = try_update_explicit_memory(
        user_message,
        USER_ID
    )

    if update_result is not None:
        add_to_conversation_memory(
            "user",
            user_message
        )
        save_task_context_to_redis(
            "user",
            user_message,
            USER_ID,
            request_id=request_id,
        )

        answer = update_result["message"]

        add_to_conversation_memory(
            "assistant",
            answer
        )
        save_task_context_to_redis(
            "assistant",
            answer,
            USER_ID,
            request_id=request_id,
        )

        return answer

    memory_save_result = try_save_explicit_memory(
        user_message,
        USER_ID,
    )

    if (
        isinstance(memory_save_result, dict)
        and memory_save_result.get("handled")
    ):
        add_to_conversation_memory("user", user_message)
        save_task_context_to_redis(
            "user",
            user_message,
            USER_ID,
            request_id=request_id,
        )

        answer = memory_save_result["message"]

        add_to_conversation_memory("assistant", answer)
        save_task_context_to_redis(
            "assistant",
            answer,
            USER_ID,
            request_id=request_id,
        )

        return answer


    # ========================================================
    # ADD USER MESSAGE TO CONVERSATION MEMORY
    # ========================================================

    add_to_conversation_memory(

        "user",

        user_message

    )

    save_task_context_to_redis(
        "user",
        user_message,
        USER_ID,
        request_id=request_id,
    )


    # ========================================================
    # LOAD LONG-TERM MEMORY
    # ========================================================

    long_term_memories = (
        load_long_term_memory(
            USER_ID,
            user_message
        )
    )


    # ========================================================
    # BUILD MEMORY CONTEXT
    # ========================================================

    memory_context = (
        build_memory_context(
            long_term_memories
        )
    )

    if memory_context:
        print("\n[Memory context sent to LLM]")
        print(memory_context)


    # ========================================================
    # CREATE LLM MESSAGES
    # ========================================================

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

    ]


    # ========================================================
    # ADD LONG-TERM MEMORY
    # ========================================================

    if memory_context:

        messages.append({

            "role": "system",

            "content": memory_context

        })


    # ========================================================
    # ADD CONVERSATION MEMORY
    # ========================================================

    messages.extend(
        get_conversation_memory()
    )


    # ========================================================
    # TOOL FAILURE TRACKING
    # ========================================================

    tool_failure_counts = {}


    # ========================================================
    # AGENT LOOP
    # ========================================================

    for step in range(
        MAX_AGENT_STEPS
    ):

        print(
            f"\n[Agent step "
            f"{step + 1}/"
            f"{MAX_AGENT_STEPS}]"
        )


        print(
            "[Calling LLM...]"
        )


        # ====================================================
        # CALL LLM
        # ====================================================

        tool_choice = "auto"

        if step == MAX_AGENT_STEPS - 1:

            messages.append({
                "role": "system",
                "content": (
                    "This is your last step. "
                    "Do not call any tools. "
                    "Answer from the code already returned "
                    "by the tools."
                )
            })

            tool_choice = "none"

        try:

            response = client.chat.completions.create(

                model=MODEL,

                messages=messages,

                tools=tool_definitions,

                tool_choice=tool_choice

            )

        except Exception as e:

            print(
                f"[LLM error]: {e}"
            )

            return (
                "The AI model request failed: "
                f"{e}"
            )


        # ====================================================
        # GET RESPONSE MESSAGE
        # ====================================================

        response_message = (
            response.choices[0].message
        )


        # ====================================================
        # FINAL ANSWER
        # ====================================================

        if not response_message.tool_calls:

            print(
                "[No more tools needed]"
            )


            final_answer = (
                response_message.content
                or
                "I could not generate a response."
            )


            # ================================================
            # SAVE AI RESPONSE TO CONVERSATION MEMORY
            # ================================================

            add_to_conversation_memory(

                "assistant",

                final_answer

            )

            save_task_context_to_redis(
                "assistant",
                final_answer,
                USER_ID,
                request_id=request_id,
            )


            return final_answer


        # ====================================================
        # TOOL CALLS FOUND
        # ====================================================

        print(
            f"\n[LLM selected "
            f"{len(response_message.tool_calls)} "
            f"tool(s)]"
        )


        # ====================================================
        # SAVE ASSISTANT TOOL-CALL MESSAGE
        # ====================================================

        messages.append(
            response_message
        )


        # ====================================================
        # EXECUTE TOOLS
        # ====================================================

        for tool_call in (
            response_message.tool_calls
        ):

            tool_name = (
                tool_call.function.name
            )


            # =================================================
            # PARSE TOOL ARGUMENTS
            # =================================================

            try:

                tool_arguments = json.loads(

                    tool_call.function.arguments

                )


                print(
                    f"[Parsed arguments: "
                    f"{tool_arguments}]"
                )


            except json.JSONDecodeError as e:

                print(
                    f"[JSON parsing error]: {e}"
                )


                tool_result = {

                    "success": False,

                    "error":
                        "Invalid tool arguments",

                    "details":
                        str(e)

                }


                messages.append({

                    "role": "tool",

                    "tool_call_id":
                        tool_call.id,

                    "name":
                        tool_name,

                    "content":
                        json.dumps(
                            tool_result,
                            ensure_ascii=False
                        )

                })


                continue


            # =================================================
            # EXECUTE TOOL
            # =================================================

            tool_result = execute_tool(

                tool_name,

                tool_arguments

            )


            # =================================================
            # TRACK TOOL FAILURES
            # =================================================

            if not tool_result.get(
                "success",
                False
            ):

                current_failures = (
                    tool_failure_counts.get(
                        tool_name,
                        0
                    )
                )


                current_failures += 1


                tool_failure_counts[
                    tool_name
                ] = current_failures


                print(
                    f"[Tool failure count: "
                    f"{tool_name} = "
                    f"{current_failures}]"
                )


            else:

                tool_failure_counts[
                    tool_name
                ] = 0


            # =================================================
            # SEND TOOL RESULT TO LLM
            # =================================================

            messages.append({

                "role": "tool",

                "tool_call_id":
                    tool_call.id,

                "name":
                    tool_name,

                "content":
                    json.dumps(
                        tool_result,
                        ensure_ascii=False
                    )

            })


            # =================================================
            # STOP REPEATED FAILURES
            # =================================================

            if (
                tool_failure_counts.get(
                    tool_name,
                    0
                )
                >= MAX_SAME_TOOL_FAILURES
            ):

                print(
                    f"[Stopping agent loop: "
                    f"{tool_name} failed "
                    f"{MAX_SAME_TOOL_FAILURES} times]"
                )


                final_answer = (

                    f"The tool '{tool_name}' "

                    f"failed "
                    f"{MAX_SAME_TOOL_FAILURES} times. "

                    "I could not complete the request."

                )


                add_to_conversation_memory(

                    "assistant",

                    final_answer

                )

                save_task_context_to_redis(
                    "assistant",
                    final_answer,
                    USER_ID,
                    request_id=request_id,
                )


                return final_answer


    # ========================================================
    # MAX STEPS REACHED
    # ========================================================

    print(
        "[Maximum agent steps reached]"
    )


    final_answer = (

        "I reached the maximum number of "
        "agent steps before completing the request."

    )


    add_to_conversation_memory(

        "assistant",

        final_answer

    )

    save_task_context_to_redis(
        "assistant",
        final_answer,
        USER_ID,
        request_id=request_id,
    )


    return final_answer