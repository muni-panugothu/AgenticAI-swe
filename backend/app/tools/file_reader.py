
import json


# def read_file(file_path):

#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             content = file.read()

#         return content
#     except Exception as e:

#         return f"Error:{e}"

#     except Exception as e:
#         return f"Error:{e}"


# Stay under the agent tool-result limit so the JSON
# is not cut off in the middle of the file.
MAX_READ_CHARS = 3200

# One function window. A larger span is what pushed the
# model request over the token limit.
MAX_LINE_SPAN = 70


def read_file(
    file_path,
    line_start=None,
    line_end=None
):

    try:

        if (
            not isinstance(line_start, int)
            or not isinstance(line_end, int)
            or line_start < 1
            or line_end < line_start
        ):
            return {
                "success": False,
                "error": (
                    "Provide line_start and line_end for one "
                    "function or a small line range. Do not "
                    "request the entire file."
                ),
            }

        span_limited = False

        if line_end - line_start + 1 > MAX_LINE_SPAN:
            line_end = line_start + MAX_LINE_SPAN - 1
            span_limited = True

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            lines = file.readlines()

        start = line_start - 1

        end = line_end

        if start < 0:
            start = 0

        if end > len(lines):
            end = len(lines)

        if start > end:
            start = end

        selected_lines = lines[start:end]

        content_parts = []
        used = 0
        line_was_cut = False

        for line in selected_lines:

            if (
                content_parts
                and used + len(line) > MAX_READ_CHARS
            ):
                break

            if (
                not content_parts
                and len(line) > MAX_READ_CHARS
            ):
                content_parts.append(
                    line[:MAX_READ_CHARS]
                )
                line_was_cut = True
                break

            content_parts.append(line)
            used += len(line)

        included = len(content_parts)
        truncated = (
            line_was_cut
            or included < len(selected_lines)
        )

        content = "".join(content_parts)

        metadata = {
            "success": True,
            "file_path": file_path,
            "line_start": start + 1,
            "line_end": start + included,
            "truncated": truncated,
        }

        if truncated:

            metadata["next_line"] = start + included + 1
            metadata["message"] = (
                "Only part of this small range was returned. "
                "Answer from these lines if they contain the "
                "function. If the function is cut off, read once "
                f"more from line {start + included + 1} for at most "
                f"{MAX_LINE_SPAN} lines. Do not request the entire file."
            )

        elif span_limited:

            metadata["message"] = (
                f"Requested range limited to {MAX_LINE_SPAN} lines. "
                "Inspect this function window and answer. "
                "Do not request the entire file."
            )

        # Keep the source after the metadata, with its original
        # newlines and indentation. Putting it inside JSON would
        # escape every line onto one long string, and spaces at
        # a wrap boundary (including indentation) get dropped.
        return (
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2
            )
            + "\ncontent:\n"
            + content
        )

    except FileNotFoundError:

        return {
            "success": False,
            "error": f"File not found: {file_path}"
        }

    except Exception as e:

        return {
            "success": False,
            "error": f"Error reading file: {str(e)}"
        }