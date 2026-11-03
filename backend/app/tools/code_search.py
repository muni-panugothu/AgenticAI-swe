import os


IGNORE_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
    "dist",
    "build"
}


SOURCE_EXTENSIONS = (
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".cs",
    ".go",
    ".html",
    ".css"
)


MAX_SEARCH_MATCHES = 20

MAX_LINE_CHARS = 180


def search_code(
    search_text,
    folder_path="app"
):

    matches = []
    truncated = False

    try:

        if not os.path.exists(folder_path):

            return {
                "success": False,
                "error": f"Folder not found: {folder_path}"
            }

        needle = search_text.lower()

        for root, directories, files in os.walk(
            folder_path
        ):

            # Ignore unnecessary directories
            directories[:] = [
                directory
                for directory in directories
                if directory not in IGNORE_DIRECTORIES
            ]

            for file_name in files:

                if not file_name.endswith(
                    SOURCE_EXTENSIONS
                ):
                    continue

                file_path = os.path.join(
                    root,
                    file_name
                )

                try:

                    with open(
                        file_path,
                        "r",
                        encoding="utf-8",
                        errors="ignore"
                    ) as file:

                        lines = file.readlines()

                except Exception:
                    continue

                for line_number, line in enumerate(
                    lines,
                    start=1
                ):

                    if needle not in line.lower():
                        continue

                    if len(matches) >= MAX_SEARCH_MATCHES:
                        truncated = True
                        break

                    text = line.strip()

                    if len(text) > MAX_LINE_CHARS:
                        text = text[:MAX_LINE_CHARS] + "..."

                    matches.append({
                        "file_path": file_path,
                        "line": line_number,
                        "text": text
                    })

                if truncated:
                    break

            if truncated:
                break

        if not matches:

            return {
                "success": True,
                "query": search_text,
                "matches": [],
                "message": "No matching lines found."
            }

        result = {
            "success": True,
            "query": search_text,
            "match_count": len(matches),
            "matches": matches
        }

        if truncated:

            result["truncated"] = True
            result["message"] = (
                "Showing the first matching lines only. "
                "Read those line numbers with read_file. "
                "Use a more specific search if these lines "
                "are not the code you need."
            )

        else:

            result["message"] = (
                "These matches already show the definition and "
                "every use. Call read_file once on this path for "
                "that function only: a few lines before it, and "
                "about 70 lines at most. Do not read the entire "
                "file. Then answer. Do not search again for this "
                "symbol, its expire calls, or the file name."
            )

        return result

    except Exception as e:

        return {
            "success": False,
            "error": f"Code search error: {str(e)}"
        }


# import os

# def search_code( search_text,
#     folder_path="."):
#     results = []

#     for root, directories, files in os.walk(folder_path):
#         for file_name in files:
#             if file_name.endswith((".py", ".js", ".html", ".jsx", ".ts", ".tsx", ".css")):
#                 file_path = os.path.join(root, file_name)
#                 try:
#                     with open(file_path, "r", encoding="utf-8") as file:
#                         content = file.read()
#                     if search_text.lower() in content.lower():
#                         results.append(file_path)
#                 except Exception:
#                     pass

#     if results:
#         return "\n".join(results)
#     return f"No files found containing: {search_text}"










