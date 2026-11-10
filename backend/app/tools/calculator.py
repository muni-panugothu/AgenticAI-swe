
import ast
import operator


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # Python 3.8 compatibility for older Numeric nodes if present
    if hasattr(ast, "Num") and isinstance(node, ast.Num):  # pragma: no cover
        return node.n
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression):
    """Evaluate a basic arithmetic expression safely (no eval of code)."""
    try:
        if expression is None or not str(expression).strip():
            return {
                "success": False,
                "error_code": "EMPTY_EXPRESSION",
                "message": "Expression is required.",
            }
        tree = ast.parse(str(expression).strip(), mode="eval")
        result = _eval_node(tree)
        return {
            "success": True,
            "expression": str(expression).strip(),
            "result": result,
        }
    except Exception as e:
        return {
            "success": False,
            "error_code": "CALCULATION_ERROR",
            "message": f"Could not evaluate expression: {e}",
        }
