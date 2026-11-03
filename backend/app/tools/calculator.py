

def calculator(expression):
    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Error:{e}"

    