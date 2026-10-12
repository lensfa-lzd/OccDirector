from json_repair import repair_json, loads


def check_parsable(input: str):
    try:
        return eval(str(loads(repair_json(input))))
    except Exception as e:
        print(e)
        return "The output can not read as a dictionary in Python.\n"


def remove_thinking_content(input: str):
    return input.split('</think>', maxsplit=1)[-1]
