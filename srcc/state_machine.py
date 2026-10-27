from typing import List, Dict, Any
from srcc.models import FunctionDefinition

class DynamicJSONFSM:
    def __init__(self, function_definitions: List[FunctionDefinition]):
        # Indexation des objets FunctionDefinition pour une recherche O(1)
        self.functions = {f.fn_name: f for f in function_definitions}
        self.prefix_start = '{"function_name": "'
        self.prefix_args = '", "arguments": {'

        # Base de caractères pour les chaînes de caractères (str)
        self.string_chars = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
                             + "NOPQRSTUVWXYZ0123456789 _-.,!?@#$%^&"
                             + "*()[]{}+=|\\/<>")

    def get_allowed_characters(self, current_text: str) -> List[str]:
        text = current_text.lstrip()

        # Prefix fn
        if len(text) < len(self.prefix_start):
            if self.prefix_start.startswith(text):
                return [self.prefix_start[len(text)]]
            return []

        text_after_start = text[len(self.prefix_start):]

        # function name
        if '"' not in text_after_start:
            allowed = set()
            for fn_name in self.functions:
                if fn_name.startswith(text_after_start):
                    if len(fn_name) > len(text_after_start):
                        allowed.add(fn_name[len(text_after_start)])
                    else:
                        allowed.add('"')
            return list(allowed)

        fn_name = text_after_start.split('"')[0]
        if fn_name not in self.functions:
            return []

        # Transition Argument
        expected_prefix = self.prefix_start + fn_name + self.prefix_args
        if len(text) < len(expected_prefix):
            if expected_prefix.startswith(text):
                return [expected_prefix[len(text)]]
            return []

        # Recup des arguments
        args_text = text[len(expected_prefix):]
        func_def = self.functions[fn_name]

        # Utilisation des attributs du BaseModel Pydantic
        arg_names = func_def.args_names
        arg_types = {
            arg_name: self._normalise_type(arg_type)
            for arg_name, arg_type in func_def.args_types.items()
        }

        return self._trace_arguments(args_text, arg_names, arg_types)

    @staticmethod
    def _normalise_type(arg_type: str) -> str:
        """Convert schema types to the types understood by the FSM."""
        if arg_type == "number":
            return "float"
        if arg_type == "integer":
            return "int"
        if arg_type == "string":
            return "str"
        return arg_type

    def _trace_arguments(self, args_text: str,
                         arg_names: List[str],
                         arg_types: Dict[str, str]) -> List[str]:
        """Analyse le texte généré pour savoir exactement
        où l'on en est dans la séquence des arguments."""
        current_idx = 0

        for i, arg_name in enumerate(arg_names):
            # recup de la cle de l'argument
            key_str = f'"{arg_name}": '
            if current_idx + len(key_str) > len(args_text):
                remaining = key_str[len(args_text) - current_idx:]
                if key_str.startswith(args_text[current_idx:]):
                    return [remaining[0]]
                return []

            if args_text[current_idx:current_idx+len(key_str)] != key_str:
                return []
            current_idx += len(key_str)

            # trouver la valeur selon le type
            arg_type = arg_types[arg_name]
            is_last_arg = (i == len(arg_names) - 1)
            terminator = '}' if is_last_arg else ', '

            if arg_type in ("int", "float"):
                allowed_chars = set("0123456789")
                if arg_type == "float":
                    allowed_chars.add(".")
                allowed_chars.add("-")

                val_start = current_idx
                while (current_idx < len(args_text)
                       and args_text[current_idx] in allowed_chars):
                    current_idx += 1

                if current_idx < len(args_text):
                    remaining = args_text[current_idx:]
                    if terminator.startswith(remaining):
                        if len(remaining) < len(terminator):
                            return [terminator[len(remaining)]]
                        else:
                            current_idx += len(terminator)
                            continue
                    elif remaining.startswith(terminator):
                        current_idx += len(terminator)
                        continue
                    else:
                        return []
                else:
                    ans = list(allowed_chars)
                    if (current_idx > val_start
                            and args_text[val_start:current_idx]
                            not in ("-", ".")):
                        ans.append(terminator[0])
                    elif current_idx == val_start:
                        return list(set(ans))
                    return list(set(ans))

            elif arg_type == "str":
                if current_idx == len(args_text):
                    return ['"']
                if args_text[current_idx] != '"':
                    return []
                current_idx += 1

                string_closed = False
                in_escape = False
                while current_idx < len(args_text):
                    char = args_text[current_idx]
                    if in_escape:
                        in_escape = False
                    elif char == '\\':
                        in_escape = True
                    elif char == '"':
                        string_closed = True
                        current_idx += 1
                        break
                    current_idx += 1

                if not string_closed:
                    return list(self.string_chars) + ['"']

                if current_idx < len(args_text):
                    remaining = args_text[current_idx:]
                    if terminator.startswith(remaining):
                        if len(remaining) < len(terminator):
                            return [terminator[len(remaining)]]
                        else:
                            current_idx += len(terminator)
                            continue
                    elif remaining.startswith(terminator):
                        current_idx += len(terminator)
                        continue
                    else:
                        return []
                else:
                    return [terminator[0]]

            elif arg_type == "bool":
                remaining = args_text[current_idx:]
                matched_bool = None
                for b_val in ("true", "false"):
                    if remaining.startswith(b_val):
                        matched_bool = b_val
                        break

                if matched_bool:
                    current_idx += len(matched_bool)
                    remaining = args_text[current_idx:]
                    if current_idx < len(args_text):
                        if terminator.startswith(remaining):
                            if len(remaining) < len(terminator):
                                return [terminator[len(remaining)]]
                            else:
                                current_idx += len(terminator)
                                continue
                        elif remaining.startswith(terminator):
                            current_idx += len(terminator)
                            continue
                        else:
                            return []
                    else:
                        return [terminator[0]]
                else:
                    for b_val in ("true", "false"):
                        if b_val.startswith(remaining):
                            return [b_val[len(remaining)]]
                    return []

        # fermeture JSON
        if current_idx == len(args_text):
            return ['}']
        elif (current_idx + 1 == len(args_text)
              and args_text[current_idx] == '}'):
            return []

        return []
