from srcc.models import FunctionDefinition, TestPrompt, Argument, Returns
from typing import Dict, List, Any
import json
import os


class Parser:
    def __init__(self,
                 path_to_fn: str = "./data/input/functions_definition.json",
                 path_to_tests: str = ("./data/input/function"
                                       "_calling_tests.json")
                 ) -> None:
        self.path_to_fn: str = path_to_fn
        self.path_to_tests: str = path_to_tests

    @property
    def functions(self) -> List[FunctionDefinition]:
        fns: List[FunctionDefinition] = []

        # J'ai renommé la fonction pour plus de clarté
        def extract_args_and_returns(function: Dict[str, Any]) -> List[Argument]:
            arguments: List[Argument] = [] # Correction de la syntaxe ici
            # On utilise bien 'function' et non 'fn'
            
            try: 
                # J'ai renommé les variables pour éviter les conflits
                for arg_name, arg_data in function["parameters"].items():
                    arg: Argument = Argument(name=arg_name, type=arg_data["type"])
                    arguments.append(arg)
            except Exception as e:
                print(f"Erreur extraction arguments: {e}")
            return arguments


        try:
            with open(self.path_to_fn, "r") as f:
                fn_list: List[Dict[str, Any]] = json.load(f)
                
                for fn in fn_list:
                    # On passe bien 'fn' (le dictionnaire) et pas 'fn_list' !
                    param = extract_args_and_returns(fn)
                    
                    fn_def: FunctionDefinition = FunctionDefinition(
                        name=fn["name"],
                        parameters=param,
                        description=fn["description"],
                        returns=fn["returns"]["type"]
                    )
                    fns.append(fn_def)
        except Exception as e:
            print(f"Erreur chargement JSON: {e}")
            
        return fns

    @property
    def tests(self) -> List[TestPrompt]:
        tests: List[TestPrompt] = []
        try:
            with open(self.path_to_tests, "r") as f:
                test_list: List[Dict[str, str]] = []
                test_list = json.load(f)
                # print(test_list)
                for tst in test_list:
                    test: TestPrompt = TestPrompt(**tst)
                    tests.append(test)
        except Exception as e:
            print(e)
        return tests

if __name__ == "__main__":
    parse = Parser()
    for fn in parse.functions:
        i = 1
        print(fn.name)
        for args in fn.parameters:
            print(f"arg {i} {args.name}: {args.type} ")
            i += 1

