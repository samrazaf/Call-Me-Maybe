#!/usr/bin/env python3
# ************************************************************************* #
#                                                                           #
#                                                      :::      ::::::::    #
#  call_me.py                                        :+:      :+:    :+:    #
#                                                  +:+ +:+         +:+      #
#  By: samrazaf <samrazaf@student.42antananari   +#+  +:+       +#+         #
#                                              +#+#+#+#+#+   +#+            #
#  Created: 2026/09/10 10:41:27 by samrazaf        #+#    #+#               #
#  Updated: 2026/09/22 18:38:01 by samrazaf        ###   ########.fr        #
#                                                                           #
# ************************************************************************* #

import pathlib
from typing import Any
import sys
import json
import argparse



def parse_JSON() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="CallMeMaybe",
        description="Translate natural-language prompts into function calls.",
    )

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
        )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the prompts JSON file.",
        )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the output JSON file.",
        )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model identifier used by the provided llm_sdk.",
        )
    return parser.parse_args()

    #if args.functions_definition:
    #    try:
    #        print(args.functions_definition)
    #        with open(args.functions_definition, 'r', encoding="utf-8") as f:
    #            data = json.load(f)
    #            return data
    #    except json.JSONDecodeError as error:
    #        print(f"WARNING: file must be in 'JSON' format")
    #    except FileNotFoundError as error:
    #        print(f"WARNING: {error}")
    #    except PermissionError as error:
    #        print(f"WARNING: {error}")
#def parsing_definition(args) -> Any:
#    if arg:
#        try:
#            print(arg)
#            with open(arg, 'r', encoding="utf-8") as f:
#                data = json.load(f)
#            return (data)
#        except FileNotFoundError as error:
#            print(f"WARNING: {error}")
#        except json.JSONDecodeError as error:
#            print(f"WARNING: file must be in JSON format")
#        except PermissionError as error:
#            print(f"WARNING: {error}")
#    else:
#        return


def load_json(path: pathlib.Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: '{path}'")
    try:
        print(path)
        with open(path, 'r', encoding="utf-8") as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError:
        raise ValueError(f"WARNING: '{path}' must be in 'JSON' format")
    except PermissionError:
        raise PermissionError(f"WARNING: permission denied: '{path}'")
        #if args.functions_definition:





if __name__ == "__main__":
    #parsing_calling_tests()
    #data_call, data_def = check_arg()
    #for data in data_call:
    #    line = data.keys()
    #    prompt = data.values()
    #    print(line, prompt)
    ##args = check_arg()
    #print(args.functions_definition)
    #print(args.input)
    #print(args.output)
    try:
        data_la = parse_JSON()
        data_l = pathlib.Path(data_la.functions_definition)
        data = load_json(data_l)
        print(data)
    except BaseException as error:
        print(error)


