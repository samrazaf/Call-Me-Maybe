#!/usr/bin/env python3
# ************************************************************************* #
#                                                                           #
#                                                      :::      ::::::::    #
#  call_me.py                                        :+:      :+:    :+:    #
#                                                  +:+ +:+         +:+      #
#  By: samrazaf <samrazaf@student.42antananari   +#+  +:+       +#+         #
#                                              +#+#+#+#+#+   +#+            #
#  Created: 2026/09/10 10:41:27 by samrazaf        #+#    #+#               #
#  Updated: 2026/09/16 17:53:29 by samrazaf        ###   ########.fr        #
#                                                                           #
# ************************************************************************* #

from typing import Any
import sys
import json
import argparse

def check_arg() -> Any:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json"
        )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json"
        )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json"
        )
    args = parser.parse_args()
    try:
        print(args.functions_definition)
        print(args.input)
        with open(args.input, 'r', encoding="utf-8") as f:
            data_call = json.load(f)
        with open(args.functions_definition, 'r', encoding="utf-8") as f:
            data_def = json.load(f)
            return data_call, data_def
    except json.JSONDecodeError as error:
        print(f"WARNING: file must be in 'JSON' format")
    except FileNotFoundError as error:
        print(f"WARNING: {error}")
    except PermissionError as error:
        print(f"WARNING: {error}")
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


if __name__ == "__main__":
    #parsing_calling_tests()
    data_call, data_def = check_arg()
    for data in data_call:
        line = data.keys()
        prompt = data.values()
        print(line, prompt)
    #args = check_arg()
    #print(args.functions_definition)
    #print(args.input)
    #print(args.output)


