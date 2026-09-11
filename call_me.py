#!/usr/bin/env python3
# ************************************************************************* #
#                                                                           #
#                                                      :::      ::::::::    #
#  call_me.py                                        :+:      :+:    :+:    #
#                                                  +:+ +:+         +:+      #
#  By: samrazaf <samrazaf@student.42antananari   +#+  +:+       +#+         #
#                                              +#+#+#+#+#+   +#+            #
#  Created: 2026/09/10 10:41:27 by samrazaf        #+#    #+#               #
#  Updated: 2026/09/11 13:50:50 by samrazaf        ###   ########.fr        #
#                                                                           #
# ************************************************************************* #

from typing import Any
import sys
import json


def check_arg() -> str | None:
    if not sys.argv[1]:
        return
    return sys.argv[1]


def parsing_calling_tests(arg) -> Any:
    try:
        print(arg)
        with open(arg, 'r', encoding="utf-8") as f:
            data = json.load(f)
            return data
    except json.JSONDecodeError as error:
        print(f"WARNING: file must be in 'JSON' format")
    except FileNotFoundError as error:
        print(f"WARNING: {error}")
    except PermissionError as error:
        print(f"WARNING: {error}")


def parsing_definition() -> Any:
    try:
        with open('./data/input/functions_definition.json',
                  'r', encoding="utf-8") as f:
            data = json.load(f)
            return (data)
    except FileNotFoundError as error:
        print(f"WARNING: {error}")
    except json.JSONDecodeError as error:
        print(f"WARNING: file must be in JSON format")
    except PermissionError as error:
        print(f"WARNING: {error}")


if __name__ == "__main__":
    #parsing_calling_tests()
    data = parsing_calling_tests(check_arg())
    print(data)
