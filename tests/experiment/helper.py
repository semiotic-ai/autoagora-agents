# Copyright 2022-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

foo = "foo"


def add(a, *, b, c):
    return a * (b + c)


def sub(a, *, b, c):
    return a * (b - c)
