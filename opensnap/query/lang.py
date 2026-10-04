"""OVQL: a small open-vocabulary query language over 3D object relations.

SELECT [WHERE <expr>] [LIMIT <k>] [THRESHOLD <p>]
expr   := term (OR term)*
term   := factor (AND factor)*
factor := NOT factor | '(' expr ')' | pred
pred   := IS('<text>') | CONTAINS(x, y, z) | NEAR(<expr>, <meters>) | ABOVE(<expr>) | BELOW(<expr>) | ON(<expr>)
"""
import re
from dataclasses import dataclass

TOKEN = re.compile(r"\s*(?:(?P<num>-?\d+(?:\.\d+)?)|(?P<str>'[^']*'|\"[^\"]*\")|(?P<id>[A-Za-z_]+)|(?P<sym>[(),*]))")


@dataclass
class Node:
    op: str
    args: tuple = ()


@dataclass
class Query:
    where: Node | None
    limit: int | None = None
    threshold: float = 0.0


def tokenize(s):
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"bad token at {pos}: {s[pos:pos + 10]!r}")
        kind = m.lastgroup
        val = m.group(kind)
        out.append((kind, val[1:-1] if kind == "str" else (val.upper() if kind == "id" else val)))
        pos = m.end()
    return out


class Parser:
    def __init__(self, text):
        self.t, self.i = tokenize(text), 0

    def peek(self, val=None):
        if self.i >= len(self.t):
            return None
        k, v = self.t[self.i]
        return v if val is None else (v == val)

    def eat(self, val=None):
        k, v = self.t[self.i]
        if val is not None and v != val:
            raise SyntaxError(f"expected {val}, got {v}")
        self.i += 1
        return k, v

    def query(self):
        self.eat("SELECT")
        if self.peek("*"):
            self.eat("*")
        where, limit, thr = None, None, 0.0
        while self.peek() is not None:
            kw = self.eat()[1]
            if kw == "WHERE":
                where = self.expr()
            elif kw == "LIMIT":
                limit = int(self.eat()[1])
            elif kw == "THRESHOLD":
                thr = float(self.eat()[1])
            else:
                raise SyntaxError(f"unexpected {kw}")
        return Query(where, limit, thr)

    def expr(self):
        node = self.term()
        while self.peek("OR"):
            self.eat("OR")
            node = Node("OR", (node, self.term()))
        return node

    def term(self):
        node = self.factor()
        while self.peek("AND"):
            self.eat("AND")
            node = Node("AND", (node, self.factor()))
        return node

    def factor(self):
        if self.peek("NOT"):
            self.eat("NOT")
            return Node("NOT", (self.factor(),))
        if self.peek("("):
            self.eat("(")
            node = self.expr()
            self.eat(")")
            return node
        name = self.eat()[1]
        self.eat("(")
        if name == "IS":
            text = self.eat()[1]
            self.eat(")")
            return Node("IS", (text,))
        if name == "CONTAINS":
            xyz = [float(self.eat()[1])]
            for _ in range(2):
                self.eat(",")
                xyz.append(float(self.eat()[1]))
            self.eat(")")
            return Node("CONTAINS", tuple(xyz))
        if name in ("NEAR", "ABOVE", "BELOW", "ON"):
            sub = self.expr()
            dist = 0.0
            if self.peek(","):
                self.eat(",")
                dist = float(self.eat()[1])
            self.eat(")")
            return Node(name, (sub, dist))
        raise SyntaxError(f"unknown predicate {name}")


def parse(text):
    return Parser(text).query()
