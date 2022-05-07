#!/usr/bin/env python3
##############################################################################
##                                                                          ##
##          MATLAB Independent, Small & Safe, High Integrity Tools          ##
##                                                                          ##
##              Copyright (C) 2022, Chaoqing Wang                           ##
##                                                                          ##
##  This file is part of MISS_HIT.                                          ##
##                                                                          ##
##  MATLAB Independent, Small & Safe, High Integrity Tools (MISS_HIT) is    ##
##  free software: you can redistribute it and/or modify                    ##
##  it under the terms of the GNU Affero General Public License as          ##
##  published by the Free Software Foundation, either version 3 of the      ##
##  License, or (at your option) any later version.                         ##
##                                                                          ##
##  MISS_HIT is distributed in the hope that it will be useful,             ##
##  but WITHOUT ANY WARRANTY; without even the implied warranty of          ##
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           ##
##  GNU Afferto General Public License for more details.                    ##
##                                                                          ##
##  You should have received a copy of the GNU Affero General Public        ##
##  License along with MISS_HIT. If not, see                                ##
##  <http://www.gnu.org/licenses/>.                                         ##
##                                                                          ##
##############################################################################

import sys
from collections import OrderedDict
from io import StringIO
from pathlib import Path
from itertools import chain
from tempfile import NamedTemporaryFile
import keyword

from miss_hit_core import command_line, pathutil, work_package, cfg_tree
from miss_hit_core.errors import Error, Message_Handler
from miss_hit_core.m_ast import *
from miss_hit_core.m_lexer import MATLAB_Lexer
from miss_hit_core.m_parser import MATLAB_Parser

try:
    import isort
except ModuleNotFoundError:
    isort = None

try:
    import black
except ModuleNotFoundError:
    black = None


class Python_Visitor(AST_Visitor):
    """Matlab To Python/Numpy output: Python"""

    def __init__(self, fd, mh=None, **kwargs):
        super().__init__()
        self.fd = fd
        self.mh = mh
        self.node_src = OrderedDict()
        pass_visitor = getattr(self, "_pass_visitor")
        self.node_visitor = {
            Function_Signature: pass_visitor,
            Action: pass_visitor,
            Cell_Expression: getattr(self, "matrix_expression_visitor"),
        }

        self.options = {
            "matlab_alias": "mp",
            "generate_main": True,
            "inline_mode": False,
            "implicit_copy": True,
        }
        self.options.update(**kwargs)
        self.options["generate_main"] = (
            self.options["generate_main"] and not self.options["inline_mode"]
        )

        self.func_alias = lambda n: f"{self.options['matlab_alias']}.{n}"

    def __setitem__(self, node, src):
        self.node_src[node.uid] = src

    def pop(self, node):
        return self.node_src.pop(node.uid)

    @staticmethod
    def indent(src: str):
        return "\n".join([f"    {l}" for l in src.split("\n")])

    def visit(self, node, n_parent, relation):
        if isinstance(node, Reference):
            node.is_index = self.__is_index(node, n_parent, relation)

    def visit_end(self, node, n_parent, relation):
        visitor = self.node_visitor.get(node.__class__, None)
        try:
            if visitor is None:
                visitor = getattr(self, f"{node.__class__.__name__.lower()}_visitor")
                self.node_visitor[node.__class__] = visitor

            visitor(node, n_parent, relation)

        # except AttributeError:
        #     self[node] = ""

        except NotImplementedError:
            self[node] = ""

        # Top of Node Root
        if n_parent is None:
            # TODO: remove redundant bracket with python formatter
            #     `black -i -S {}.py` can only solve tiny part of it
            src = self.pop(node) + "\n"
            if self.fd:
                self.fd.write(src)
            else:
                print(src)

    def _pass_visitor(self, node: Node, n_parent, relation):
        pass

    def dynamic_selection_visitor(self, node: Dynamic_Selection, n_parent, relation):
        self[node] = f"{self.pop(node.n_prefix)}[{self.pop(node.n_field)}]"
        # ToDo: miss_hit_core lack support for setfield, e.g. s.(f) = v

    def selection_visitor(self, node: Selection, n_parent, relation):
        self[node] = f"{self.pop(node.n_prefix)}.{self.pop(node.n_field)}"

    def while_statement_visitor(self, node: While_Statement, n_parent, relation):
        self[node] = (
            f"while {self.pop(node.n_guard)}:\n"
            f"{self.indent(self.pop(node.n_body))}\n"
        )

    def general_for_statement_visitor(
        self, node: General_For_Statement, n_parent, relation
    ):
        self[node] = (
            f"for {self.pop(node.n_ident)} in {self.pop(node.n_expr)}:\n"
            f"{self.indent(self.pop(node.n_body))}\n"
        )

    def reshape_visitor(self, node: Reshape, n_parent, relation):
        self[node] = ":"

    def range_expression_visitor(self, node: Range_Expression, n_parent, relation):
        n_first = self.pop(node.n_first)
        n_last = self.pop(node.n_last)

        assert n_first is not None and n_last is not None
        (bra0, ket0), (bra1, ket1), (bra2, ket2) = map(
            self.__bracket, (node.n_first, node.n_stride, node.n_last)
        )

        inside_index = isinstance(n_parent, Reference) and getattr(n_parent, "is_index")
        bra, ket = ("", "") if inside_index else ("M[", "]")
        sep = ":"
        n_stride = (
            f"{sep}{bra1}{self.pop(node.n_stride)}{ket1}"
            if node.n_stride is not None
            else ""
        )
        self[node] = (
            f"{bra}"
            f"{bra0}{n_first}{ket0}"
            f"{n_stride}"
            f"{sep}{bra2}{n_last}{ket2}"
            f"{ket}"
        )

    def cell_reference_visitor(self, node: Cell_Reference, n_parent, relation):
        args = ", ".join(f"{self.pop(i)}" for i in node.l_args)
        self[node] = f"{self.pop(node.n_ident)}[I[{args}]]"

    def __node_contains_end(
        self, node: (Identifier, Reshape, Binary_Operation, Range_Expression)
    ):
        if isinstance(node, Identifier):
            return node.t_ident.value == "end"
        if isinstance(node, Reshape):
            return True
        if isinstance(node, Range_Expression):
            return (
                self.__node_contains_end(node.n_first)
                or self.__node_contains_end(node.n_last)
                or self.__node_contains_end(node.n_stride)
            )
        if isinstance(node, Binary_Operation):
            return self.__node_contains_end(node.n_lhs) or self.__node_contains_end(
                node.n_rhs
            )
        return False

    def __is_index(self, node: Reference, n_parent, relation):
        is_index = False

        if any(self.__node_contains_end(i) for i in node.l_args):
            is_index = True
        elif (
            isinstance(
                n_parent,
                (
                    Simple_Assignment_Statement,
                    Compound_Assignment_Statement,
                ),
            )
            and relation == "LHS"
        ):
            is_index = True

        return is_index

    def reference_visitor(self, node: Reference, n_parent, relation):
        # TODO: determine reference is function call or slice
        is_index = getattr(node, "is_index")
        bra, ket = ("[I[", "]]") if is_index else ("(", ")")

        args = ", ".join(self.pop(i) for i in node.l_args)
        self[node] = f"{self.pop(node.n_ident)}{bra}{args}{ket}"

    @staticmethod
    def _function_root(node: Identifier):
        f = node
        while f is not None and not isinstance(f, Function_Definition):
            f = f.n_parent
        return f

    def identifier_visitor(self, node: Identifier, n_parent, relation):
        value = node.t_ident.value

        # Python and Matlab keywords set are different. Luckily, Matlab forbids identifier start with `_`.
        prefix = (
            "_"
            if value
            in (
                *keyword.kwlist,  # python keyword can not be overwritten
                "I",
                "M",
                "C",  # mat2py keywords
            )
            else ""
        )

        if value == "nargout":
            f = self._function_root(node)
            if f:
                f.contains_nargout = True
        if value == "nargin":
            f = self._function_root(node)
            if f:
                f.contains_nargin = True
        self[node] = f"{prefix}{value}"

    def number_literal_visitor(self, node: Number_Literal, n_parent, relation):
        self[node] = node.t_value.value.replace("i", "j")

    def char_array_literal_visitor(self, node: Char_Array_Literal, n_parent, relation):
        self[node] = f"'{node.t_string.value}'"

    def string_literal_visitor(self, node: String_Literal, n_parent, relation):
        self[node] = f"'{node.t_string.value}'"

    def special_block_visitor(self, node: Special_Block, n_parent, relation):
        raise NotImplementedError

    def entity_constraints_visitor(self, node: Entity_Constraints, n_parent, relation):
        raise NotImplementedError

    def function_file_visitor(self, node: Function_File, n_parent, relation):
        header = "import mat2py as mp\n" "from mat2py.core import *\n"

        n_sig = node.l_functions[0].n_sig
        generate_main = self.options["generate_main"] and (
            len(n_sig.l_inputs) + len(n_sig.l_outputs) == 0
        )

        l_functions = [self.pop(l) for l in node.l_functions]
        if generate_main:
            l_functions = [*l_functions[1:], l_functions[0]]

        footer = (
            (
                f'if __name__ == "__main__":\n'
                f'{self.indent(f"{n_sig.n_name.t_ident.value}()")}'
            )
            if generate_main
            else ""
        )

        func = "\n".join(l_functions)
        self[node] = "\n".join(
            [header, func, footer][(1 if self.options["inline_mode"] else 0) :]
        ).lstrip("\n")

    def script_file_visitor(self, node: Script_File, n_parent, relation):
        header = "import mat2py as mp\n" "from mat2py.core import *\n"

        if self.options["generate_main"]:
            body = (
                f"def main():\n"
                f"{self.indent(self.pop(node.n_statements))}\n\n\n"
                f'if __name__ == "__main__":\n'
                f'{self.indent("main()")}'
            )
        else:
            body = self.pop(node.n_statements)

        func = "\n".join([self.pop(l) for l in node.l_functions])
        self[node] = "\n".join(
            [header, func, body][(1 if self.options["inline_mode"] else 0) :]
        ).lstrip("\n")

    def sequence_of_statements_visitor(
        self, node: Sequence_Of_Statements, n_parent, relation
    ):
        self[node] = (
            "\n".join([self.pop(l) for l in node.l_statements])
            if node.l_statements
            else "pass"
        )

    def function_pointer_visitor(self, node: Function_Pointer, n_parent, relation):
        self[node] = self.pop(node.n_name)

    def function_definition_visitor(
        self, node: Function_Definition, n_parent, relation
    ):
        contains_nargout = getattr(node, "contains_nargout", False)
        n_name = self.pop(node.n_sig.n_name)
        n_body = self.pop(node.n_body)
        l_inputs = ", ".join([self.pop(i) for i in node.n_sig.l_inputs])
        if contains_nargout:
            l_inputs += ", nargout=None"
        n_outputs = len(node.n_sig.l_outputs)
        l_outputs = ", ".join([self.pop(i) for i in node.n_sig.l_outputs])
        if n_outputs > 1 and contains_nargout:
            n_body = (
                f"{l_outputs} = (None,)*{n_outputs}\n\n"
                + n_body
                + "\nreturn{nargout_str}\n"
            )
        elif n_outputs > 0:
            n_body = n_body + "\nreturn{nargout_str}\n"
        nargout_str = f"({l_outputs})[:nargout]" if contains_nargout else l_outputs
        self[node] = f"def {n_name}({l_inputs}):\n{self.indent(n_body)}\n".format(
            nargout_str=f" {nargout_str}" if n_outputs > 0 else ""
        )

    def compound_assignment_statement_visitor(
        self, node: Compound_Assignment_Statement, n_parent, relation
    ):
        l_lhs = ", ".join([i if i != "~" else "_" for i in map(self.pop, node.l_lhs)])
        self[node] = f"{l_lhs} = {self.pop(node.n_rhs)}"

    def simple_assignment_statement_visitor(
        self, node: Simple_Assignment_Statement, n_parent, relation
    ):
        bra, ket = (
            ("copy(", ")")
            if self.options["implicit_copy"] and isinstance(node.n_rhs, Identifier)
            else ("", "")
        )
        self[node] = f"{self.pop(node.n_lhs)} = {bra}{self.pop(node.n_rhs)}{ket}"

    def function_call_visitor(self, node: Function_Call, n_parent, relation):
        args = ", ".join(self.pop(i) for i in node.l_args)
        self[node] = f"{self.pop(node.n_name)}({args})"

    @staticmethod
    def __bracket(node: Node):
        return (
            ("", "")
            if isinstance(
                node,
                (
                    Identifier,
                    Number_Literal,
                    String_Literal,
                    Char_Array_Literal,
                    Reference,
                    Cell_Reference,
                ),
            )
            else ("(", ")")
        )

    @staticmethod
    def __is_matrix(node: Node):
        if isinstance(node, (Matrix_Expression, Range_Expression)):
            return True
        elif isinstance(node, (Unary_Operation,)) and node.t_op.value in ("'", ".'"):
            return True
        else:
            return False

    @classmethod
    def __is_scalar(cls, node: Node):
        if isinstance(node, (Number_Literal,)):
            return True
        # elif isinstance(node, Identifier) and node.t_ident in ('pi', 'eps', ):
        #     return True
        elif isinstance(node, (Binary_Operation, Binary_Logical_Operation)):
            return cls.__is_scalar(node.n_lhs) and cls.__is_scalar(node.n_rhs)
        elif isinstance(node, (Unary_Operation,)):
            return cls.__is_scalar(node.n_expr)
        else:
            return False

    def switch_statement_visitor(self, node: Switch_Statement, n_parent, relation):
        l_actions = []
        n_expr_l = self.pop(node.n_expr)

        bra, ket = self.__bracket(node.n_expr)

        n_expr_l = f"{bra}{n_expr_l}{ket}"

        for i, a in enumerate(node.l_actions):
            key = "else" if a.n_expr is None else "elif" if i > 0 else "if"
            n_expr_r = "" if a.n_expr is None else self.pop(a.n_expr)
            bra, ket = self.__bracket(a.n_expr)
            n_expr = "" if a.n_expr is None else f" {n_expr_l} == {bra}{n_expr_r}{ket}"
            n_body = self.pop(a.n_body)
            n_body = self.indent("pass" if len(n_body) == 0 else n_body)
            l_actions.append(f"{key}{n_expr}:\n{n_body}")

        self[node] = "\n".join(l_actions)

    def if_statement_visitor(self, node: If_Statement, n_parent, relation):
        l_actions = []
        for i, a in enumerate(node.l_actions):
            key = "else" if a.n_expr is None else "elif" if i > 0 else "if"
            n_expr = "" if a.n_expr is None else " " + self.pop(a.n_expr)
            n_body = self.pop(a.n_body)
            n_body = self.indent("pass" if len(n_body) == 0 else n_body)
            l_actions.append(f"{key}{n_expr}:\n{n_body}")

        self[node] = "\n".join(l_actions)

    def continue_statement_visitor(self, node: Continue_Statement, n_parent, relation):
        self[node] = "continue"

    def break_statement_visitor(self, node: Break_Statement, n_parent, relation):
        self[node] = "break"

    def return_statement_visitor(self, node: Return_Statement, n_parent, relation):
        self[node] = "return{nargout_str}"

    def row_visitor(self, node: Row, n_parent, relation):
        if len(node.l_items) == 1:
            self[node] = self.pop(node.l_items[0])
        else:
            self[node] = f'[{", ".join(self.pop(i) for i in node.l_items)}]'

    def row_list_visitor(self, node: Row_List, n_parent, relation):
        if len(node.l_items) == 0:
            self[node] = f"[]"
        else:
            no_indent = len(node.l_items) == 1
            self[
                node
            ] = f'{", ".join(self.pop(i) for i in node.l_items)}{"" if no_indent else ", "}'

    def matrix_expression_visitor(
        self, node: (Matrix_Expression, Cell_Expression), n_parent, relation
    ):
        keyword = {Matrix_Expression: "M", Cell_Expression: "C"}[type(node)]
        self[node] = f"{keyword}[{self.pop(node.n_content)}]"
        # TODO: be careful with empty func_alias

    def unary_operation_visitor(self, node: Unary_Operation, n_parent, relation):
        t_op = node.t_op.value
        bra, ket = self.__bracket(node.n_expr)
        n_expr = f"{bra}{self.pop(node.n_expr)}{ket}"

        if t_op == ".'":
            self[node] = f"{n_expr}.T"
        elif t_op == "'":
            self[node] = f"{n_expr}.H"
        elif t_op == "~":
            if not n_expr.startswith("("):
                n_expr = f"({n_expr})"
            self[node] = f"_not{n_expr}"
        else:
            self[node] = f"{t_op}{n_expr}"

    def binary_logical_operation_visitor(
        self, node: Binary_Logical_Operation, n_parent, relation
    ):
        self.binary_operation_visitor(node, n_parent, relation)

    def binary_operation_visitor(self, node: Binary_Operation, n_parent, relation):
        t_op = node.t_op.value
        n_lhs = self.pop(node.n_lhs)
        n_rhs = self.pop(node.n_rhs)

        if t_op == "*" and (
            self.__is_scalar(node.n_rhs) or self.__is_scalar(node.n_lhs)
        ):
            t_op = ".*"

        if t_op == "*":
            t_op = "@"
            if not (self.__is_matrix(node.n_lhs) or self.__is_matrix(node.n_rhs)):
                if len(n_lhs) > len(n_rhs):
                    n_rhs = f"M[{n_rhs}]"
                else:
                    n_lhs = f"M[{n_lhs}]"

        if t_op in ("\\", "/", ".\\", "^") and not self.__is_scalar(node.n_rhs):
            func_name = {
                "\\": "mldivide",
                "/": "mrdivide",
                ".\\": "ldivide",
                "^": "mpower",
            }[t_op]
            self[node] = f"{func_name}({n_lhs}, {n_rhs})"
            return

        t_op = {
            "~=": "!=",
            "&&": "and",
            "||": "or",
            "./": "/",
            ".*": "*",
            ".^": "**",
            "^": "**",
        }.get(t_op, t_op)

        (bra0, ket0), (bra1, ket1) = map(self.__bracket, (node.n_lhs, node.n_rhs))

        self[node] = f"{bra0}{n_lhs}{ket0} {t_op} {bra1}{n_rhs}{ket1}"
        # TODO: replace operation to numpy format

    def import_statement_visitor(self, node: Import_Statement, n_parent, relation):
        raise NotImplementedError

    def metric_justification_pragma_visitor(
        self, node: Metric_Justification_Pragma, n_parent, relation
    ):
        raise NotImplementedError

    def naked_expression_statement_visitor(
        self, node: Naked_Expression_Statement, n_parent, relation
    ):
        self[node] = self.pop(node.n_expr)
        # TODO: determine a variable display or script call


class MH_Python_Result(work_package.Result):
    def __init__(self, wp, lines=None):
        super().__init__(wp, True)
        self.lines = lines


class MH_Python(command_line.MISS_HIT_Back_End):
    def __init__(self, options):
        super().__init__("MH Python")
        assert isinstance(options.matlab_alias, str) and re.search(
            r"^[A-Za-z_][A-Za-z0-9_]*", options.matlab_alias
        )
        self.matlab_alias = options.matlab_alias
        self.inline_mode = options.inline_mode
        self.python_alongside = options.python_alongside
        self.format_isort = options.format and isort is not None
        self.format_black = options.format and black is not None

    @classmethod
    def process_wp(cls, wp):
        # Create lexer
        lexer = MATLAB_Lexer(wp.mh, wp.get_content(), wp.filename, wp.blockname)
        if wp.cfg.octave:
            lexer.set_octave_mode()
        if len(lexer.text.strip()) == 0:
            return MH_Python_Result(wp)

        # Create parse tree
        try:
            parser = MATLAB_Parser(wp.mh, lexer, wp.cfg)
            n_cu = parser.parse_file()
        except Error:
            return MH_Python_Result(wp)

        with StringIO() as fd:
            try:
                n_cu.visit(
                    None,
                    Python_Visitor(
                        fd,
                        wp.mh,
                        matlab_alias=wp.options.matlab_alias,
                        inline_mode=wp.options.inline_mode,
                        **wp.extra_options,
                    ),
                    "Root",
                )
                return MH_Python_Result(wp, fd.getvalue())
            except Error:
                return MH_Python_Result(wp)

    def process_result(self, result: MH_Python_Result):
        if not isinstance(result, MH_Python_Result):
            return
        if result.lines is None:
            return

        lines = result.lines
        if self.format_isort:
            lines = isort.code(lines)
        if self.format_black:
            lines = black.format_str(lines, mode=black.Mode())
        if self.inline_mode:
            lines = lines.strip("\n")

        if self.python_alongside:
            with open(Path(result.wp.filename).with_suffix(".py"), "w") as fp:
                fp.write(lines)
        else:
            print(lines)

    def post_process(self):
        pass


def parse_args(argv=None):
    clp = command_line.create_basic_clp()

    # Extra language options
    clp["language_options"].add_argument(
        "--matlab-alias", default="mp", help="Matlab equivalent package name"
    )

    clp["language_options"].add_argument(
        "--inline-mode",
        default=False,
        action="store_true",
        help="Inline mode for no extra decorate python code",
    )

    # Extra output options
    clp["output_options"].add_argument(
        "--python-alongside",
        default=False,
        action="store_true",
        help="Create .py file alongside the .m file",
    )

    clp["output_options"].add_argument(
        "--format",
        default=False,
        action="store_true",
        help="Format the generated code with isort & black if installed",
    )

    # Extra debug options

    if argv is not None:
        _argv = sys.argv
        try:
            sys.argv = argv
            return command_line.parse_args(clp)
        finally:
            sys.argv = _argv
    else:
        return command_line.parse_args(clp)


def process_one_file(path: [Path, str], options=None, mh=None):
    if options is None:
        options = parse_args()

    if mh is None:
        mh = Message_Handler("debug")

        mh.show_context = not options.brief
        mh.show_style = False
        mh.show_checks = True
        mh.autofix = False

    cfg_tree.register_item(mh, path, options)
    wp = work_package.create(False, path, options.input_encoding, mh, options, {})
    backend = MH_Python(options)
    wp.register_file()
    backend.process_result(backend.process_wp(wp))

    for msg in chain.from_iterable(wp.mh.messages.get(wp.filename, {}).values()):
        if (
            msg.kind == "error"
            and "expected IDENTIFIER, reached EOF instead" == msg.message
        ):
            raise EOFError(msg.message)
        elif msg.kind.endswith("error"):
            mh.emit_message(msg)
            raise SyntaxError(msg.message)


def process_one_block(src: str, inline=True, format=False):
    with NamedTemporaryFile("w", suffix=".m") as f:
        f.write(src)
        f.flush()

        target_path = Path(f.name).with_suffix(".py")

        try:
            options = parse_args(
                ["mh_python", "--single", "--python-alongside", f.name]
            )
            if inline is True:
                options.inline_mode = True
            if format is True:
                options.format = True

            process_one_file(f.name, options)
            return target_path.read_text()
        finally:
            if target_path.exists():
                target_path.unlink()


def main_handler():
    options = parse_args()

    mh = Message_Handler("debug")

    mh.show_context = not options.brief
    mh.show_style = False
    mh.show_checks = True
    mh.autofix = False

    python_backend = MH_Python(options)
    command_line.execute(mh, options, {}, python_backend)


def main():
    command_line.ice_handler(main_handler)


if __name__ == "__main__":
    main()
