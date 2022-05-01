import unittest
import re
from pix2tex.dataset.demacro import pydemacro


def norm(s):
    s = re.sub(r'\n+', '\n', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def f(s):
    return norm(pydemacro(s))


class TestDemacroCases(unittest.TestCase):
    def test_noargs(self):
        inp = r'''
        \newcommand*{\noargs}{sample text}
        \noargs[a]\noargs{b}\noargs
        '''
        expected = r'''sample text[a]sample text{b}sample text'''
        self.assertEqual(f(inp), norm(expected))

    def test_optional_arg(self):
        inp = r'''
        \newcommand{\example}[2][YYY]{Mandatory arg: #2; Optional arg: #1.}     
        \example{BBB}
        \example[XXX]{AAA}
        '''
        expected = r'''
        Mandatory arg: BBB; Optional arg: YYY.
        Mandatory arg: AAA; Optional arg: XXX.
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_optional_arg_and_positional_args(self):
        inp = r'''
        \newcommand{\plusbinomial}[3][2]{(#2 + #3)^{#1}}
        \plusbinomial[4]{y}{x}
        '''
        expected = r'''(y + x)^{4}'''
        self.assertEqual(f(inp), norm(expected))

    def test_alt_definition1(self):
        inp = r'''
        \newcommand\d{replacement}
        \d
        '''
        expected = r'''replacement'''
        self.assertEqual(f(inp), norm(expected))

    def test_arg_with_bs_and_cb(self):
        # def 1 argument and with backslash (bs) and cruly brackets (cb) in definition
        inp = r'''
        \newcommand{\eq}[1]{\begin{equation}#1\end{equation}}
        \eq{\sqrt{2}\approx1.4}
        \eq[unexpected argument]{\sqrt{2}\approx1.4}
        '''
        expected = r'''
        \begin{equation}\sqrt{2}\approx1.4\end{equation}
        \begin{equation}\sqrt{2}\approx1.4\end{equation}
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition(self):
        inp = r'''
        \newcommand{\multiline}[2]{%
        Arg 1: \bf{#1}
        Arg 2: #2
        }
        \multiline{1}{two}
        '''
        expected = r'''
        Arg 1: \bf{1}
        Arg 2: two
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt1(self):
        inp = r'''
        \newcommand{\identity}[1]
        {#1}
        \identity{x}
        '''
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt2(self):
        inp = r'''
        \newcommand
        {\identity}[1]{#1}
        \identity{x}
        '''
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt3(self):
        inp = r'''
        \newcommand
        {\identity}[1]
        {#1}
        \identity{x}
        '''
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_multiline_definition_alt4(self):
        inp = r'''
        \newcommand
        {\identity}
        [1]
        {#1}
        \identity{x}
        '''
        expected = 'x'
        self.assertEqual(f(inp), norm(expected))

    def test_nested_definition(self):
        inp = r'''
        \newcommand{\cmd}[1]{command #1}
        \newcommand{\nested}[2]{\cmd{#1} \cmd{#2}}
        \nested{\alpha}{\beta}
        '''
        expected = r'''
        command \alpha command \beta
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_def(self):
        # check if \def is handled correctly.
        inp = r'''
        \def\defcheck#1#2{Defcheck arg1: #1 arg2: #2}
        \defcheck{1}{two}
        '''
        expected = r'''
        Defcheck arg1: 1 arg2: two
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt0(self):
        inp = r'''\def\be{\begin{equation}} \def\ee{\end{equation}} %some comment
        \be
        1+1=2
        \ee'''
        expected = r'''
        \begin{equation}
        1+1=2
        \end{equation}
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt1(self):
        inp = r'''\def\be{\begin{equation}}\def\ee{\end{equation}}
        \be
        1+1=2
        \ee'''
        expected = r'''
        \begin{equation}
        1+1=2
        \end{equation}
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt2(self):
        inp = r'''\def
        \be{\begin{equation}}
        \def\ee
        {\end{equation}}
        \be
        1+1=2
        \ee'''
        expected = r'''
        \begin{equation}
        1+1=2
        \end{equation}
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_multi_def_lines_alt3(self):
        inp = r'''
        \def\be
        {
            \begin{equation}
        }
        \def
        \ee
        {\end{equation}}
        \be
        1+1=2
        \ee'''
        expected = r'''
        \begin{equation}
        1+1=2
        \end{equation}
        '''
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt0(self):
        inp = r'''\let\a\alpha\let\b=\beta
        \a \b'''
        expected = r'''\alpha \beta'''
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt1(self):
        inp = r'''\let\a\alpha \let\b=\beta
        \a \b'''
        expected = r'''\alpha \beta'''
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt2(self):
        inp = r'''\let\a\alpha \let\b=\beta
        \a \b'''
        expected = r'''\alpha \beta'''
        self.assertEqual(f(inp), norm(expected))

    def test_let_alt3(self):
        inp = r'''
        \let
        \a
        \alpha
        \let\b=
        \beta
        \a \b'''
        expected = r'''\alpha \beta'''
        self.assertEqual(f(inp), norm(expected))


if __name__ == '__main__':
    unittest.main()
