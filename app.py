# streamlit_app.py
# Visualizador LR(1) en Streamlit – basado en tu implementación original

import re
import itertools
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional

import streamlit as st
import pandas as pd

st.set_page_config(page_title="LR(1) Parser Visualizer", layout="wide")

# ===============================
#  LÓGICA DE PARSER (tu backend)
# ===============================

class Grammar:
    def __init__(self):
        self.productions = []           # lista de (lhs, rhs_list)
        self.start_symbol = None
        self.terminals = set()
        self.non_terminals = set()
        self._rhs_symbols = set()       # todos los símbolos que aparecen en RHS

    def add_production(self, lhs, rhs):
        if self.start_symbol is None:
            self.start_symbol = lhs
        self.non_terminals.add(lhs)
        self.productions.append((lhs, rhs))
        for s in rhs:
            self._rhs_symbols.add(s)

    def finalize_symbols(self):
        # terminales = símbolos en RHS que no son no-terminales y no son ε
        self.terminals = set(
            sym for sym in self._rhs_symbols
            if sym not in self.non_terminals and sym != 'ε'
        )


@dataclass(frozen=True)
class Item:
    lhs: str
    rhs: Tuple[str, ...]
    dot_pos: int
    lookahead: str

    def __str__(self):
        rhs_list = list(self.rhs)
        rhs_with_dot = rhs_list[:self.dot_pos] + ['•'] + rhs_list[self.dot_pos:]
        return f"[{self.lhs} -> {' '.join(rhs_with_dot)}, {self.lookahead}]"


class LR1Parser:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.first_sets: Dict[str, Set[str]] = {}
        self.first_table = []
        self.states: List[Set[Item]] = []
        self.goto_table: Dict[Tuple[int, str], int] = {}
        self.action_table: Dict[int, Dict[str, Tuple[str, Optional[int]]]] = {}
        self.augmented_start = None
        self.base_start = None
        self.build_parser()

    def compute_first_sets(self):
        """Calcula los conjuntos FIRST para todos los símbolos"""
        # Inicializar FIRST sets
        for terminal in self.grammar.terminals:
            self.first_sets[terminal] = {terminal}
        for non_terminal in self.grammar.non_terminals:
            self.first_sets[non_terminal] = set()

        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.grammar.productions:
                old_size = len(self.first_sets[lhs])

                if not rhs or list(rhs) == ['ε']:
                    self.first_sets[lhs].add('ε')
                else:
                    for symbol in rhs:
                        sym_first = self.first_sets.get(symbol, {symbol})
                        self.first_sets[lhs].update(sym_first - {'ε'})
                        if 'ε' not in sym_first:
                            break
                    else:
                        self.first_sets[lhs].add('ε')

                if len(self.first_sets[lhs]) != old_size:
                    changed = True

    def compute_first_table(self):
        self.first_table = []
        for nt in sorted(self.grammar.non_terminals):
            first_of_nt = sorted(self.first_sets.get(nt, []))
            self.first_table.append({"nonterminal": nt, "first": first_of_nt})

    def first_of_string(self, symbols: List[str]) -> Set[str]:
        if not symbols:
            return {'ε'}
        result = set()
        for symbol in symbols:
            first_symbol = self.first_sets.get(symbol, {symbol})
            result.update(first_symbol - {'ε'})
            if 'ε' not in first_symbol:
                break
        else:
            result.add('ε')
        return result

    def closure(self, items: Set[Item]) -> Set[Item]:
        closure_set = set(items)
        changed = True
        while changed:
            changed = False
            new_items = set()
            for item in closure_set:
                if item.dot_pos < len(item.rhs):
                    next_symbol = item.rhs[item.dot_pos]
                    if next_symbol in self.grammar.non_terminals:
                        beta = list(item.rhs[item.dot_pos + 1:]) + [item.lookahead]
                        first_beta = self.first_of_string(beta)
                        for lhs, rhs in self.grammar.productions:
                            if lhs == next_symbol:
                                for la in first_beta:
                                    if la != 'ε':
                                        cand = Item(lhs, tuple(rhs), 0, la)
                                        if cand not in closure_set:
                                            new_items.add(cand)
                                            changed = True
            closure_set.update(new_items)
        return closure_set

    def goto(self, items: Set[Item], symbol: str) -> Set[Item]:
        goto_items = set()
        for item in items:
            if item.dot_pos < len(item.rhs) and item.rhs[item.dot_pos] == symbol:
                goto_items.add(Item(item.lhs, item.rhs, item.dot_pos + 1, item.lookahead))
        return self.closure(goto_items)

    def build_parser(self):
        self.compute_first_sets()
        self.compute_first_table()

        # Determinar símbolo de inicio aumentado
        augmented_symbols = [nt for nt in self.grammar.non_terminals if nt.endswith("'")]
        if augmented_symbols:
            self.augmented_start = augmented_symbols[0]
            for lhs, rhs in self.grammar.productions:
                if lhs == self.augmented_start and len(rhs) == 1:
                    self.base_start = rhs[0]
                    break
            else:
                self.base_start = self.augmented_start.rstrip("'")
        else:
            self.base_start = self.grammar.start_symbol
            self.augmented_start = self.base_start + "'"
            self.grammar.productions.insert(0, (self.augmented_start, [self.base_start]))
            self.grammar.non_terminals.add(self.augmented_start)

        # Estado inicial
        initial_item = Item(self.augmented_start, tuple([self.base_start]), 0, '$')
        initial_state = self.closure({initial_item})

        self.states = [initial_state]
        unmarked = [0]

        while unmarked:
            state_id = unmarked.pop(0)
            current_state = self.states[state_id]
            symbols = set()
            for it in current_state:
                if it.dot_pos < len(it.rhs):
                    symbols.add(it.rhs[it.dot_pos])

            for symbol in symbols:
                goto_state = self.goto(current_state, symbol)
                if goto_state:
                    existing = None
                    for i, st in enumerate(self.states):
                        if st == goto_state:
                            existing = i
                            break
                    if existing is None:
                        new_id = len(self.states)
                        self.states.append(goto_state)
                        unmarked.append(new_id)
                        self.goto_table[(state_id, symbol)] = new_id
                    else:
                        self.goto_table[(state_id, symbol)] = existing

        self.build_action_table()

    def get_augmented_grammar(self):
        out = []
        for lhs, rhs in self.grammar.productions:
            for dot_pos in range(len(rhs) + 1):
                rhs_with_dot = rhs[:dot_pos] + ['•'] + rhs[dot_pos:]
                out.append({
                    "lhs": lhs,
                    "rhs": ' '.join(rhs_with_dot),
                    "production": f"{lhs} -> {' '.join(rhs_with_dot)}"
                })
        return out

    def build_action_table(self):
        for sid, state in enumerate(self.states):
            self.action_table[sid] = {}
            for item in state:
                if item.dot_pos < len(item.rhs):
                    next_symbol = item.rhs[item.dot_pos]
                    if next_symbol in self.grammar.terminals:
                        if (sid, next_symbol) in self.goto_table:
                            ns = self.goto_table[(sid, next_symbol)]
                            self.action_table[sid][next_symbol] = ('shift', ns)
                else:
                    if item.lhs == self.augmented_start and item.lookahead == '$':
                        self.action_table[sid]['$'] = ('accept', None)
                    else:
                        prod_num = None
                        for i, (lhs, rhs) in enumerate(self.grammar.productions):
                            if lhs == item.lhs and tuple(rhs) == item.rhs:
                                prod_num = i
                                break
                        if prod_num is not None:
                            self.action_table[sid][item.lookahead] = ('reduce', prod_num)

    def to_dot(self) -> str:
        def esc(s: str) -> str:
            return s.replace('"', r'\"')

        lines = []
        lines.append('digraph LR1 {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style="rounded,filled", fillcolor="#ffffff", fontname="Inter"];')
        lines.append('  edge [fontname="Inter"];')
        for i, state in enumerate(self.states):
            items_txt = "\\n".join(esc(str(item)) for item in state)
            label = f'I{i}\\n{items_txt}'
            lines.append(f'  I{i} [label="{label}"];')
        for (sid, symbol), tid in self.goto_table.items():
            lines.append(f'  I{sid} -> I{tid} [label="{esc(symbol)}"];')
        lines.append('}')
        return "\n".join(lines)

    def parse(self, input_string: str):
        # Tokenización: respeta paréntesis
        tokens = []
        i = 0
        while i < len(input_string):
            ch = input_string[i]
            if ch.isspace():
                i += 1
                continue
            elif ch in '()':
                tokens.append(ch)
                i += 1
            else:
                tk = ''
                while i < len(input_string) and (not input_string[i].isspace()) and input_string[i] not in '()':
                    tk += input_string[i]
                    i += 1
                if tk:
                    tokens.append(tk)
        tokens.append('$')

        stack = [0]
        symbol_stack = []
        parse_tree_stack = []
        steps = []
        ptr = 0
        step_count = 0
        while ptr < len(tokens) and step_count < 200:
            step_count += 1
            state = stack[-1]
            tok = tokens[ptr]
            if state not in self.action_table or tok not in self.action_table[state]:
                steps.append({
                    "step": step_count, "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[ptr:]), "action": "ERROR"
                })
                return False, steps, None

            action, value = self.action_table[state][tok]
            if action == 'shift':
                stack.append(value)
                symbol_stack.append(tok)
                parse_tree_stack.append({"symbol": tok, "children": []})
                steps.append({
                    "step": step_count, "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[ptr:]), "action": f"s{value}"
                })
                ptr += 1

            elif action == 'reduce':
                lhs, rhs = self.grammar.productions[value]
                children = []
                for _ in range(len(rhs)):
                    if symbol_stack: symbol_stack.pop()
                    if stack: stack.pop()
                    if parse_tree_stack:
                        children.insert(0, parse_tree_stack.pop())
                parent = {"symbol": lhs, "children": children}
                parse_tree_stack.append(parent)
                symbol_stack.append(lhs)
                curr = stack[-1] if stack else 0
                if (curr, lhs) in self.goto_table:
                    stack.append(self.goto_table[(curr, lhs)])
                steps.append({
                    "step": step_count, "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[ptr:]), "action": f"r{value}"
                })

            elif action == 'accept':
                steps.append({
                    "step": step_count, "stack": ' '.join(symbol_stack),
                    "input": '$', "action": "acc"
                })
                tree = parse_tree_stack[-1] if parse_tree_stack else None
                return True, steps, tree

        return False, steps, None


def tokenize_rhs(rhs_text: str) -> List[str]:
    rhs_symbols = []
    i = 0
    while i < len(rhs_text):
        if rhs_text[i].isspace():
            i += 1
            continue
        elif rhs_text[i] == "'":
            i += 1
            terminal = ''
            while i < len(rhs_text) and rhs_text[i] != "'":
                terminal += rhs_text[i]; i += 1
            if i < len(rhs_text) and rhs_text[i] == "'":
                i += 1
            rhs_symbols.append(terminal)
        elif rhs_text[i] in '()':
            rhs_symbols.append(rhs_text[i]); i += 1
        else:
            symbol = ''
            while i < len(rhs_text) and (not rhs_text[i].isspace()) and rhs_text[i] not in "()'":
                symbol += rhs_text[i]; i += 1
            if symbol:
                rhs_symbols.append(symbol)
    return rhs_symbols


def parse_grammar(grammar_text: str) -> Grammar:
    g = Grammar()
    lines = grammar_text.strip().split('\n')
    current_lhs = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('|'):
            if current_lhs is None:
                continue
            rhs_text = line[1:].strip()
            rhs_symbols = ['ε'] if rhs_text in ('ε', '', 'epsilon') else tokenize_rhs(rhs_text)
            g.add_production(current_lhs, rhs_symbols)
            continue

        if '->' in line or '→' in line:
            parts = re.split(r'\s*(?:->|→)\s*', line)
            if len(parts) != 2:
                continue
            lhs, rhs_full = parts[0].strip(), parts[1].strip()
            current_lhs = lhs
            alts = [alt.strip() for alt in rhs_full.split('|')]
            for rhs in alts:
                rhs = rhs.replace('\xa0', ' ')
                rhs_symbols = ['ε'] if rhs in ('ε', '', 'epsilon') else tokenize_rhs(rhs)
                g.add_production(lhs, rhs_symbols)
    g.finalize_symbols()
    return g

# ===============================
#    HELPERS DE PRESENTACIÓN
# ===============================

def action_table_to_dataframe(action_table: Dict[int, Dict[str, Tuple[str, Optional[int]]]]) -> pd.DataFrame:
    symbols = set()
    for row in action_table.values():
        symbols.update(row.keys())
    symbols = sorted(symbols, key=lambda s: (s != '$', s))

    data = []
    for sid in sorted(action_table.keys()):
        row = {'Estado': sid}
        for s in symbols:
            act = action_table[sid].get(s)
            if not act:
                row[s] = ''
            else:
                typ, val = act
                if typ == 'shift':
                    row[s] = f"S{val}"
                elif typ == 'reduce':
                    row[s] = f"R{val}"
                elif typ == 'accept':
                    row[s] = 'ACC'
                else:
                    row[s] = f"{typ} {val if val is not None else ''}".strip()
        data.append(row)
    df = pd.DataFrame(data)
    return df[['Estado'] + symbols]

def steps_to_dataframe(steps: List[Dict]) -> pd.DataFrame:
    rows = []
    for s in steps:
        rows.append({
            "Step": s.get("step"),
            "Stack": s.get("stack"),
            "Input": s.get("input"),
            "Action": s.get("action"),
        })
    return pd.DataFrame(rows)

def parse_tree_to_dot(tree: Dict) -> str:
    # Construye un árbol en DOT a partir del dict {"symbol": str, "children":[...] }
    lines = ["digraph T {", '  node [shape=circle, style="filled", fillcolor="#eef2ff", fontname="Inter"];']
    counter = itertools.count()
    def add(node):
        my_id = f"n{next(counter)}"
        label = node.get("symbol", "")
        lines.append(f'  {my_id} [label="{label}"];')
        for ch in node.get("children", []):
            cid = add(ch)
            lines.append(f"  {my_id} -> {cid};")
        return my_id
    add(tree)
    lines.append("}")
    return "\n".join(lines)

# ===============================
#              UI
# ===============================

st.title("LR(1) Parser Visualizer (Streamlit)")

with st.expander("📋 Instrucciones rápidas", expanded=False):
    st.markdown("""
1) Escribe tu **gramática** (puedes usar `|` para alternativas y `ε` para epsilon).  
2) Haz clic en **Construir parser** para ver FIRST, AFD y tabla ACTION.  
3) Escribe una **cadena de entrada** y pulsa **Analizar cadena** para ver los pasos y el árbol.
    """)

default_grammar = """A -> ( A ) | a"""

col_g, col_i = st.columns([2,1])
with col_g:
    grammar_text = st.text_area(
        "Gramática (formato tradicional)",
        value=default_grammar,
        height=200,
        help="Ejemplo: A -> ( A ) | a"
    )
with col_i:
    input_string = st.text_area(
        "Cadena de entrada",
        value="(a)",
        height=200,
        help="Ejemplos: a, (a), ((a))"
    )

c1, c2, c3 = st.columns([1,1,2])
build_clicked = c1.button("🔧 Construir parser")
analyze_clicked = c2.button("▶️ Analizar cadena")

if build_clicked or analyze_clicked:
    try:
        grammar = parse_grammar(grammar_text)
        parser = LR1Parser(grammar)

        # --- Sección 1: Gramática aumentada
        st.subheader("1) Gramática aumentada")
        ag = parser.get_augmented_grammar()
        df_ag = pd.DataFrame(ag)[["lhs", "rhs"]].rename(columns={"lhs":"LHS","rhs":"RHS (con •)"})
        st.dataframe(df_ag, use_container_width=True)

        # --- Sección 2: Conjuntos FIRST (solo para no terminales)
        st.subheader("2) Conjuntos FIRST")
        first_rows = []
        for nt in sorted(grammar.non_terminals):
            first_rows.append({"No terminal": nt, "FIRST": ", ".join(sorted(parser.first_sets.get(nt, [])))})
        st.dataframe(pd.DataFrame(first_rows), use_container_width=True)

        # --- Sección 3: AFD LR(1)
        st.subheader("3) Autómata LR(1)")
        dot_lr1 = parser.to_dot()
        st.graphviz_chart(dot_lr1, use_container_width=True)
        st.download_button("⬇️ Descargar AFD (.dot)", dot_lr1, file_name="lr1_automata.dot", mime="text/vnd.graphviz")

        # --- Sección 4: Tabla ACTION
        st.subheader("4) Tabla ACTION")
        df_action = action_table_to_dataframe(parser.action_table)
        st.dataframe(df_action, use_container_width=True)

        # Mostrar estados con sus ítems
        with st.expander("📦 Colección canónica de estados (ítems LR(1))", expanded=False):
            for i, state in enumerate(parser.states):
                st.markdown(f"**Estado I{i}**")
                for it in sorted(state, key=lambda x: (x.lhs, x.dot_pos, x.lookahead, x.rhs)):
                    st.code(str(it), language="text")

        # --- Sección 5: Análisis de cadena
        if analyze_clicked:
            st.subheader("5) Trazado del análisis (shift/reduce)")
            accepted, steps, tree = parser.parse(input_string)

            status = "✅ Cadena aceptada" if accepted else "❌ Cadena rechazada"
            st.markdown(f"**Estado:** {status}")

            df_steps = steps_to_dataframe(steps)
            st.dataframe(df_steps, use_container_width=True)

            # Árbol de derivación (si aceptada)
            st.subheader("6) Árbol de derivación")
            if accepted and tree:
                dot_tree = parse_tree_to_dot(tree)
                st.graphviz_chart(dot_tree, use_container_width=True)
                st.download_button("⬇️ Descargar Árbol (.dot)", dot_tree, file_name="parse_tree.dot", mime="text/vnd.graphviz")
            else:
                st.info("No hay árbol para mostrar.")

    except Exception as e:
        st.error(f"Error al procesar: {e}")
