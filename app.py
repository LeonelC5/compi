import os
import re
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS 
# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# --- Configuración del Servidor Flask ---
app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, 'frontend'),
            template_folder=os.path.join(BASE_DIR, 'frontend'))
CORS(app)

# --- Implementación del Parser LR(1) ---

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
        # solo registramos lo que aparece en RHS; NO clasificamos aún
        for s in rhs:
            self._rhs_symbols.add(s)

    def finalize_symbols(self):
        # terminales = símbolos en RHS que no son no-terminales y no son ε
        self.terminals = set(sym for sym in self._rhs_symbols
                            if sym not in self.non_terminals and sym != 'ε')


class Item:
    def __init__(self, lhs, rhs, dot_pos, lookahead):
        self.lhs = lhs
        self.rhs = rhs
        self.dot_pos = dot_pos
        self.lookahead = lookahead
        
    def __str__(self):
        rhs_with_dot = self.rhs[:self.dot_pos] + ['•'] + self.rhs[self.dot_pos:]
        return f"[{self.lhs} -> {' '.join(rhs_with_dot)}, {self.lookahead}]"
    
    def __eq__(self, other):
        return (self.lhs == other.lhs and 
                self.rhs == other.rhs and 
                self.dot_pos == other.dot_pos and 
                self.lookahead == other.lookahead)
    
    def __hash__(self):
        return hash((self.lhs, tuple(self.rhs), self.dot_pos, self.lookahead))

class LR1Parser:
    def __init__(self, grammar):
        self.grammar = grammar
        self.first_sets = {}
        self.states = []
        self.goto_table = {}
        self.action_table = {}
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
        
        # Iterativo hasta que no haya cambios
        changed = True
        while changed:
            changed = False
            for lhs, rhs in self.grammar.productions:
                old_size = len(self.first_sets[lhs])
                
                if not rhs or rhs == ['ε']:  # Producción vacía
                    self.first_sets[lhs].add('ε')
                else:
                    for symbol in rhs:
                        self.first_sets[lhs].update(self.first_sets[symbol] - {'ε'})
                        if 'ε' not in self.first_sets[symbol]:
                            break
                    else:
                        # Todos los símbolos derivan ε
                        self.first_sets[lhs].add('ε')
                
                if len(self.first_sets[lhs]) != old_size:
                    changed = True
    
    def compute_first_table(self):
        """Tabla FIRST por no terminal (no por producción)."""
        self.first_table = []
        # ordenar los no terminales para que el orden sea estable
        for nt in sorted(self.grammar.non_terminals):
            first_of_nt = sorted(self.first_sets.get(nt, []))
            self.first_table.append({
                "nonterminal": nt,
                "first": first_of_nt
            })

    def first_of_string(self, symbols):
        """Calcula FIRST de una secuencia de símbolos"""
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
    
    def closure(self, items):
        """Calcula la clausura de un conjunto de ítems"""
        closure_set = set(items)
        
        changed = True
        while changed:
            changed = False
            new_items = set()
            
            for item in closure_set:
                if item.dot_pos < len(item.rhs):
                    next_symbol = item.rhs[item.dot_pos]
                    if next_symbol in self.grammar.non_terminals:
                        # Beta es lo que sigue después del símbolo
                        beta = item.rhs[item.dot_pos + 1:] + [item.lookahead]
                        first_beta = self.first_of_string(beta)
                        
                        for lhs, rhs in self.grammar.productions:
                            if lhs == next_symbol:
                                for lookahead in first_beta:
                                    if lookahead != 'ε':
                                        new_item = Item(lhs, rhs, 0, lookahead)
                                        if new_item not in closure_set:
                                            new_items.add(new_item)
                                            changed = True
            
            closure_set.update(new_items)
        
        return closure_set

    def goto(self, items, symbol):
        """Calcula GOTO(items, symbol)"""
        goto_items = set()
        
        for item in items:
            if (item.dot_pos < len(item.rhs) and 
                item.rhs[item.dot_pos] == symbol):
                new_item = Item(item.lhs, item.rhs, item.dot_pos + 1, item.lookahead)
                goto_items.add(new_item)
        
        return self.closure(goto_items)
    
    def build_parser(self):
        """Construye los estados y las tablas del parser LR(1)"""
        self.compute_first_sets()
        self.compute_first_table()
        
        # Determinar el símbolo de inicio aumentado
        augmented_symbols = [nt for nt in self.grammar.non_terminals if nt.endswith("'")]
        
        if augmented_symbols:
            # Ya hay símbolo aumentado en la gramática
            self.augmented_start = augmented_symbols[0]
            # Encontrar el símbolo base
            for lhs, rhs in self.grammar.productions:
                if lhs == self.augmented_start and len(rhs) == 1:
                    self.base_start = rhs[0]
                    break
            else:
                self.base_start = self.augmented_start.rstrip("'")
        else:
            # Crear símbolo aumentado
            self.base_start = self.grammar.start_symbol
            self.augmented_start = self.base_start + "'"
            # Agregar la producción aumentada
            self.grammar.productions.insert(0, (self.augmented_start, [self.base_start]))
            self.grammar.non_terminals.add(self.augmented_start)
        
        # Estado inicial
        initial_item = Item(self.augmented_start, [self.base_start], 0, '$')
        initial_state = self.closure({initial_item})
        
        self.states = [initial_state]
        unmarked = [0]
        
        # Construir todos los estados
        while unmarked:
            state_id = unmarked.pop(0)
            current_state = self.states[state_id]
            
            # Encontrar todos los símbolos que pueden seguir al punto
            symbols = set()
            for item in current_state:
                if item.dot_pos < len(item.rhs):
                    symbols.add(item.rhs[item.dot_pos])
            
            # Para cada símbolo, calcular GOTO
            for symbol in symbols:
                goto_state = self.goto(current_state, symbol)
                if goto_state:
                    # Buscar si este estado ya existe
                    existing_state_id = None
                    for i, state in enumerate(self.states):
                        if state == goto_state:
                            existing_state_id = i
                            break
                    
                    if existing_state_id is None:
                        # Nuevo estado
                        new_state_id = len(self.states)
                        self.states.append(goto_state)
                        unmarked.append(new_state_id)
                        self.goto_table[(state_id, symbol)] = new_state_id
                    else:
                        self.goto_table[(state_id, symbol)] = existing_state_id
        
        # Construir las tablas ACTION y GOTO
        self.build_action_table()
    
    def get_augmented_grammar(self):
        """Genera la gramática aumentada mostrando todas las posiciones del punto"""
        augmented_productions = []
        
        for lhs, rhs in self.grammar.productions:
            # Para cada producción, generar todas las posiciones del punto
            for dot_pos in range(len(rhs) + 1):
                rhs_with_dot = rhs[:dot_pos] + ['•'] + rhs[dot_pos:]
                production_str = f"{lhs} -> {' '.join(rhs_with_dot)}"
                augmented_productions.append({
                    "lhs": lhs,
                    "rhs": ' '.join(rhs_with_dot),
                    "production": production_str
                })
        
        return augmented_productions

    def build_action_table(self):
        """Construye la tabla ACTION"""
        for state_id, state in enumerate(self.states):
            self.action_table[state_id] = {}
            
            for item in state:
                if item.dot_pos < len(item.rhs):
                    # SHIFT
                    next_symbol = item.rhs[item.dot_pos]
                    if next_symbol in self.grammar.terminals:
                        if (state_id, next_symbol) in self.goto_table:
                            next_state = self.goto_table[(state_id, next_symbol)]
                            self.action_table[state_id][next_symbol] = ('shift', next_state)
                else:
                    # REDUCE o ACCEPT
                    if item.lhs == self.augmented_start and item.lookahead == '$':
                        self.action_table[state_id]['$'] = ('accept', None)
                    else:
                        # Encontrar el número de la producción
                        prod_num = None
                        for i, (lhs, rhs) in enumerate(self.grammar.productions):
                            if lhs == item.lhs and rhs == item.rhs:
                                prod_num = i
                                break
                        
                        if prod_num is not None:
                            self.action_table[state_id][item.lookahead] = ('reduce', prod_num)

    def to_dot(self):
        def esc(s: str) -> str:
            return s.replace('"', r'\"')

        lines = []
        lines.append('digraph LR1 {')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=box, style="rounded,filled", fillcolor="#ffffff", fontname="Inter"];')
        lines.append('  edge [fontname="Inter"];')

        # Nodos: cada estado con sus ítems
        for i, state in enumerate(self.states):
            items_txt = "\\n".join(esc(str(item)) for item in state)
            label = f'I{i}\\n{items_txt}'
            lines.append(f'  I{i} [label="{label}"];')

        # Aristas: desde goto_table
        for (sid, symbol), tid in self.goto_table.items():
            lines.append(f'  I{sid} -> I{tid} [label="{esc(symbol)}"];')

        lines.append('}')
        return "\n".join(lines)
    
    def parse(self, input_string):
        """Analiza una cadena usando el parser LR(1)"""
        # Tokenizar la entrada - manejar paréntesis correctamente
        tokens = []
        i = 0
        while i < len(input_string):
            char = input_string[i]
            if char.isspace():
                i += 1
                continue
            elif char in '()':
                tokens.append(char)
                i += 1
            else:
                # Recoger caracteres alfanuméricos
                token = ''
                while i < len(input_string) and not input_string[i].isspace() and input_string[i] not in '()':
                    token += input_string[i]
                    i += 1
                if token:
                    tokens.append(token)
        
        tokens.append('$')
        
        stack = [0]  # Pila de estados
        symbol_stack = []  # Pila de símbolos para mostrar en la tabla
        steps = []
        parse_tree_stack = []  # Para construir el árbol
        
        i = 0
        step_count = 0
        while i < len(tokens) and step_count < 100:  # Límite para evitar bucles infinitos
            step_count += 1
            state = stack[-1]
            token = tokens[i]
            
            if state not in self.action_table or token not in self.action_table[state]:
                steps.append({
                    "step": step_count,
                    "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[i:]),
                    "action": "ERROR"
                })
                return False, steps, None
            
            action, value = self.action_table[state][token]
            
            if action == 'shift':
                stack.append(value)
                symbol_stack.append(token)
                # Crear nodo hoja para el árbol
                parse_tree_stack.append({
                    "symbol": token,
                    "children": []
                })
                steps.append({
                    "step": step_count,
                    "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[i:]),
                    "action": f"s{value}"
                })
                i += 1
            elif action == 'reduce':
                lhs, rhs = self.grammar.productions[value]
                
                # Recoger hijos para el árbol
                children = []
                for _ in range(len(rhs)):
                    if symbol_stack:
                        symbol_stack.pop()
                    if stack:
                        stack.pop()
                    if parse_tree_stack:
                        children.insert(0, parse_tree_stack.pop())
                
                # Crear nodo padre
                parent_node = {
                    "symbol": lhs,
                    "children": children
                }
                parse_tree_stack.append(parent_node)
                symbol_stack.append(lhs)
                
                # GOTO
                current_state = stack[-1] if stack else 0
                if (current_state, lhs) in self.goto_table:
                    stack.append(self.goto_table[(current_state, lhs)])
                
                steps.append({
                    "step": step_count,
                    "stack": ' '.join(symbol_stack),
                    "input": ' '.join(tokens[i:]),
                    "action": f"r{value + 1}"  # +1 para que comience en r1, r2, etc.
                })
            elif action == 'accept':
                steps.append({
                    "step": step_count,
                    "stack": ' '.join(symbol_stack),
                    "input": '$',
                    "action": "acc"
                })
                
                # El árbol final está en la cima de parse_tree_stack
                tree = parse_tree_stack[-1] if parse_tree_stack else None
                return True, steps, tree
        
        return False, steps, None

def parse_grammar(grammar_text):
    """
    Parsea gramáticas que pueden contener el símbolo "|" para alternativas.
    Maneja tanto:
    - Alternativas en una línea: S -> A | B | C
    - Alternativas en líneas separadas:
      S -> A
        | B
        | C
    """
    grammar = Grammar()
    lines = grammar_text.strip().split('\n')
    
    current_lhs = None
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        
        # Verificar si la línea comienza con "|" (alternativa continua)
        if line.startswith('|'):
            if current_lhs is None:
                continue  # Ignorar "|" sin LHS previo
            
            # Quitar el "|" y procesar como RHS
            rhs_text = line[1:].strip()
            
            # Manejar producciones vacías
            if rhs_text in ('ε', '', 'epsilon'):
                rhs_symbols = ['ε']
            else:
                rhs_symbols = tokenize_rhs(rhs_text)
            
            grammar.add_production(current_lhs, rhs_symbols)
            continue
        
        # Verificar si tiene "->" o "→"
        if '->' in line or '→' in line:
            # Soporta "->" y "→"
            parts = re.split(r'\s*(?:->|→)\s*', line)
            if len(parts) != 2:
                continue

            lhs, rhs_full = parts[0].strip(), parts[1].strip()
            current_lhs = lhs  # Guardar para posibles alternativas siguientes
            
            # Dividir por "|" para manejar alternativas en la misma línea
            alternatives = [alt.strip() for alt in rhs_full.split('|')]
            
            for rhs in alternatives:
                rhs = rhs.replace('\xa0', ' ')  # normaliza espacios no estándar
                
                # Manejar producciones vacías
                if rhs in ('ε', '', 'epsilon'):
                    rhs_symbols = ['ε']
                else:
                    rhs_symbols = tokenize_rhs(rhs)
                
                grammar.add_production(lhs, rhs_symbols)

    grammar.finalize_symbols()
    return grammar

def tokenize_rhs(rhs_text):
    """Tokeniza el lado derecho de una producción"""
    rhs_symbols = []
    i = 0
    while i < len(rhs_text):
        if rhs_text[i].isspace():
            i += 1
            continue
        elif rhs_text[i] == "'":
            # Terminal entre comillas simples
            i += 1  # saltar primera comilla
            terminal = ''
            while i < len(rhs_text) and rhs_text[i] != "'":
                terminal += rhs_text[i]
                i += 1
            if i < len(rhs_text) and rhs_text[i] == "'":
                i += 1  # saltar segunda comilla
            rhs_symbols.append(terminal)
        elif rhs_text[i] in '()':
            rhs_symbols.append(rhs_text[i])
            i += 1
        else:
            # Recoger símbolo completo
            symbol = ''
            while i < len(rhs_text) and not rhs_text[i].isspace() and rhs_text[i] not in "()'":
                symbol += rhs_text[i]
                i += 1
            if symbol:
                rhs_symbols.append(symbol)
    
    return rhs_symbols

# --- Endpoints de la API ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    import os
    path = os.path.join(BASE_DIR, 'frontend', 'index.html')
    exists = os.path.exists(path)
    return f"Archivo index.html existe: {exists} en {path}"

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

@app.route('/parse', methods=['POST'])
def handle_parse_request():
    data = request.json
    grammar_text = data.get('grammar')
    input_string = data.get('input_string')

    if not grammar_text:
        return jsonify({"error": "La gramática no puede estar vacía."}), 400

    try:
        # Parsear la gramática
        grammar = parse_grammar(grammar_text)
        
        # Crear el parser LR(1)
        parser = LR1Parser(grammar)
        
        # Analizar la cadena
        parse_tree = None
        if input_string:
            accepted, parse_steps, parse_tree = parser.parse(input_string)
        else:
            accepted = True
            parse_steps = [{
                "step": 1,
                "stack": "N/A",
                "input": "Cadena vacía",
                "action": "No hay cadena para analizar"
            }]
        
        # Formatear los datos para el frontend
        states_data = []
        for i, state in enumerate(parser.states):
            state_info = {
                "id": i,
                "items": [str(item) for item in state]
            }
            states_data.append(state_info)
        
        # Convertir goto_table para JSON (tuplas a strings)
        goto_table_json = {}
        for (state_id, symbol), target_state in parser.goto_table.items():
            key = f"{state_id},{symbol}"
            goto_table_json[key] = target_state
        
        # Filtrar FIRST sets solo para no terminales
        first_sets_nonterminals = {k: list(v) for k, v in parser.first_sets.items() 
                                   if k in grammar.non_terminals}
        
        return jsonify({
            "accepted": accepted,
            "augmented_grammar": parser.get_augmented_grammar(),
            "first_sets": first_sets_nonterminals,
            "first_table": parser.first_table,
            "canonical_collection": states_data,
            "parsing_table_action": parser.action_table,
            "parsing_table_goto": goto_table_json,
            "parsing_steps": parse_steps,
            "parse_tree": parse_tree,
            "lr1_dot": parser.to_dot()   # AFD
        })

    except Exception as e:
        import traceback
        print("Error details:", traceback.format_exc())  # Para debug
        return jsonify({"error": f"Error al procesar la gramática: {str(e)}"}), 500
    
# --- Iniciar el servidor ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
