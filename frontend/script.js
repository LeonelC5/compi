document.getElementById('parseBtn').addEventListener('click', async () => {
  const grammar = document.getElementById('grammar').value.trim();
  const inputString = document.getElementById('inputString').value.trim();
  const parseBtn = document.getElementById('parseBtn');
  const btnText = parseBtn.querySelector('.btn-text');

  if (!grammar) {
    showError('Por favor, ingrese una gramática.');
    return;
  }

  // Mostrar estado de carga
  parseBtn.disabled = true;
  btnText.innerHTML = '<div class="loading"><div class="spinner"></div>Analizando...</div>';

  try {
    const response = await fetch('/parse', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grammar: grammar,
        input_string: inputString
      })
    });

    const data = await response.json();

    if (response.ok) {
      displayResults(data);
    } else {
      showError(data.error || 'Error desconocido en el servidor.');
    }
  } catch (error) {
    showError(`Error de conexión: ${error.message}`);
  } finally {
    // Restaurar botón
    parseBtn.disabled = false;
    btnText.innerHTML = 'Analizar Gramática';
  }
});

function showError(message) {
  const acceptanceEl = document.getElementById('acceptance');
  acceptanceEl.className = 'status error';
  acceptanceEl.innerHTML = `❌ Error: ${message}`;
  
  // Limpiar otros resultados
  document.getElementById('parsingSteps').innerHTML = '';
  document.getElementById('actionHeader').innerHTML = '';
  document.getElementById('actionBody').innerHTML = '';
  
  document.getElementById('result').style.display = 'block';
}

function displayResults(data) {
  // 1. Mostrar gramática aumentada
  displayAugmentedGrammar(data.augmented_grammar);
  expandSection('augmentedGrammar');

  // 2. Mostrar conjuntos FIRST
  displayFirstSets(data.first_sets);
  expandSection('firstSets');

  // 3. Mostrar autómata AFD LR(1)
  renderLR1Graph(data.lr1_dot);
  expandSection('lr1Graph');

  // 4. Mostrar tabla ACTION
  displayActionTable(data.parsing_table_action);
  expandSection('actionTableContent');

  // Mostrar resultados
  document.getElementById('result').style.display = 'block';
}
function displayAugmentedGrammar(augmentedGrammar) {
  const container = document.getElementById('augmentedGrammar');
  container.innerHTML = '';
  
  if (!augmentedGrammar || augmentedGrammar.length === 0) {
    container.innerHTML = '<p class="no-data">No hay gramática aumentada para mostrar.</p>';
    return;
  }

  // Crear contenedor de columnas
  const columnsContainer = document.createElement('div');
  columnsContainer.className = 'augmented-grammar-container';

  // Agrupar por símbolo no terminal
  const grouped = {};
  augmentedGrammar.forEach(production => {
    if (!grouped[production.lhs]) {
      grouped[production.lhs] = [];
    }
    grouped[production.lhs].push(production);
  });

  // Crear la visualización en columnas
  Object.entries(grouped).forEach(([lhs, productions]) => {
    const nonTerminalDiv = document.createElement('div');
    nonTerminalDiv.className = 'grammar-group';
    
    const header = document.createElement('h4');
    header.className = 'grammar-header';
    header.textContent = `${lhs}`;
    nonTerminalDiv.appendChild(header);
    
    const productionsList = document.createElement('div');
    productionsList.className = 'productions-list';
    
    productions.forEach(production => {
      const prodDiv = document.createElement('div');
      prodDiv.className = 'production-item';
      prodDiv.innerHTML = `<code>${production.rhs}</code>`;
      productionsList.appendChild(prodDiv);
    });
    
    nonTerminalDiv.appendChild(productionsList);
    columnsContainer.appendChild(nonTerminalDiv);
  });
  
  container.appendChild(columnsContainer);
}

function renderLR1Graph(dot) {
  const container = document.getElementById('lr1Graph');
  container.innerHTML = '';
  if (!dot) {
    container.innerHTML = '<p class="no-data">No hay grafo LR(1) para mostrar.</p>';
    return;
  }
  try {
    const viz = new Viz();
    viz.renderSVGElement(dot)
      .then(svg => {
        svg.style.maxWidth = '100%';
        container.appendChild(svg);
      })
      .catch(err => {
        container.innerHTML = `<pre class="error">${String(err)}</pre>`;
      });
  } catch (e) {
    container.innerHTML = `<pre class="error">${String(e)}</pre>`;
  }
}

function displayParsingSteps(steps) {
  const stepsList = document.getElementById('parsingSteps');
  stepsList.innerHTML = '';
  
  if (!steps || steps.length === 0) {
    stepsList.innerHTML = '<p class="no-data">No hay pasos de parsing para mostrar.</p>';
    return;
  }

  // Crear tabla con diseño moderno que replica exactamente la imagen
  const tableContainer = document.createElement('div');
  tableContainer.className = 'trace-table-container';
  
  const table = document.createElement('table');
  table.className = 'trace-table';
  
  // Header con el título "Trace" centrado
  const titleRow = document.createElement('tr');
  titleRow.innerHTML = `<th colspan="4" class="trace-title">Trace</th>`;
  table.appendChild(titleRow);
  
  // Sub-header con las columnas
  const headerRow = document.createElement('tr');
  headerRow.innerHTML = `
    <th class="step-col">Step</th>
    <th class="stack-col">Stack</th>
    <th class="input-col">Input</th>
    <th class="action-col">Action</th>
  `;
  table.appendChild(headerRow);
  
  // Filas de datos
  steps.forEach((step, index) => {
    const row = document.createElement('tr');
    const stepNumber = step.step || (index + 1);
    
    // Formatear la pila - mostrar estados con espacios
    let stackDisplay = step.stack;
    if (typeof stackDisplay === 'string') {
      // Reemplazar comas por espacios para mejor legibilidad
      stackDisplay = stackDisplay.replace(/,/g, ' ');
    }
    
    // Formatear la entrada
    let inputDisplay = step.input;
    if (inputDisplay && inputDisplay !== '$') {
      inputDisplay = inputDisplay.replace(/\s+/g, ' ').trim();
    }
    
    // Formatear la acción exactamente como en la imagen
    let actionDisplay = step.action;
    if (actionDisplay.includes('Shift')) {
      actionDisplay = actionDisplay.replace('Shift ', 's');
    } else if (actionDisplay.startsWith('S')) {
      actionDisplay = actionDisplay.toLowerCase(); // s3, s4, etc.
    } else if (actionDisplay.startsWith('R')) {
      actionDisplay = actionDisplay.toLowerCase(); // r1, r2, r3, etc.
    } else if (actionDisplay === 'ACC' || actionDisplay.includes('Accept')) {
      actionDisplay = 'acc';
    } else if (actionDisplay.includes('ERROR')) {
      actionDisplay = 'error';
    }
    
    row.innerHTML = `
      <td class="step-cell">${stepNumber}</td>
      <td class="stack-cell">${stackDisplay}</td>
      <td class="input-cell">${inputDisplay}</td>
      <td class="action-cell ${getActionClass(actionDisplay)}">${actionDisplay}</td>
    `;
    table.appendChild(row);
  });
  
  tableContainer.appendChild(table);
  stepsList.appendChild(tableContainer);
}

// Función auxiliar para obtener la clase CSS según el tipo de acción
function getActionClass(action) {
  if (action.startsWith('s')) return 'shift-action';
  if (action.startsWith('r')) return 'reduce-action';
  if (action === 'acc') return 'accept-action';
  if (action === 'error') return 'error-action';
  return '';
}


function displayActionTable(actionTable) {
  const actionHeader = document.getElementById('actionHeader'); // <tr>
  const actionBody = document.getElementById('actionBody');     // <tbody>
  
  actionHeader.innerHTML = '';
  actionBody.innerHTML = '';
  
  if (!actionTable || Object.keys(actionTable).length === 0) {
    actionBody.innerHTML = '<tr><td colspan="100%">No hay tabla ACTION para mostrar.</td></tr>';
    return;
  }

  // 1) Obtener símbolos y ordenarlos: "$" primero, luego alfabético
  const symbolsSet = new Set();
  Object.values(actionTable).forEach(row => {
    Object.keys(row).forEach(sym => symbolsSet.add(sym));
  });
  const symbols = Array.from(symbolsSet).sort((a, b) => {
    if (a === '$') return -1;
    if (b === '$') return 1;
    return a.localeCompare(b);
  });

  // 2) HEADER: usar directamente el <tr id="actionHeader">
  const thState = document.createElement('th');
  thState.textContent = 'Estado';
  actionHeader.appendChild(thState);

  symbols.forEach(symbol => {
    const th = document.createElement('th');
    th.textContent = symbol;
    actionHeader.appendChild(th);
  });

  // 3) BODY: filas en el mismo orden de símbolos
  Object.keys(actionTable).sort((a, b) => parseInt(a) - parseInt(b)).forEach(state => {
    const row = document.createElement('tr');

    const stateCell = document.createElement('td');
    stateCell.textContent = state;
    stateCell.className = 'state-cell';
    row.appendChild(stateCell);

    symbols.forEach(symbol => {
      const cell = document.createElement('td');
      const action = actionTable[state][symbol];      if (action) {
        if (Array.isArray(action)) {
          // Manejar tuplas como ('shift', 2) o ('reduce', 'R1')
          const actionType = action[0];
          const actionValue = action[1];
            if (actionType === 'shift') {
            cell.textContent = `S${actionValue}`;
            cell.className = 'shift-action';
          } else if (actionType === 'reduce') {
            cell.textContent = `R${actionValue}`;
            cell.className = 'reduce-action';
          } else if (actionType === 'accept') {
            cell.textContent = 'ACC';
            cell.className = 'accept-action';
          } else {
            cell.textContent = `${actionType} ${actionValue}`.trim();
          }
        } else {
          cell.textContent = action;
          
          // Clasificar por contenido del string
          if (cell.textContent.includes('shift') || cell.textContent.startsWith('S')) {
            cell.className = 'shift-action';
          } else if (cell.textContent.includes('reduce') || cell.textContent.startsWith('R')) {
            cell.className = 'reduce-action';
          } else if (cell.textContent.includes('accept') || cell.textContent === 'ACC') {
            cell.className = 'accept-action';
          }
        }
      }

      row.appendChild(cell);
    });

    actionBody.appendChild(row);
  });
}

function displayFirstSets(firstSets) {
  const firstSetsContainer = document.getElementById('firstSets');
  firstSetsContainer.innerHTML = '';
  
  if (!firstSets || Object.keys(firstSets).length === 0) {
    firstSetsContainer.innerHTML = '<p class="no-data">No hay conjuntos FIRST para mostrar.</p>';
    return;
  }

  const table = document.createElement('table');
  table.className = 'first-sets-table';
  
  // Header
  const headerRow = document.createElement('tr');
  headerRow.innerHTML = `
    <th>Símbolo</th>
    <th>FIRST</th>
  `;
  table.appendChild(headerRow);
  
  // Body
  Object.entries(firstSets).forEach(([symbol, firstSet]) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td><code>${symbol}</code></td>
      <td><code>{${firstSet.join(', ')}}</code></td>
    `;
    table.appendChild(row);
  });
  
  firstSetsContainer.appendChild(table);
}

function displayFirstTable(firstTable) {
  const container = document.getElementById('firstTable');
  if (!container) return;
  container.innerHTML = '';

  if (!firstTable || firstTable.length === 0) {
    container.innerHTML = '<p class="no-data">No hay tabla FIRST para mostrar.</p>';
    return;
  }

  const table = document.createElement('table');
  table.className = 'first-sets-table';

  const headerRow = document.createElement('tr');
  headerRow.innerHTML = `
    <th>Nonterminal</th>
    <th>FIRST</th>
  `;
  table.appendChild(headerRow);

  firstTable.forEach(row => {
    const tr = document.createElement('tr');
    const firstList = Array.isArray(row.first) ? row.first.join(', ') : '';
    tr.innerHTML = `
      <td><code>${row.nonterminal}</code></td>
      <td><code>{ ${firstList} }</code></td>
    `;
    table.appendChild(tr);
  });

  container.appendChild(table);
}

// Event listener para el botón de análisis de cadena
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const grammar = document.getElementById('grammar').value.trim();
  const inputString = document.getElementById('inputString').value.trim();
  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnText = analyzeBtn.querySelector('.btn-text');

  if (!grammar) {
    showError('Por favor, primero ingrese y analice una gramática.');
    return;
  }

  if (!inputString) {
    showError('Por favor, ingrese una cadena para analizar.');
    return;
  }

  // Mostrar estado de carga
  analyzeBtn.disabled = true;
  btnText.innerHTML = '<div class="loading"><div class="spinner"></div>Analizando...</div>';

  try {
    const response = await fetch('/parse', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grammar: grammar,
        input_string: inputString
      })
    });

    const data = await response.json();

    if (response.ok) {
      // Mostrar solo el resultado del análisis
      const acceptanceEl = document.getElementById('acceptance');
      const stepsTitle = document.getElementById('stepsTitle');
      
      if (data.accepted) {
        acceptanceEl.className = 'status success';
        acceptanceEl.innerHTML = '✅ Cadena aceptada por la gramática';
      } else {
        acceptanceEl.className = 'status error';
        acceptanceEl.innerHTML = '❌ Cadena rechazada por la gramática';
      }
      
      acceptanceEl.style.display = 'block';
      stepsTitle.style.display = 'block';      // Mostrar pasos del parsing
      displayParsingSteps(data.parsing_steps);
      
      // Mostrar árbol de derivación si existe
      if (data.parse_tree && data.accepted) {
        displayParseTree(data.parse_tree);
        expandSection('parseTree');
      }
    } else {
      showError(data.error || 'Error desconocido en el servidor.');
    }
  } catch (error) {
    showError(`Error de conexión: ${error.message}`);
  } finally {
    // Restaurar botón
    analyzeBtn.disabled = false;
    btnText.innerHTML = 'Analizar Cadena';
  }
});

function displayParseTree(tree) {
  const treeContainer = document.getElementById('parseTree');
  treeContainer.innerHTML = '';
  
  if (!tree) {
    treeContainer.innerHTML = '<p class="no-data">No hay árbol de derivación para mostrar.</p>';
    return;
  }

  // Crear contenedor principal del árbol
  const treeWrapper = document.createElement('div');
  treeWrapper.className = 'tree-wrapper';
  
  // Función recursiva para crear la estructura del árbol
  function buildNode(node) {
    const nodeContainer = document.createElement('div');
    nodeContainer.className = 'tree-node-container';
    
    // Crear el círculo del nodo
    const nodeCircle = document.createElement('div');
    nodeCircle.className = 'tree-node-circle';
    nodeCircle.textContent = node.symbol;
    nodeContainer.appendChild(nodeCircle);
    
    // Si tiene hijos, crear las ramas
    if (node.children && node.children.length > 0) {
      const childrenContainer = document.createElement('div');
      childrenContainer.className = 'tree-children-container';
      
      node.children.forEach((child, index) => {
        const childBranch = document.createElement('div');
        childBranch.className = 'tree-branch';
        
        // Línea vertical hacia abajo
        const verticalLine = document.createElement('div');
        verticalLine.className = 'tree-line-vertical';
        childBranch.appendChild(verticalLine);
        
        // Nodo hijo
        const childNode = buildNode(child);
        childBranch.appendChild(childNode);
        
        childrenContainer.appendChild(childBranch);
      });
      
      // Línea horizontal que conecta los hijos
      if (node.children.length > 1) {
        const horizontalLine = document.createElement('div');
        horizontalLine.className = 'tree-line-horizontal';
        nodeContainer.appendChild(horizontalLine);
      }
      
      nodeContainer.appendChild(childrenContainer);
    }
    
    return nodeContainer;
  }
  
  const treeRoot = buildNode(tree);
  treeWrapper.appendChild(treeRoot);
  treeContainer.appendChild(treeWrapper);
}

// Mejorar la experiencia del usuario con atajos de teclado
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === 'Enter') {
    document.getElementById('parseBtn').click();
  }
});

// Auto-resize de textareas
document.querySelectorAll('textarea').forEach(textarea => {
  textarea.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
  });
});

// Funcionalidad de acordeón
document.addEventListener('DOMContentLoaded', () => {
  // Inicializar acordeón
  initializeAccordion();
});

function initializeAccordion() {
  const accordionHeaders = document.querySelectorAll('.accordion-header');
  
  accordionHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const sectionId = header.getAttribute('data-section');
      const content = document.getElementById(sectionId);
      const icon = header.querySelector('.accordion-icon');
      const container = header.parentElement;
      
      if (content && icon && container) {
        toggleAccordionSection(header, content, icon, container);
      }
    });
  });
}

function toggleAccordionSection(header, content, icon, container) {
  const isCollapsed = content.classList.contains('collapsed');
  
  if (isCollapsed) {
    // Expandir
    content.classList.remove('collapsed');
    content.classList.add('expanding');
    header.classList.remove('collapsed');
    container.classList.remove('collapsed');
    
    // Remover clase de animación después de que termine
    setTimeout(() => {
      content.classList.remove('expanding');
    }, 400);
  } else {
    // Colapsar
    content.classList.add('collapsing');
    
    setTimeout(() => {
      content.classList.add('collapsed');
      content.classList.remove('collapsing');
      header.classList.add('collapsed');
      container.classList.add('collapsed');
    }, 400);
  }
}

// Función para expandir una sección específica (útil cuando se cargan resultados)
function expandSection(sectionId) {
  const header = document.querySelector(`[data-section="${sectionId}"]`);
  const content = document.getElementById(sectionId);
  const icon = header?.querySelector('.accordion-icon');
  const container = header?.parentElement;
  
  if (header && content && content.classList.contains('collapsed')) {
    toggleAccordionSection(header, content, icon, container);
  }
}

// Función para colapsar todas las secciones
function collapseAllSections() {
  const accordionHeaders = document.querySelectorAll('.accordion-header');
  
  accordionHeaders.forEach(header => {
    const sectionId = header.getAttribute('data-section');
    const content = document.getElementById(sectionId);
    const icon = header.querySelector('.accordion-icon');
    const container = header.parentElement;
    
    if (content && !content.classList.contains('collapsed')) {
      toggleAccordionSection(header, content, icon, container);
    }
  });
}

// Función para expandir todas las secciones
function expandAllSections() {
  const accordionHeaders = document.querySelectorAll('.accordion-header');
  
  accordionHeaders.forEach(header => {
    const sectionId = header.getAttribute('data-section');
    const content = document.getElementById(sectionId);
    const icon = header.querySelector('.accordion-icon');
    const container = header.parentElement;
    
    if (content && content.classList.contains('collapsed')) {
      toggleAccordionSection(header, content, icon, container);
    }
  });
}

// Event listeners para botones de control
document.addEventListener('DOMContentLoaded', () => {
  // Botón expandir todo
  const expandAllBtn = document.getElementById('expandAllBtn');
  if (expandAllBtn) {
    expandAllBtn.addEventListener('click', expandAllSections);
  }
  
  // Botón colapsar todo
  const collapseAllBtn = document.getElementById('collapseAllBtn');
  if (collapseAllBtn) {
    collapseAllBtn.addEventListener('click', collapseAllSections);
  }
});