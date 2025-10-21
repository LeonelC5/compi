# 🚀 Generador de Parser LR(1) con Python

Este proyecto es una aplicación web que visualiza el proceso de construcción de un analizador sintáctico LR(1). Permite a los usuarios ingresar una gramática en formato Lark, generar la tabla de análisis y probarla con una cadena de entrada, todo desde una interfaz web amigable.

La aplicación utiliza una arquitectura cliente-servidor:
* **Backend:** Un servidor en **Python** con **Flask** que utiliza la librería **Lark** para realizar todos los cálculos del parser.
* **Frontend:** Una interfaz de usuario construida con **HTML, CSS y JavaScript** que se comunica con el backend.

## 📋 Prerrequisitos

Antes de empezar, asegúrate de tener instalado lo siguiente en tu sistema:
* **Python 3.8** o superior.
* **pip** (el gestor de paquetes de Python).
* Un navegador web moderno (Chrome, Firefox, Edge, etc.).

## ⚙️ Guía de Despliegue Local

Sigue estos pasos para ejecutar el proyecto en tu propia máquina.

### 1. Clonar el Repositorio

Abre tu terminal o línea de comandos y clona este repositorio en una carpeta de tu elección.

```bash
# Reemplaza la URL con la de tu propio repositorio
git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio
