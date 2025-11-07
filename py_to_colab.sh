#!/bin/bash

# Converte arquivo .py em Notebook Jupyter (.ipynb) compatível com Colab
python3 << 'PYTHON'
import json

with open('/root/maria-helena-scripts/maria_helena_lstm_colab.py', 'r') as f:
    content = f.read()

# Dividir por células (comentários com "# CÉLULA X:")
cells = []
current_cell = ""

for line in content.split('\n'):
    if line.startswith('# CÉLULA'):
        if current_cell.strip():
            cells.append(current_cell.strip())
        current_cell = ""
    else:
        current_cell += line + "\n"

if current_cell.strip():
    cells.append(current_cell.strip())

# Criar notebook
notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell.split('\n')
        } for cell in cells
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('/root/maria-helena-scripts/maria_helena_lstm.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Notebook criado: maria_helena_lstm.ipynb")
PYTHON
