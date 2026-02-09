#!/bin/bash
# Script de execução - Modelador 3D

# Verificar se ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "Ambiente virtual não encontrado!"
    echo "Execute primeiro: ./instalar.sh"
    exit 1
fi

# Ativar ambiente e executar
source venv/bin/activate
python main.py
