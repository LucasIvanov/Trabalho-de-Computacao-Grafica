#!/bin/bash
# Script de instalação automática - Modelador 3D
# Para Linux (Ubuntu/Debian)

echo "======================================="
echo "Instalação - Modelador 3D"
echo "======================================="

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python 3 não encontrado!"
    echo "Instale com: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "✓ Python encontrado: $(python3 --version)"

# Criar ambiente virtual
echo ""
echo "Criando ambiente virtual..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "ERRO ao criar ambiente virtual!"
    echo "Execute: sudo apt install python3-venv python3-full"
    exit 1
fi

echo "✓ Ambiente virtual criado"

# Ativar ambiente virtual
echo ""
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Instalar dependências
echo ""
echo "Instalando dependências (pygame e numpy)..."
pip install --upgrade pip
pip install pygame numpy

if [ $? -ne 0 ]; then
    echo "ERRO ao instalar dependências!"
    exit 1
fi

echo ""
echo "======================================="
echo "✓ Instalação concluída com sucesso!"
echo "======================================="
echo ""
echo "Para executar o programa:"
echo "  1. Ative o ambiente: source venv/bin/activate"
echo "  2. Execute: python main.py"
echo ""
echo "Ou use o script: ./executar.sh"
echo ""
