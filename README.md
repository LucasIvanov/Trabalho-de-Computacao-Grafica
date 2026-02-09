# Modelador 3D - Trabalho Final de Computação Gráfica

**UNIOESTE - 2025**  
---

## Requisitos Implementados

✅ **1. Modelagem de cenas 3D com cubos**  
✅ **2. Projeção perspectiva**  
✅ **3. Transformações geométricas** (translação, rotação, escala uniforme)  
✅ **4. Sombreamento constante com Z-buffer**  
✅ **5. Sombreamento Phong com Z-buffer**  
✅ **6. Parâmetros editáveis em tempo real**  
✅ **7. Pipeline de Alvy Ray Smith completo**

---

## Instalação

### Dependências
- Python 3.10+
- pygame 2.5.2+
- numpy 1.24.0+

### Linux (Ubuntu/Debian) 

**Opção 1: Instalação Automática**
```bash
# Dar permissão e executar script
chmod +x instalar.sh
./instalar.sh

# Executar programa
./executar.sh
```

**Opção 2: Manual**
```bash
# 1. Instalar Python e venv (se necessário)
sudo apt install python3 python3-venv python3-pip

# 2. Criar ambiente virtual
python3 -m venv venv

# 3. Ativar ambiente
source venv/bin/activate

# 4. Instalar dependências
pip install pygame numpy

# 5. Executar
python main.py
```

### Windows

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar ambiente
venv\Scripts\activate

# 3. Instalar dependências
pip install pygame numpy

# 4. Executar
python main.py
```
---

## Controles

### Objetos
- **N** - Adicionar novo cubo
- **TAB** - Selecionar próximo objeto
- **DELETE** - Remover objeto selecionado

### Transformações (objeto selecionado)
- **W/S** - Mover Y (cima/baixo)
- **A/D** - Mover X (esquerda/direita)
- **Q/E** - Mover Z (frente/trás)
- **X** - Rotacionar no eixo X (15°)
- **Y** - Rotacionar no eixo Y (15°)
- **Z** - Rotacionar no eixo Z (15°)
- **+/-** - Escalar (uniforme)
- **1/2/3/4** - Mudar o posicionamento da luz
- **U/J** - Aumentar/diminuir difusão
- **I/K** - Aumentar/diminuir brilho

### Visualização
- **Mouse** (arrastar) - Rotacionar câmera
- **F1** - Alternar sombreamento (Constant ↔ Phong)
- **ESC** - Sair

---

## Pipeline de Alvy Ray Smith

O programa implementa todas as etapas do pipeline gráfico:

1. **Transformação de Modelo** - Aplica transformações aos vértices
2. **Transformação de Visão** - Converte para espaço da câmera
3. **Transformação de Projeção** - Projeção perspectiva
4. **Divisão Perspectiva** - Normalização (NDC)
5. **Recorte** - Back-face culling
6. **Transformação de Viewport** - Conversão para tela
7. **Rasterização** - Scanline com Z-buffer
8. **Sombreamento** - Constant ou Phong

### Z-buffer
Algoritmo de ocultação de superfícies que determina quais faces são visíveis. Utiliza interpolação baricêntrica para calcular profundidade de cada pixel.

### Sombreamento
- **Constant (Flat):** Iluminação calculada uma vez por face
- **Phong:** Modelo Blinn-Phong com componentes ambiente, difuso e especular

---

## Estrutura do Código

O arquivo `main.py` contém:

- **Transformações** - Matrizes 4x4 (translação, rotação, escala)
- **Cube** - Geometria e transformações do cubo
- **Camera** - Câmera virtual e projeção perspectiva
- **Light** - Fonte de luz pontual
- **Renderer** - Pipeline completo de renderização
- **Modeler3D** - Aplicação principal

---
