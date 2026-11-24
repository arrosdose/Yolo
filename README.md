# Monitoramento de acesso a áreas restritas

Protótipo de alarme com câmera e visão computacional. O sistema detecta pessoas com **YOLO**, verifica com **OpenCV** se elas vestem o azul do uniforme e, se alguém ficar vários quadros sem uniforme, dispara um **alerta visual e sonoro**.

Trabalho de Visão Computacional (Etapa 1), Curso de Ciência da Computação, Unochapecó, Chapecó/SC, 2026.

**Integrantes:** Camile Favero Piassa, Guilherme Scherer dos Santos, Gustavo Arthur Tedesco e Ígor Zanatta Palliga.

---

## 1. Problema

O problema ocorre no **estoque de um provedor de serviços de internet**, onde ficam equipamentos de rede e materiais de alto valor. O acesso deve ser restrito aos funcionários autorizados.

O local tem um alarme com **sensores de movimento**, mas foram relatadas **falhas de detecção**: em algumas situações o alarme não dispara, mesmo ativado, quando alguém entra. O ambiente é usado várias vezes ao dia por cerca de **30 colaboradores** de setores diferentes.

O levantamento foi feito a partir do relato de um integrante do grupo que trabalha na empresa.

## 2. Solução

Um sistema de monitoramento por câmera que analisa cada pessoa que entra e verifica uma característica visual do uniforme dos funcionários. Para simplificar o protótipo, essa característica é a **presença de tons de azul na roupa**. Quando a pessoa não está de azul por tempo suficiente, o sistema emite um alerta de possível entrada não autorizada.

### Como funciona (a cada quadro do vídeo)

```
Câmera/vídeo → YOLO detecta pessoas → recorte do tronco → % de azul (HSV)
            → contador de quadros suspeitos → alerta (visual + sonoro)
```

1. **Detecção de pessoas** (`detecta_pessoas`): o YOLOv4-tiny encontra os objetos da imagem e o código mantém só a classe 0 do COCO, "pessoa".
2. **Recorte do tronco** (`tem_uniforme`): de cada caixa, usa só a região de 15% a 55% da altura e de 20% a 80% da largura, para pegar a camisa e evitar cabeça, pernas e fundo.
3. **Medição do azul** (`mascara_azul`): converte o recorte para HSV e aplica `cv2.inRange` com três faixas de azul (escuro, médio e claro). Uma abertura morfológica remove ruído. Se **12% ou mais** do tronco for azul, a pessoa é considerada funcionária (caixa verde); senão, "SEM UNIFORME" (caixa vermelha).
4. **Decisão** (`Monitor.processa`): um contador soma 1 a cada quadro com alguém sem uniforme e volta a zero quando o quadro está normal. Ao chegar a **25 quadros seguidos**, o alarme liga. Assim, um quadro borrado ou com sombra não gera alarme falso.
5. **Alerta:** borda e faixa vermelhas com "ALERTA: POSSIVEL ENTRADA NAO AUTORIZADA" no vídeo e uma sirene: no Colab, gravada no vídeo nos instantes de alarme; no computador, um bipe em tempo real.

## 3. Tecnologias e modelos

| Item | Uso |
|---|---|
| **Python 3** | linguagem do projeto |
| **OpenCV 4.x** (`cv2`) | leitura do modelo (`cv2.dnn`), conversão para HSV, máscaras, desenho, leitura e gravação de vídeo |
| **YOLOv4-tiny** (Darknet, pré-treinado no COCO, 80 classes) | detecção de pessoas, sem treinamento adicional |
| **NumPy** | operações com as imagens e geração do áudio da sirene |
| **Matplotlib** | exibição das imagens de teste no notebook |
| **FFmpeg** | junta o vídeo anotado com o áudio da sirene (já vem no Colab) |
| **Google Colab** | ambiente de execução do notebook |

> **Por que OpenCV 4?** O OpenCV 5 removeu a leitura de modelos Darknet (`readNetFromDarknet`). Por isso o projeto fixa `opencv < 4.12`.

O modelo YOLOv4-tiny vem do repositório [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet): `yolov4-tiny.cfg` (arquitetura) e `yolov4-tiny.weights` (pesos, cerca de 23 MB).

## 4. Instalação e execução

### Opção A: Google Colab (recomendada)

1. Abra o [Google Colab](https://colab.research.google.com) e vá em **Arquivo → Fazer upload de notebook**. Escolha `notebooks/Alarme_estoque_YOLO.ipynb`.
2. Rode a **Seção 0** (instala o OpenCV 4). Se aparecer **"Reiniciar sessão"**, clique nele e **não** rode essa célula de novo.
3. Rode a **Seção 1** (baixa o modelo) e a **Seção 2** (lógica do alarme).
4. **Seção 3, teste em imagem:**
   - a primeira célula baixa `person.jpg` (pessoa de azul, resultado esperado: sem alarme);
   - a segunda usa `pessoa2.jpg` (pessoa sem azul, resultado esperado: alarme). **Envie essa foto antes** pelo painel de arquivos do Colab (ícone de pasta).
5. **Seção 4, teste em vídeo:** envie um vídeo pelo painel de arquivos e coloque o nome dele em `ENTRADA`.
6. **Seções 5 e 6:** geram a sirene, juntam ao vídeo e mostram o resultado. Ligue o volume.
7. **Seção 7:** para baixar o vídeo, tire o `##` das duas linhas.

### Opção B: no computador (tempo real, com webcam)

Requer Python 3.9 ou mais novo.

```bash
# 1. Criar e ativar um ambiente virtual (opcional, recomendado)
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Baixar o modelo YOLOv4-tiny para a pasta modelo/
python baixar_modelo.py

# 4. Rodar
python alarme_local.py                              # webcam padrão
python alarme_local.py --fonte 1                    # outra câmera
python alarme_local.py --fonte video.mp4            # arquivo de vídeo
python alarme_local.py --fonte video.mp4 --saida resultado.mp4   # e grava o resultado
