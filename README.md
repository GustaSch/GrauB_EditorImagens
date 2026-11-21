# Editor de Imagens e Vídeo (estilo *stories*)
 
Trabalho do **Grau B** — *Processamento Gráfico: Fundamentos* (UNISINOS).
Protótipo de aplicativo de edição de imagens e vídeo em **Python + OpenCV**, com foco em
**Processamento de Imagens**: filtros, operações aritméticas entre imagens, seleção de canais
e *stickers*. A interação é feita por **HighGUI (janela do OpenCV) + teclado + console**.
 
> **Autores:**
> %
> %
> Gustavo Ribeiro Schwert
 
## Requisitos
 
* Python 3.9 ou mais novo (no Windows, marque *Add Python to PATH* ao instalar)
* Pacotes: `numpy` e `opencv-python` (arquivo `requirements.txt`)
* Webcam (somente para o Modo Vídeo) e sistema com interface gráfica
## Instalação e execução
 
**Windows (PowerShell/CMD):**
```powershell
git clone https://github.com/SEU_USUARIO/GrauB_EditorImagens.git
cd GrauB_EditorImagens
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --image assets\samples\exemplo1.jpg
```
 
**Linux/macOS:**
```bash
git clone https://github.com/SEU_USUARIO/GrauB_EditorImagens.git
cd GrauB_EditorImagens
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --image assets/samples/exemplo1.jpg
```
 
Outras opções: `python main.py` (editor vazio) e `python main.py --camera 1` (outra webcam).
 
## Comandos
 
| Tecla | Ação |
|---|---|
| `l` | Carregar foto (digite o caminho no console) |
| `v` | Iniciar / parar a webcam (imagem espelhada) |
| `n` / `b` | Próximo / anterior efeito (mostra **descrição e parâmetros** no console) |
| `t` | Listar todos os efeitos |
| `p` | Escolher o próximo parâmetro do efeito |
| `+` / `-` | Aumentar / diminuir o parâmetro escolhido |
| `m` | Canal: **RGB → GRAY → R → G → B** |
| `a` (ou Enter / Espaço) | **Aplicar** o efeito selecionado |
| `c` | Alternar **encadeamento** de filtros (um filtro sobre o resultado do anterior) |
| `i` | Carregar a 2ª imagem usada nas operações aritméticas |
| `1`–`9`, `0` | Escolher sticker / nenhum; depois **clique** na foto para posicionar |
| `,` / `.` | Diminuir / aumentar o tamanho do sticker |
| `f` | Filtro de gatinho por **detecção de rosto** (foto e vídeo) |
| `z` | Desfazer a última ação (filtro ou sticker) |
| `s` | **Salvar** a imagem / o quadro atual em `output/` |
| `r` | **Resetar**: remove filtros e stickers, volta à imagem original |
| `ESC` ou `q` | Sair |
 
Fluxo típico: `l` → `n` até o filtro desejado (leia a descrição) → `p`, `+`/`-` para ajustar →
`m` para escolher o canal → `a` para aplicar → `s` para salvar.
 
## Estrutura
 
```
main.py              # ponto de entrada e laço da janela (HighGUI)
editor.py            # estado do editor, pipeline, teclas, mouse, HUD, webcam, rosto
filters.py           # filtros, operações aritméticas, descrições e seleção de canal
stickers.py          # carregamento de stickers (alfa), spritesheet e overlay
imageio_utils.py     # leitura/gravação de imagens e redimensionamento
generate_assets.py   # gera stickers (PNG com alfa) e imagens de exemplo
assets/              # stickers, stickers/face (orelhas e focinho) e samples
tests/test_core.py   # testes automáticos
output/              # imagens salvas (ignorada pelo git)
```
 
Para usar seus próprios stickers, coloque PNGs com transparência em `assets/stickers/`
(até 9 são selecionáveis, em ordem alfabética). `python generate_assets.py` recria os incluídos.
 
## Testes
 
```bash
python -m unittest discover -s tests -v
```
 
Cobrem todos os efeitos em todos os canais, valores das operações aritméticas, overlay com alfa
e recorte, animação do spritesheet, pipeline (aplicar, encadear, desfazer, resetar), salvamento
com caminho acentuado, tratamento de erros (arquivo inexistente, webcam indisponível) e o laço
principal com janela e webcam simuladas.
 
