import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm # Barra de progresso (opcional, mas bom ter)

# ==============================================================================
# ⚙️ CONFIGURAÇÕES (Mexa aqui)
# ==============================================================================
# Pasta onde estão as imagens originais
caminho_entrada = Path(r'D:\github\geoeyeai\output') 

# Pasta onde serão salvas as corrigidas
caminho_saida = Path(r'D:\github\geoeyeai\cell_data\outputnovo')

# O prefixo que a imagem precisa ter para ser girada
prefixo_alvo = "trinca_"

# Ângulo de rotação:
# 90  = Anti-horário (Esquerda) ⬅️
# -90 = Horário (Direita) ➡️
# 180 = De cabeça para baixo ⬇️
angulo_rotacao = 90 

# ==============================================================================
# 🚀 O SCRIPT
# ==============================================================================

# Cria a pasta de saída se ela não existir
caminho_saida.mkdir(parents=True, exist_ok=True)

print(f"📂 Lendo imagens de: {caminho_entrada}")
print(f"🎯 Buscando prefixo: '{prefixo_alvo}'")
print(f"🔄 Girando: {angulo_rotacao} graus")
print("-" * 50)

contador = 0
arquivos = list(caminho_entrada.iterdir())

for arquivo in tqdm(arquivos, desc="Processando"):
    # Verifica se é imagem e se começa com o prefixo
    if arquivo.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] and arquivo.name.startswith(prefixo_alvo):
        try:
            # 1. Abre a imagem
            with Image.open(arquivo) as img:
                
                # 2. Gira a imagem
                # expand=True é CRUCIAL: ele redimensiona a tela para a imagem não ser cortada
                img_rotacionada = img.rotate(angulo_rotacao, expand=True)
                
                # 3. Define o caminho de salvamento
                destino = caminho_saida / arquivo.name
                
                # 4. Salva (mantendo a qualidade máxima se for JPG)
                img_rotacionada.save(destino, quality=95)
                
                contador += 1
                
        except Exception as e:
            print(f"❌ Erro ao processar {arquivo.name}: {e}")

print("-" * 50)
print(f"✅ Concluído! {contador} imagens foram giradas e salvas em:")
print(f"📂 {caminho_saida}")