import os
import pandas as pd
from gtts import gTTS
from moviepy.editor import *
from datetime import datetime
import random

# Cria pasta de saída
os.makedirs("output", exist_ok=True)

# Lê a lista de salmos
df = pd.read_csv("salmos.csv")

# Seleciona o primeiro salmo pendente
linha = df[df["Status"] == "pendente"].head(1)
if linha.empty:
    print("Nenhum salmo pendente encontrado.")
    exit()

salmo_num = int(linha["Salmo"].values[0])
tema = linha["Tema"].values[0]

# Gera o texto principal
texto = f"Salmo {salmo_num}. Tema: {tema}. Louvemos ao Senhor. Este é um salmo de fé e esperança."

# Cria o áudio
tts = gTTS(text=texto, lang="pt", slow=False)
tts.save("output/audio.mp3")

# Cria o vídeo
if not os.path.exists("assets"):
    os.makedirs("assets/images", exist_ok=True)
    os.makedirs("assets/music", exist_ok=True)

# Imagem de fundo (pega uma imagem qualquer da pasta)
imagens = [f"assets/images/{f}" for f in os.listdir("assets/images") if f.lower().endswith((".jpg", ".png"))]
imagem_fundo = random.choice(imagens) if imagens else None

# Música de fundo
musicas = [f"assets/music/{f}" for f in os.listdir("assets/music") if f.lower().endswith((".mp3", ".wav"))]
musica_fundo = random.choice(musicas) if musicas else None

# Montagem do vídeo
audio = AudioFileClip("output/audio.mp3")
duracao = audio.duration

if imagem_fundo:
    clip_img = ImageClip(imagem_fundo).set_duration(duracao)
else:
    clip_img = ColorClip(size=(1920,1080), color=(0,0,0), duration=duracao)

if musica_fundo:
    musica = AudioFileClip(musica_fundo).volumex(0.2)
    audio_final = CompositeAudioClip([audio, musica])
else:
    audio_final = audio

video = clip_img.set_audio(audio_final)
video.write_videofile("output/video.mp4", fps=24)

# Atualiza o CSV
df.loc[df["Salmo"] == salmo_num, "Status"] = "publicado"
df.to_csv("salmos.csv", index=False)

print(f"✅ Vídeo do Salmo {salmo_num} gerado com sucesso!")
