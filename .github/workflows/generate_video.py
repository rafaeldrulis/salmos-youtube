import os, json, random
import pandas as pd
from gtts import gTTS
from moviepy.editor import *
from PIL import Image

# Pastas
os.makedirs("output", exist_ok=True)
os.makedirs("assets/images", exist_ok=True)
os.makedirs("assets/music", exist_ok=True)

# Lê CSV e pega o próximo "pendente"
df = pd.read_csv("salmos.csv")
linha = df[df["Status"]=="pendente"].head(1)
if linha.empty:
    print("Nenhum salmo pendente.")
    raise SystemExit(0)

salmo = int(linha["Salmo"].values[0])
tema  = str(linha["Tema"].values[0])

# Título/descrição para facilitar upload manual
title = f"Salmo {salmo} – {tema} | Oração e Meditação"
desc  = (
    f"Oração e meditação no Salmo {salmo} ({tema}).\n\n"
    "Inscreva-se para receber orações diárias.\n"
    f"#salmo{salmo} #oração #fé #esperança"
)

# Texto falado (curto e universal)
texto = f"Salmo {salmo}. Tema: {tema}. Senhor, recebe nossa oração. Guia-nos com fé e esperança. Amém."

# Narração
tts = gTTS(text=texto, lang="pt", slow=False)
tts.save("output/audio.mp3")
audio = AudioFileClip("output/audio.mp3")
dur = max(35, audio.duration)  # garante pelo menos ~35s

# Imagem de fundo (ou tela preta)
imgs = [f"assets/images/{f}" for f in os.listdir("assets/images") if f.lower().endswith((".jpg",".jpeg",".png"))]
img_bg = random.choice(imgs) if imgs else None
clip_bg = ImageClip(img_bg).set_duration(dur) if img_bg else ColorClip(size=(1920,1080), color=(0,0,0), duration=dur)

# Música de fundo (se houver)
mus = [f"assets/music/{f}" for f in os.listdir("assets/music") if f.lower().endswith((".mp3",".wav",".m4a"))]
if mus:
    music = AudioFileClip(random.choice(mus)).volumex(0.18)
    audio_final = CompositeAudioClip([audio, music.set_duration(dur)])
else:
    audio_final = audio

# Vídeo final
video = clip_bg.set_audio(audio_final)
video.write_videofile("output/video.mp4", fps=24, codec="libx264", audio_codec="aac")

# Thumbnail simples (pega a imagem de fundo ou gera PNG preta)
thumb_path = "output/thumbnail.jpg"
if img_bg and os.path.exists(img_bg):
    # redimensiona para 1280x720 mantendo proporção
    im = Image.open(img_bg).convert("RGB")
    im = im.resize((1280, 720))
    im.save(thumb_path, "JPEG", quality=92)
else:
    im = Image.new("RGB", (1280, 720), (0, 0, 0))
    im.save(thumb_path, "JPEG", quality=92)

# Atualiza CSV (marca como publicado)
df.loc[df["Salmo"]==salmo, "Status"] = "publicado"
df.to_csv("salmos.csv", index=False)

# Metadados úteis para upload manual
with open("output/meta.json","w",encoding="utf-8") as f:
    json.dump({"salmo": salmo, "tema": tema, "title": title, "description": desc}, f, ensure_ascii=False)

print(f"✅ Vídeo do Salmo {salmo} gerado com sucesso!")
print(f"Título sugerido: {title}")
