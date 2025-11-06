import os, json, random, math, wave, struct
import numpy as np
import pandas as pd
from gtts import gTTS
from moviepy.editor import *
from PIL import Image, ImageDraw

# ---------- Configs ----------
WIDTH, HEIGHT = 1920, 1080
THUMB_W, THUMB_H = 1280, 720
MUSIC_SECONDS_MIN = 40          # duração mínima da trilha
SR = 44100                      # sample rate trilha
BPM = 60                        # "andamento" do pad
# -----------------------------

os.makedirs("output", exist_ok=True)

# 1) Pegar próximo salmo pendente
df = pd.read_csv("salmos.csv")
linha = df[df["Status"] == "pendente"].head(1)
if linha.empty:
    print("Nenhum salmo pendente.")
    raise SystemExit(0)

salmo = int(linha["Salmo"].values[0])
tema  = str(linha["Tema"].values[0])

title = f"Salmo {salmo} – {tema} | Oração e Meditação"
desc  = (
    f"Oração e meditação no Salmo {salmo} ({tema}).\n\n"
    "Inscreva-se para receber orações diárias.\n"
    f"#salmo{salmo} #oração #fé #esperança"
)

# 2) Texto curto e universal (evita trechos com copyright)
texto = f"Salmo {salmo}. Tema: {tema}. Senhor, recebe nossa oração. Guia-nos com fé e esperança. Amém."

# 3) Narração (gTTS)
tts = gTTS(text=texto, lang="pt", slow=False)
tts.save("output/audio.mp3")
nar = AudioFileClip("output/audio.mp3")
nar_duration = nar.duration

# 4) Gerar trilha ambiente ORIGINAL (sintetizada)
def envelope(sig, sr, attack=0.8, release=2.5):
    n = len(sig)
    t = np.linspace(0, n/sr, n, endpoint=False)
    env = np.ones_like(sig, dtype=np.float32)
    # fade in
    a_len = int(sr * attack)
    if a_len > 0 and a_len < n:
        env[:a_len] = np.linspace(0.0, 1.0, a_len)
    # fade out
    r_len = int(sr * release)
    if r_len > 0 and r_len < n:
        env[-r_len:] = np.linspace(1.0, 0.0, r_len)
    return sig * env

def pad_chord(duration, root_hz=220.0):
    """Gera um acorde suave (root + quinta + oitava) com leve detune."""
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)
    freqs = [root_hz, root_hz*1.5, root_hz*2.0]  # fundamental, quinta, oitava
    sig = np.zeros_like(t, dtype=np.float32)
    for f in freqs:
        detune = f * (1 + random.uniform(-0.003, 0.003))
        sig += np.sin(2*np.pi*detune*t).astype(np.float32)
    sig /= len(freqs)
    sig = envelope(sig, SR)
    k = 100
    sig = np.convolve(sig, np.ones(k)/k, mode='same')
    return sig.astype(np.float32)

# duração alvo: pelo menos a narração + “respiro”
target_dur = max(MUSIC_SECONDS_MIN, int(nar_duration) + 8)

pad = np.zeros(int(SR*target_dur), dtype=np.float32)
roots = [196.0, 220.0, 246.94, 174.61]  # G3, A3, B3, F3 aprox
block = 4.0
pos = 0
while pos < target_dur:
    d = min(block, target_dur - pos)
    root = random.choice(roots)
    seg = pad_chord(d, root_hz=root)
    pad[int(pos*SR):int((pos+d)*SR)] = seg
    pos += d

pad = pad / max(1e-6, np.max(np.abs(pad))) * 0.25

wav_path = "output/ambient.wav"
with wave.open(wav_path, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    for s in (pad * 32767).astype(np.int16):
        wf.writeframes(struct.pack('<h', int(s)))

music = AudioFileClip(wav_path)

# 5) Criar fundo procedural (gradiente + glows)
def gradient_bg(w, h, top=(15,20,60), bottom=(0,0,0)):
    img = Image.new("RGB", (w, h), bottom)
    draw = ImageDraw.Draw(img)
    for y in range(h):
        r = int(top[0] + (bottom[0]-top[0]) * y / h)
        g = int(top[1] + (bottom[1]-top[1]) * y / h)
        b = int(top[2] + (bottom[2]-top[2]) * y / h)
        draw.line([(0,y),(w,y)], fill=(r,g,b))
    def glow(cx, cy, radius, color, alpha=0.25):
        overlay = Image.new("RGBA", (w,h), (0,0,0,0))
        od = ImageDraw.Draw(overlay)
        for r in range(radius, 0, -8):
            a = int(alpha*255 * (r/radius))
            od.ellipse((cx-r, cy-r, cx+r, cy+r), fill=color+(a,))
        return overlay
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, glow(int(w*0.25), int(h*0.3), 300, (240,220,120), 0.18))
    img = Image.alpha_composite(img, glow(int(w*0.75), int(h*0.7), 260, (120,180,255), 0.15))
    return img.convert("RGB")

bg_img = gradient_bg(WIDTH, HEIGHT)
bg_path = "output/bg.jpg"
bg_img.save(bg_path, "JPEG", quality=92)

# 6) Montar vídeo
clip_bg = ImageClip(bg_path).set_duration(nar_duration + 4)
audio_final = CompositeAudioClip([nar.volumex(1.0), music.volumex(0.8).set_duration(clip_bg.duration)])
video = clip_bg.set_audio(audio_final)
video.write_videofile("output/video.mp4", fps=24, codec="libx264", audio_codec="aac")

# 7) Thumbnail (reduz o mesmo fundo para 1280x720)
thumb = bg_img.resize((THUMB_W, THUMB_H))
thumb.save("output/thumbnail.jpg", "JPEG", quality=92)

# 8) Atualiza CSV (marca como publicado) + meta.json
df.loc[df["Salmo"]==salmo, "Status"] = "publicado"
df.to_csv("salmos.csv", index=False)

with open("output/meta.json","w",encoding="utf-8") as f:
    json.dump({"salmo": salmo, "tema": tema, "title": title, "description": desc}, f, ensure_ascii=False)

print(f"✅ Vídeo do Salmo {salmo} gerado com sucesso (sem imagens/músicas externas)!")
print(f"Título sugerido: {title}")
