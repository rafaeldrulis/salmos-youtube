import os, json, random, math, wave, struct, subprocess, shlex
import numpy as np
import pandas as pd
from gtts import gTTS
from PIL import Image, ImageDraw

# ---------- Configs ----------
WIDTH, HEIGHT = 1920, 1080
THUMB_W, THUMB_H = 1280, 720
MUSIC_SECONDS_MIN = 40          # duração mínima da trilha
SR = 44100                      # sample rate trilha
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

# 2) Texto curto e universal
texto = f"Salmo {salmo}. Tema: {tema}. Senhor, recebe nossa oração. Guia-nos com fé e esperança. Amém."

# 3) Narração (gTTS)
nar_path = "output/audio.mp3"
gTTS(text=texto, lang="pt", slow=False).save(nar_path)

# 4) Trilha ambiente procedural (numPy -> WAV)
def envelope(sig, sr, attack=0.8, release=2.5):
    n = len(sig)
    env = np.ones_like(sig, dtype=np.float32)
    a_len = int(sr * attack)
    if 0 < a_len < n:
        env[:a_len] = np.linspace(0.0, 1.0, a_len)
    r_len = int(sr * release)
    if 0 < r_len < n:
        env[-r_len:] = np.linspace(1.0, 0.0, r_len)
    return sig * env

def pad_chord(duration, root_hz=220.0):
    t = np.linspace(0, duration, int(SR*duration), endpoint=False)
    freqs = [root_hz, root_hz*1.5, root_hz*2.0]
    sig = np.zeros_like(t, dtype=np.float32)
    for f in freqs:
        detune = f * (1 + random.uniform(-0.003, 0.003))
        sig += np.sin(2*np.pi*detune*t).astype(np.float32)
    sig /= len(freqs)
    sig = envelope(sig, SR)
    k = 100
    sig = np.convolve(sig, np.ones(k)/k, mode='same')
    return sig.astype(np.float32)

def get_duration(audio_file):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_file}"'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    return float(out)

nar_duration = get_duration(nar_path)
target_dur = int(max(MUSIC_SECONDS_MIN, nar_duration + 8))

pad = np.zeros(int(SR*target_dur), dtype=np.float32)
roots = [196.0, 220.0, 246.94, 174.61]
block = 4.0
pos = 0.0
while pos < target_dur:
    d = min(block, target_dur - pos)
    root = random.choice(roots)
    seg = pad_chord(d, root_hz=root)
    a = int(pos*SR); b = int((pos+d)*SR)
    pad[a:b] = seg[:(b-a)]
    pos += d

pad = pad / max(1e-6, np.max(np.abs(pad))) * 0.25
amb_wav = "output/ambient.wav"
with wave.open(amb_wav, "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes((pad * 32767).astype(np.int16).tobytes())

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

bg = gradient_bg(WIDTH, HEIGHT)
bg_path = "output/bg.jpg"
bg.save(bg_path, "JPEG", quality=92)

# 6) Mixar narração + música com FFmpeg
mix_audio = "output/mix.m4a"
cmd_mix = (
    f'ffmpeg -y -i "{nar_path}" -i "{amb_wav}" '
    f'-filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.55[a1];'
    f'[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0" '
    f'-c:a aac -b:a 192k "{mix_audio}"'
)
subprocess.run(shlex.split(cmd_mix), check=True)

# 7) Gerar vídeo com imagem estática + áudio final
video_path = "output/video.mp4"
cmd_vid = (
    f'ffmpeg -y -loop 1 -i "{bg_path}" -i "{mix_audio}" '
    f'-c:v libx264 -t {target_dur} -pix_fmt yuv420p -c:a copy "{video_path}"'
)
subprocess.run(shlex.split(cmd_vid), check=True)

# 8) Atualizar CSV
df.loc[df["Salmo"] == salmo, "Status"] = "feito"
df.to_csv("salmos.csv", index=False)

# 9) Salvar metadados
meta = {"salmo": salmo, "tema": tema, "title": title, "desc": desc}
json.dump(meta, open("output/meta.json", "w"), ensure_ascii=False, indent=2)

print(f"\n✅ Vídeo gerado com sucesso: {video_path}")
