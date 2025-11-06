import os, json, random, math, wave, struct, subprocess, shlex, asyncio, re, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIGS
# =========================
WIDTH, HEIGHT = 1920, 1080
THUMB_W, THUMB_H = 1280, 720
SR = 44100
MIN_DURATION = 600  # 10 min
VOICE = "pt-BR-FranciscaNeural"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

os.makedirs("output", exist_ok=True)

def info(msg): print("ℹ️", msg)
def ok(msg):   print("✅", msg)
def warn(msg): print("⚠️", msg)
def err(msg):  print("❌", msg, file=sys.stderr)

# =========================
# CSV (cria se faltar)
# =========================
DEFAULT_CSV = """Salmo,Tema,Status
1,Fé e obediência,pendente
23,O Senhor é meu pastor,pendente
27,Confiança e coragem,pendente
46,Deus é refúgio e fortaleza,pendente
51,Arrependimento e misericórdia,pendente
91,Proteção divina,pendente
121,Socorro que vem do Senhor,pendente
"""
if not os.path.exists("salmos.csv"):
    with open("salmos.csv","w",encoding="utf-8") as f:
        f.write(DEFAULT_CSV)
        info("salmos.csv criado (padrão).")

# =========================
# LER SALMO
# =========================
df = pd.read_csv("salmos.csv")
linha = df[df["Status"].astype(str).str.strip().str.lower()=="pendente"].head(1)
if linha.empty:
    info("Nenhum salmo pendente. Finalizando.")
    raise SystemExit(0)

salmo = int(linha["Salmo"].values[0])
tema  = str(linha["Tema"].values[0]).strip()

title = f"Salmo {salmo} – {tema} | Oração guiada de 10 minutos"
description_header = (
    f"Oração guiada inspirada no Salmo {salmo} ({tema}).\n"
    "Feche os olhos, respire e entregue-se à presença divina.\n\n"
    "Inscreva-se para receber novas orações diariamente.\n"
    f"#salmo{salmo} #oração #fé #esperança\n"
)

# =========================
# 1) TEXTO (fallback sem keys)
# =========================
def gerar_oracao_fallback(salmo:int, tema:str)->str:
    base = [
        f"Senhor, Tu és nosso refúgio e fortaleza, conforme o Salmo {salmo}, cujo tema é {tema}.",
        "Conheces nossas dores e esperanças; acolhe-nos com Tua paz que excede todo entendimento.",
        "No silêncio, cura memórias, serena pensamentos e fortalece a fé.",
        "Ilumina nossos passos, transforma medo em coragem e dor em esperança.",
        "Abençoa as famílias; dá ânimo aos cansados; cura os enfermos; consola os que choram.",
        "Ensina-nos a amar, perdoar e servir com humildade e generosidade.",
        "Que Teu Espírito renove nossa mente e nos faça descansar em Tua graça.",
        "Quando a noite for longa, sê a nossa luz; quando o caminho for estreito, sê nosso guia.",
        "Que a Tua misericórdia cubra nossa casa, nossos sonhos e decisões.",
        "Dá-nos um coração agradecido para reconhecer Teus cuidados diários.",
        "Que o céu da Tua presença se abra sobre nós, renovando tudo o que somos.",
        "Em Ti encontramos descanso; nada nos falta, pois Tu és o nosso Pastor. Amém."
    ]
    texto=[]
    for _ in range(22):  # ~10 min
        for p in base:
            texto.append(p)
    return " ".join(texto)

texto_oracao = gerar_oracao_fallback(salmo, tema)
texto_oracao = re.sub(r"\s+"," ",texto_oracao).strip()
ok("Texto de oração gerado.")

# =========================
# 2) TTS (Edge-TTS com fallback gTTS)
# =========================
nar_path = "output/narracao.mp3"

async def synth_edge(text, out_path):
    import edge_tts
    com = edge_tts.Communicate(text, VOICE, rate="+0%", volume="+0%")
    await com.save(out_path)

def synth_gtts(text, out_path):
    from gtts import gTTS
    gTTS(text=text, lang="pt", slow=False).save(out_path)

used_tts = None
try:
    import edge_tts  # só para testar disponibilidade
    asyncio.get_event_loop().run_until_complete(synth_edge(texto_oracao, nar_path))
    used_tts = "edge-tts"
    ok("Narração gerada com Edge-TTS.")
except Exception as e:
    warn(f"Edge-TTS falhou ({e}); usando gTTS.")
    try:
        synth_gtts(texto_oracao, nar_path)
        used_tts = "gTTS"
        ok("Narração gerada com gTTS.")
    except Exception as e2:
        err(f"Falha ao gerar narração: {e2}")
        raise

def media_duration(path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "{path}"'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    try:
        return float(out)
    except:
        return 0.0

nar_dur    = media_duration(nar_path)
target_dur = max(int(round(nar_dur))+5, MIN_DURATION)
info(f"Duração narração ~ {nar_dur:.1f}s | alvo >= {target_dur}s")

# =========================
# 3) TRILHA AMBIENTE (procedural)
# =========================
def envelope(sig, sr, attack=1.2, release=3.0):
    n=len(sig); env=np.ones_like(sig, dtype=np.float32)
    a=min(int(sr*attack), n-1); r=min(int(sr*release), n-1)
    if a>0: env[:a]=np.linspace(0,1,a)
    if r>0: env[-r:]=np.linspace(1,0,r)
    return sig*env

def pad_chord(dur, root=220.0):
    t=np.linspace(0,dur,int(SR*dur),endpoint=False)
    freqs=[root, root*5/4, root*3/2]
    s=np.zeros_like(t, dtype=np.float32)
    for f in freqs:
        det=f*(1+random.uniform(-0.004,0.004))
        s+=np.sin(2*np.pi*det*t).astype(np.float32)
    s/=len(freqs); s=envelope(s,SR); s=np.convolve(s,np.ones(200)/200,mode="same")
    return s.astype(np.float32)

roots=[196.0,220.0,246.94,174.61]
music=np.zeros(int(SR*target_dur), dtype=np.float32)
pos=0.0
while pos<target_dur:
    d=min(6.0, target_dur-pos)
    seg=pad_chord(d, random.choice(roots))
    a=int(pos*SR); b=a+len(seg)
    music[a:b]=seg
    pos+=d
music=music/max(1e-6,np.max(np.abs(music)))*0.20

amb_wav="output/ambiente.wav"
with wave.open(amb_wav,"w") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((music*32767).astype(np.int16).tobytes())
ok("Trilha ambiente gerada.")

# =========================
# 4) CENAS CELESTIAIS (sem internet/keys)
# =========================
def radial_mask(w,h, cx,cy, r, power=2.2):
    y,x=np.ogrid[:h,:w]
    dist=np.sqrt((x-cx)**2+(y-cy)**2)
    m=1.0 - np.clip(dist/r,0,1)
    return np.power(m, power)

def fbm_clouds(w,h, octaves=5, persistence=0.55, lacunarity=2.1, seed=None):
    rng=np.random.default_rng(seed)
    base=rng.random((h//8+1, w//8+1)).astype(np.float32)
    img=np.zeros((h,w),dtype=np.float32)
    amp=1.0; amp_sum=0.0
    for _ in range(octaves):
        im = Image.fromarray((base*255).astype(np.uint8)).resize((w,h), Image.BICUBIC)
        arr=np.array(im).astype(np.float32)/255.0
        img += arr*amp
        amp_sum += amp
        base = rng.random((max(2,base.shape[0]//2), max(2,base.shape[1]//2))).astype(np.float32)
        amp *= persistence
    img/=max(1e-6,amp_sum)
    img=np.clip(img,0,1)
    return img

def make_celestial(w=WIDTH,h=HEIGHT, seed=None):
    top=(18,28,68); bottom=(220,180,90)
    bg=np.zeros((h,w,3),dtype=np.float32)
    for y in range(h):
        t=y/(h-1)
        r=top[0]*(1-t)+bottom[0]*t
        g=top[1]*(1-t)+bottom[1]*t
        b=top[2]*(1-t)+bottom[2]*t
        bg[y,:]=[r,g,b]
    bg/=255.0
    clouds=fbm_clouds(w,h, seed=seed)
    clouds=np.power(clouds,1.6)[...,None]
    sky = np.clip(bg*0.6 + clouds*0.7, 0, 1)
    ray = radial_mask(w,h, int(w*0.3), int(h*0.35), int(min(w,h)*0.65), 2.8)
    ray2= radial_mask(w,h,int(w*0.75),int(h*0.75),int(min(w,h)*0.55), 3.2)
    rays=((ray+0.8*ray2)[...,None])
    gold=np.array([255,230,140], dtype=np.float32)/255.0
    sky = np.clip(sky + rays*gold*0.35, 0, 1)
    img=(sky*255).astype(np.uint8)
    return Image.fromarray(img, "RGB")

img_paths=[]; num_scenes=10
for i in range(num_scenes):
    p=f"output/bg_{i:02d}.jpg"
    make_celestial(seed=1000+i*37).save(p,"JPEG",quality=92)
    img_paths.append(p)
ok("Cenas celestiais geradas.")

# =========================
# 5) SEGMENTOS COM FADE (transição suave)
# =========================
per_img = target_dur / num_scenes
fade_d  = min(2.5, per_img/4)
seg_list_path="output/segments.txt"
seg_paths=[]

for i, p in enumerate(img_paths):
    seg=f"output/seg_{i:02d}.mp4"
    seg_paths.append(seg)
    cmd = (
        f'ffmpeg -y -loop 1 -t {per_img:.3f} -i "{p}" '
        f'-vf "scale={WIDTH}:{HEIGHT},format=yuv420p,'
        f'fade=t=in:st=0:d={fade_d:.2f},'
        f'fade=t=out:st={per_img-fade_d:.2f}:d={fade_d:.2f}" '
        f'-r 24 -c:v libx264 -pix_fmt yuv420p "{seg}"'
    )
    subprocess.run(shlex.split(cmd), check=True)

with open(seg_list_path,"w") as f:
    for sp in seg_paths:
        f.write(f"file '{os.path.abspath(sp)}'\n")

slides_mp4="output/slides.mp4"
cmd_concat=f'ffmpeg -y -f concat -safe 0 -i "{seg_list_path}" -c:v libx264 -pix_fmt yuv420p -r 24 "{slides_mp4}"'
subprocess.run(shlex.split(cmd_concat), check=True)
ok("Slides concatenados.")

# =========================
# 6) LEGENDAS (SRT)
# =========================
def split_sentences(text):
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def sec_to_ts(s):
    h=int(s//3600); s-=h*3600
    m=int(s//60);   s-=m*60
    ms=int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"

frases = split_sentences(texto_oracao)
leg_dur = max(target_dur*0.98, nar_dur)
dur_por = leg_dur / max(1,len(frases))

srt_path="output/subs.srt"
t=0.0
with open(srt_path,"w",encoding="utf-8") as srt:
    for i, fr in enumerate(frases, start=1):
        ini=t; fim=min(ini+dur_por, leg_dur)
        srt.write(f"{i}\n{sec_to_ts(ini)} --> {sec_to_ts(fim)}\n{fr}\n\n")
        t=fim
ok("Legendas SRT geradas.")

# =========================
# 7) MIX ÁUDIO (narração + trilha) com fades
# =========================
mix_audio="output/mix.m4a"
afade_out_start = max(0.0, target_dur-3.0)
cmd_mix = (
    f'ffmpeg -y -i "{nar_path}" -i "{amb_wav}" '
    f'-filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.45[a1];'
    f'[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0,'
    f'afade=t=in:ss=0:d=2,afade=t=out:st={afade_out_start:.2f}:d=3" '
    f'-c:a aac -b:a 192k "{mix_audio}"'
)
subprocess.run(shlex.split(cmd_mix), check=True)
ok("Áudio mixado.")

# =========================
# 8) OVERLAY DOURADO + LEGENDAS QUEIMADAS
# =========================
def golden_vignette(w,h):
    cx,cy=int(w*0.5),int(h*0.5)
    R=int(min(w,h)*0.65)
    y,x=np.ogrid[:h,:w]
    dist=np.sqrt((x-cx)**2+(y-cy)**2)
    m=1.0 - np.clip(dist/R,0,1)
    m=np.power(m,2.0)
    gold=np.zeros((h,w,4),dtype=np.float32)
    gold[...,:3]=np.array([255,215,120])/255.0
    gold[...,3]=(m*0.45).astype(np.float32)
    return Image.fromarray((gold*255).astype(np.uint8), "RGBA")

overlay_path="output/golden_overlay.png"
golden_vignette(WIDTH, HEIGHT).save(overlay_path,"PNG")

video_path="output/video.mp4"
T=float(target_dur)

# IMPORTANTE: use caminhos ABSOLUTOS no filtro de legendas para evitar erros
abs_srt = os.path.abspath(srt_path)
abs_overlay = os.path.abspath(overlay_path)
abs_slides = os.path.abspath(slides_mp4)

vf = (
    f"[0:v][2:v]overlay=shortest=1:format=auto:"
    f"enable='between(t,0,3)+between(t,{T-3:.3f},{T:.3f})'[v0];"
    f"[v0]subtitles='{abs_srt}':fontsdir=/usr/share/fonts/:"
    f"force_style='FontName=DejaVuSans,FontSize=34,PrimaryColour=&H00FFF7C2&,OutlineColour=&H00101010&,BorderStyle=1,Outline=2'[vf]"
)

cmd_final = (
    f'ffmpeg -y -i "{abs_slides}" -i "{mix_audio}" -loop 1 -t {T:.3f} -i "{abs_overlay}" '
    f'-filter_complex "{vf}" -map "[vf]" -map 1:a -c:v libx264 -pix_fmt yuv420p -c:a copy -shortest "{video_path}"'
)
subprocess.run(shlex.split(cmd_final), check=True)
ok("Vídeo final renderizado.")

# =========================
# 9) ATUALIZA CSV + META + THUMB
# =========================
df.loc[df["Salmo"]==salmo,"Status"]="feito"
df.to_csv("salmos.csv", index=False)

thumb = Image.open(img_paths[0]).resize((THUMB_W, THUMB_H))
thumb.save("output/thumbnail.jpg","JPEG",quality=92)

meta={"salmo":salmo,"tema":tema,"title":title,"description":description_header}
with open("output/meta.json","w",encoding="utf-8") as f:
    json.dump(meta,f,ensure_ascii=False, indent=2)

print("\n✅ VÍDEO GERADO COM SUCESSO!")
print(f"Duração alvo: {target_dur/60:.1f} min | TTS: {used_tts}")
