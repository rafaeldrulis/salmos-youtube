import os, json, random, math, wave, struct, subprocess, shlex, asyncio, re
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# =========================
# CONFIGS GERAIS
# =========================
WIDTH, HEIGHT = 1920, 1080
THUMB_W, THUMB_H = 1280, 720
SR = 44100
MIN_DURATION = 600  # 10 minutos mínimo
VOICE = "pt-BR-FranciscaNeural"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

os.makedirs("output", exist_ok=True)

# =========================
# CSV PADRÃO (cria se faltar)
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

# =========================
# LER PRÓXIMO SALMO
# =========================
df = pd.read_csv("salmos.csv")
linha = df[df["Status"].astype(str).str.strip().str.lower()=="pendente"].head(1)
if linha.empty:
    print("Nenhum salmo pendente.")
    raise SystemExit(0)

salmo = int(linha["Salmo"].values[0])
tema  = str(linha["Tema"].values[0]).strip()

title = f"Salmo {salmo} – {tema} | Oração guiada de 10 minutos"
description_header = (
    f"Oração guiada inspirada no Salmo {salmo} ({tema}).\n"
    "Feche os olhos, respire, entregue-se à presença divina.\n\n"
    "Inscreva-se para receber novas orações diariamente.\n"
    f"#salmo{salmo} #oração #fé #esperança\n"
)

# =========================
# 1) GERAR TEXTO (~10 min) SEM KEYS
# =========================
def gerar_oracao_fallback(salmo:int, tema:str)->str:
    base = [
        f"Senhor, Tu és nosso refúgio e fortaleza, como proclama o Salmo {salmo}, cujo tema é {tema}.",
        "Conheces as aflições e as esperanças do nosso coração; acolhe-nos com Tua paz.",
        "No silêncio da oração, cura as memórias, serena os pensamentos e fortalece a fé.",
        "Ilumina nossos passos, transforma medo em coragem e dor em esperança.",
        "Abençoa as famílias, dá ânimo aos cansados, cura os enfermos, consola os que choram.",
        "Ensina-nos a amar, a perdoar e a servir com humildade e generosidade.",
        "Que Teu Espírito Santo nos visite agora, trazendo vida nova e confiança.",
        "Quando a noite for longa, sê a nossa luz; quando o caminho for estreito, sê o nosso guia.",
        "Que a Tua misericórdia cubra nossa casa, nossos sonhos e decisões.",
        "Dá-nos um coração agradecido, capaz de reconhecer Teus cuidados de cada dia.",
        "Que o céu da Tua presença se abra sobre nós, renovando tudo o que somos.",
        "Em Ti encontramos descanso, e nada nos falta, porque Tu és o nosso Pastor. Amém."
    ]
    texto=[]
    # repete e intercala para dar ~10 min de narração
    for _ in range(22):
        for p in base:
            texto.append(p)
    return " ".join(texto)

texto_oracao = gerar_oracao_fallback(salmo, tema)
texto_oracao = re.sub(r"\s+"," ",texto_oracao).strip()

# =========================
# 2) NARRAÇÃO (EDGE-TTS)
# =========================
async def synth(text, out_path):
    import edge_tts
    com = edge_tts.Communicate(text, VOICE, rate="+0%", volume="+0%")
    await com.save(out_path)

nar_path = "output/narracao.mp3"
asyncio.get_event_loop().run_until_complete(synth(texto_oracao, nar_path))

def media_duration(path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "{path}"'
    out = subprocess.check_output(cmd, shell=True, text=True).strip()
    try:
        return float(out)
    except:
        return 0.0

nar_dur   = media_duration(nar_path)
target_dur= max(int(round(nar_dur))+5, MIN_DURATION)

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

# =========================
# 4) IMAGENS “CÉU/PARAÍSO” SEM KEYS (fractal noise + gradientes)
# =========================
def radial_mask(w,h, cx,cy, r, power=2.2):
    """gera máscara radial (0..1) com decaimento suave"""
    y,x=np.ogrid[:h,:w]
    dist=np.sqrt((x-cx)**2+(y-cy)**2)
    m=1.0 - np.clip(dist/r,0,1)
    return np.power(m, power)

def fbm_clouds(w,h, octaves=5, persistence=0.55, lacunarity=2.1, seed=None):
    rng=np.random.default_rng(seed)
    base=rng.random((h//8+1, w//8+1)).astype(np.float32)
    img=np.zeros((h,w),dtype=np.float32)
    freq=1.0; amp=1.0; amp_sum=0.0
    for _ in range(octaves):
        small = base
        sh,sw=small.shape
        # upscale bicubic via PIL
        im = Image.fromarray((small*255).astype(np.uint8)).resize((w,h), Image.BICUBIC)
        arr=np.array(im).astype(np.float32)/255.0
        img += arr*amp
        amp_sum += amp
        # next octave
        base = rng.random((max(2,int(sh/lacunarity)), max(2,int(sw/lacunarity)))).astype(np.float32)
        amp *= persistence
        freq *= lacunarity
    img/=max(1e-6,amp_sum)
    img=np.clip(img,0,1)
    return img

def make_celestial(w=WIDTH,h=HEIGHT, seed=None):
    # gradiente de fundo azul→dourado
    top=(18,28,68); bottom=(220,180,90)  # azul escuro → dourado suave
    bg=np.zeros((h,w,3),dtype=np.float32)
    for y in range(h):
        t=y/(h-1)
        r=top[0]*(1-t)+bottom[0]*t
        g=top[1]*(1-t)+bottom[1]*t
        b=top[2]*(1-t)+bottom[2]*t
        bg[y,:]=[r,g,b]
    bg/=255.0

    # nuvens FBM
    clouds=fbm_clouds(w,h, seed=seed)
    clouds=np.power(clouds,1.6)  # contraste suave
    clouds=clouds[...,None]

    # compõe nuvens sobre gradiente
    sky = np.clip(bg*0.6 + clouds*0.7, 0, 1)

    # raios de luz radiais dourados
    ray = radial_mask(w,h, int(w*0.3), int(h*0.35), int(min(w,h)*0.65), power=2.8)
    ray2= radial_mask(w,h, int(w*0.75),int(h*0.75), int(min(w,h)*0.55), power=3.2)
    rays=((ray+0.8*ray2)[...,None])
    gold=np.array([255,230,140], dtype=np.float32)/255.0
    sky = np.clip(sky + rays*gold*0.35, 0, 1)

    img=(sky*255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")

# gera 10 cenas celestiais diferentes
img_paths=[]
num_scenes=10
for i in range(num_scenes):
    im=make_celestial(seed=1000+i*37)
    p=f"output/bg_{i:02d}.jpg"
    im.save(p,"JPEG", quality=92)
    img_paths.append(p)

# =========================
# 5) GERAR SEGMENTOS COM FADE (transição suave)
# =========================
per_img = target_dur / num_scenes
fade_d  = min(2.5, per_img/4)  # fade in/out ~2.5s (ou 1/4 do segmento)
seg_list_path="output/segments.txt"
seg_paths=[]

for i, p in enumerate(img_paths):
    seg=f"output/seg_{i:02d}.mp4"
    seg_paths.append(seg)
    # fade in/out na própria imagem
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

# concatena segmentos já com fades (resultado: slides.mp4)
slides_mp4="output/slides.mp4"
cmd_concat=f'ffmpeg -y -f concat -safe 0 -i "{seg_list_path}" -c:v libx264 -pix_fmt yuv420p -r 24 "{slides_mp4}"'
subprocess.run(shlex.split(cmd_concat), check=True)

# =========================
# 6) LEGENDAS SRT (queimadas depois)
# =========================
def split_sentences(text):
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def sec_to_ts(s):
    h = int(s//3600); s-=h*3600
    m = int(s//60);   s-=m*60
    ms= int((s-int(s))*1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"

frases = split_sentences(texto_oracao)
leg_dur = max(target_dur*0.98, nar_dur)  # reserva 2% pro fade final
dur_por = leg_dur / max(1,len(frases))

srt_path="output/subs.srt"
t=0.0
with open(srt_path,"w",encoding="utf-8") as srt:
    for i, fr in enumerate(frases, start=1):
        ini=t; fim=min(ini+dur_por, leg_dur)
        srt.write(f"{i}\n{sec_to_ts(ini)} --> {sec_to_ts(fim)}\n{fr}\n\n")
        t=fim

# =========================
# 7) MIX ÁUDIO (narração + trilha) com AFADES
# =========================
mix_audio="output/mix.m4a"
afade_out_start = max(0.0, target_dur-3.0)  # 3s finais
cmd_mix = (
    f'ffmpeg -y -i "{nar_path}" -i "{amb_wav}" '
    f'-filter_complex "[0:a]volume=1.0[a0];[1:a]volume=0.45[a1];'
    f'[a0][a1]amix=inputs=2:duration=longest:dropout_transition=0,'
    f'afade=t=in:ss=0:d=2,afade=t=out:st={afade_out_start:.2f}:d=3" '
    f'-c:a aac -b:a 192k "{mix_audio}"'
)
subprocess.run(shlex.split(cmd_mix), check=True)

# =========================
# 8) OVERLAY DE BRILHO DOURADO (início/fim) + QUEIMAR LEGENDAS
# =========================
# cria PNG de vinheta dourada (radial) para overlay
def golden_vignette(w,h):
    cx,cy=int(w*0.5), int(h*0.5)
    R=int(min(w,h)*0.65)
    y,x=np.ogrid[:h,:w]
    dist=np.sqrt((x-cx)**2+(y-cy)**2)
    m=1.0 - np.clip(dist/R, 0,1)
    m=np.power(m,2.0)
    # centro translúcido dourado
    gold=np.zeros((h,w,4),dtype=np.float32)
    gold[...,:3]=np.array([255,215,120])/255.0
    gold[..., 3]= (m*0.45).astype(np.float32)  # alpha
    img=(gold*255).astype(np.uint8)
    return Image.fromarray(img, mode="RGBA")

overlay_path="output/golden_overlay.png"
gold = golden_vignette(WIDTH, HEIGHT)
gold.save(overlay_path, "PNG")

# aplica overlay dourado só no início (0-3s) e fim (T-3..T) + queima legendas
video_path="output/video.mp4"
T = float(target_dur)
vf = (
    f"[0:v][2:v]overlay=shortest=1:format=auto:"
    f"enable='between(t,0,3)+between(t,{T-3:.3f},{T:.3f})'[v0];"
    f"[v0]subtitles={srt_path}:fontsdir=/usr/share/fonts/:"
    f"force_style='FontName=DejaVuSans,FontSize=34,PrimaryColour=&H00FFF7C2&,OutlineColour=&H00101010&,BorderStyle=1,Outline=2'[v1];"
    f"[v1]format=yuv420p[vf]"
)

cmd_final = (
    f'ffmpeg -y -i "{slides_mp4}" -i "{mix_audio}" -loop 1 -t {T:.3f} -i "{overlay_path}" '
    f'-filter_complex "{vf}" -map "[vf]" -map 1:a -c:v libx264 -pix_fmt yuv420p -c:a copy -shortest "{video_path}"'
)
subprocess.run(shlex.split(cmd_final), check=True)

# =========================
# 9) ATUALIZA CSV + META + THUMB
# =========================
df.loc[df["Salmo"]==salmo,"Status"]="feito"
df.to_csv("salmos.csv", index=False)

thumb = Image.open(img_paths[0]).resize((THUMB_W, THUMB_H))
thumb.save("output/thumbnail.jpg","JPEG", quality=92)

meta={"salmo":salmo,"tema":tema,"title":title,"description":description_header}
with open("output/meta.json","w",encoding="utf-8") as f:
    json.dump(meta,f,ensure_ascii=False, indent=2)

print("\n✅ VÍDEO GERADO COM SUCESSO!")
print(f"Duração alvo: {target_dur/60:.1f} min")
