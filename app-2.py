from flask import Flask, request, send_file, after_this_request, jsonify
import subprocess, os, tempfile, traceback
import multiprocessing, json, textwrap, shutil
import requests as http_requests

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CPU_CORES    = str(multiprocessing.cpu_count())
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FONTS_DIR    = os.path.join(BASE_DIR, "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

_FONT_CANDIDATES = [
    (os.path.join(FONTS_DIR, "Autumn_Regular.ttf"), "Autumn"),
    (os.path.join(BASE_DIR,  "Autumn_Regular.ttf"),  "Autumn"),
    (os.path.join(FONTS_DIR, "Anton-Regular.ttf"),   "Anton"),
    (os.path.join(BASE_DIR,  "Anton-Regular.ttf"),   "Anton"),
]
FONT_PATH, FONT_NAME = next(
    ((p, n) for p, n in _FONT_CANDIDATES if os.path.exists(p)),
    ("", "Impact")
)

if FONT_PATH and not FONT_PATH.startswith(FONTS_DIR):
    _dest = os.path.join(FONTS_DIR, os.path.basename(FONT_PATH))
    if not os.path.exists(_dest):
        shutil.copy2(FONT_PATH, _dest)
    FONT_PATH = _dest

RESOLUTIONS = {
    "720x1280":  ("720",  "1280"),
    "1080x1080": ("1080", "1080"),
    "1280x720":  ("1280", "720"),
}

# ─────────────────────────────────────────────
#  STATUS
# ─────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "groq":      bool(GROQ_API_KEY),
        "font_name": FONT_NAME,
        "font_ok":   bool(FONT_PATH),
    })

# ─────────────────────────────────────────────
#  PÁGINAS
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8").read()

@app.route("/manifest.json")
def manifest():
    return send_file(os.path.join(BASE_DIR, "manifest.json"), mimetype="application/manifest+json")

@app.route("/service-worker.js")
def service_worker():
    resp = send_file(os.path.join(BASE_DIR, "service-worker.js"), mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_file(os.path.join(BASE_DIR, "static", filename))

# ─────────────────────────────────────────────
#  TRANSCRIÇÃO
# ─────────────────────────────────────────────
def _groq_transcrever(audio_bytes, filename):
    resp = http_requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": (filename, audio_bytes, "audio/mpeg")},
        data={
            "model":                     "whisper-large-v3-turbo",
            "language":                  "pt",
            "response_format":           "verbose_json",
            "timestamp_granularities[]": "word",
        },
        timeout=300
    )
    if resp.status_code != 200:
        raise Exception(f"Groq {resp.status_code}: {resp.text}")

    data     = resp.json()
    texto    = data.get("text", "").strip()
    segs     = [
        {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": s.get("text", "").strip()}
        for s in (data.get("segments") or [])
    ]
    palavras = [
        {"word": w.get("word", "").strip(), "start": float(w.get("start", 0)), "end": float(w.get("end", 0))}
        for w in (data.get("words") or [])
        if w.get("word", "").strip()
    ]
    return texto, segs, palavras

@app.route("/transcrever", methods=["POST"])
def transcrever():
    if not GROQ_API_KEY:
        return jsonify({"erro": "GROQ_API_KEY não configurada."}), 400
    aud = request.files.get("audio")
    if not aud:
        return jsonify({"erro": "Nenhum áudio enviado."}), 400
    try:
        texto, segs, palavras = _groq_transcrever(aud.read(), aud.filename or "audio.mp3")
        return jsonify({"texto": texto, "segmentos": segs, "palavras": palavras})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# ─────────────────────────────────────────────
#  GERADOR DE PROMPT DE IMAGEM
# ─────────────────────────────────────────────
@app.route("/gerar-prompt", methods=["POST"])
def gerar_prompt():
    if not GROQ_API_KEY:
        return jsonify({"erro": "GROQ_API_KEY não configurada."}), 400
    data = request.json
    transcricao = (data or {}).get("texto", "").strip()
    if not transcricao:
        return jsonify({"erro": "Texto vazio."}), 400
    try:
        resp = http_requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at creating image generation prompts. "
                            "Given a text in Portuguese, create a detailed and creative prompt in English "
                            "to generate a YouTube/TikTok video cover image. "
                            "The prompt should be vivid, cinematic, and visually striking. "
                            "Reply ONLY with the prompt, no explanations, no quotes, no extra text."
                        )
                    },
                    {"role": "user", "content": transcricao}
                ],
                "max_tokens": 300,
                "temperature": 0.85
            },
            timeout=30
        )
        if resp.status_code != 200:
            return jsonify({"erro": f"Groq: {resp.text}"}), 500
        prompt = resp.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"prompt": prompt})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

# ─────────────────────────────────────────────
#  GERADOR ASS
# ─────────────────────────────────────────────
def _ts_ass(s: float) -> str:
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    cs = int(round((s - int(s)) * 100))
    return f"{h}:{m:02d}:{sc:02d}.{cs:02d}"

PALAVRAS_POR_GRUPO = 4

def gerar_ass(dados, w, h, modo_dados="segmentos"):
    font_size = int(w * 0.074)
    margin_v  = int(h * 0.08)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Tron,{FONT_NAME},{font_size},&H00FFFFFF,&H88AAAAAA,&H00000000,&HAA000000,-1,0,0,0,100,100,0,0,3,6,2,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if modo_dados == "palavras" and dados:
        grupos = [dados[i:i+PALAVRAS_POR_GRUPO] for i in range(0, len(dados), PALAVRAS_POR_GRUPO)]
        for grupo in grupos:
            if not grupo: continue
            start = grupo[0]["start"]
            end   = grupo[-1]["end"]
            if end <= start: end = start + 0.5
            partes = []
            for p in grupo:
                dur_cs = max(1, int(round((p["end"] - p["start"]) * 100)))
                txt_w  = p["word"].strip().replace("{","").replace("}","").replace("\\","")
                partes.append(f"{{\\k{dur_cs}}}{txt_w}")
            lines.append(f"Dialogue: 0,{_ts_ass(start)},{_ts_ass(end)},Tron,,0,0,0,,{' '.join(partes)}")
    else:
        for seg in dados:
            txt = seg.get("text", "").strip()
            if not txt: continue
            if len(txt) > 35:
                txt = textwrap.fill(txt, width=35, max_lines=2, placeholder="...").replace("\n", "\\N")
            txt = txt.replace("{","").replace("}","")
            lines.append(f"Dialogue: 0,{_ts_ass(seg['start'])},{_ts_ass(seg['end'])},Tron,,0,0,0,,{txt}")
    return header + "\n".join(lines)

def _esc(txt):
    return txt.replace("\\","\\\\").replace("'","\\'").replace(":","\\:").replace("[","\\[").replace("]","\\]").replace(",","\\,")

def build_vf_estatico(w, h, legenda):
    scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    if not legenda.strip(): return scale
    txt  = _esc(legenda.strip())
    font = f"fontfile={FONT_PATH}" if FONT_PATH else f"font={FONT_NAME}"
    fs   = int(int(w) * 0.074)
    mb   = int(int(h) * 0.06)
    dt   = f"drawtext={font}:text='{txt}':fontcolor=white:fontsize={fs}:bordercolor=black:borderw=6:shadowcolor=black@0.65:shadowx=2:shadowy=3:box=1:boxcolor=black@0.40:boxborderw=14:x=(w-text_w)/2:y=h-text_h-{mb}"
    return f"{scale},{dt}"

# ─────────────────────────────────────────────
#  CONVERSOR — motor turbinado para arquivos grandes
# ─────────────────────────────────────────────
@app.route("/converter", methods=["POST"])
def converter():
    img_file      = request.files.get("imagem")
    aud_file      = request.files.get("audio")
    resolucao     = request.form.get("resolucao",    "1080x1080")
    legenda_txt   = request.form.get("legenda",      "").strip()
    modo_leg      = request.form.get("modo_legenda", "nenhuma")
    palavras_json = request.form.get("palavras",     "")
    segs_json     = request.form.get("segmentos",    "")

    if not img_file or not aud_file:
        return "Imagem e áudio são obrigatórios.", 400

    w_str, h_str = RESOLUTIONS.get(resolucao, ("1080", "1080"))
    w, h = int(w_str), int(h_str)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            img_ext  = os.path.splitext(img_file.filename)[1] or ".jpg"
            aud_ext  = os.path.splitext(aud_file.filename)[1] or ".mp3"
            img_path = os.path.join(tmp, "img" + img_ext)
            aud_path = os.path.join(tmp, "aud" + aud_ext)
            img_file.save(img_path)
            aud_file.save(aud_path)

            fd, out_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)

            scale_vf = (
                f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"format=yuv420p"
            )
            vf       = scale_vf
            ass_path = None

            # ── Legenda automática ──
            if modo_leg == "auto":
                dados_ass  = None
                modo_dados = "segmentos"
                if palavras_json:
                    try:
                        pws = json.loads(palavras_json)
                        if pws: dados_ass = pws; modo_dados = "palavras"
                    except Exception: pass
                if dados_ass is None and segs_json:
                    try: dados_ass = json.loads(segs_json)
                    except Exception: pass
                if dados_ass:
                    try:
                        ass_content = gerar_ass(dados_ass, w, h, modo_dados)
                        ass_path    = os.path.join(tmp, "legenda.ass")
                        with open(ass_path, "w", encoding="utf-8") as f:
                            f.write(ass_content)
                        fontsdir = f":fontsdir={FONTS_DIR}" if os.path.isdir(FONTS_DIR) else ""
                        vf = f"{scale_vf},ass={ass_path}{fontsdir}"
                    except Exception as err:
                        app.logger.error(f"Erro ASS: {err}")
                        ass_path = None

            if ass_path is None and modo_leg == "estatica" and legenda_txt:
                vf = build_vf_estatico(w_str, h_str, legenda_txt)

            # ── Detectar duração do áudio ──
            probe = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", aud_path
            ], capture_output=True, text=True)
            duracao = 0
            try:
                pinfo   = json.loads(probe.stdout)
                duracao = float(pinfo.get("format", {}).get("duration", 0))
            except Exception:
                pass

            # ── Detectar se áudio precisa de conversão ──
            probe2  = subprocess.run([
                "ffprobe", "-v", "quiet",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-print_format", "json",
                aud_path
            ], capture_output=True, text=True)
            codec_aud = "aac"
            try:
                cinfo = json.loads(probe2.stdout)
                codec_aud = cinfo["streams"][0]["codec_name"]
            except Exception:
                pass

            # Se o áudio já é AAC e não há legenda, copia direto (muito mais rápido)
            audio_copy = (codec_aud == "aac" and modo_leg == "nenhuma" and not legenda_txt)

            cmd = [
                "ffmpeg", "-y",
                "-probesize",     "100M",       # lê mais do arquivo para detectar streams
                "-analyzeduration","100M",       # analisa mais para arquivos grandes
                "-loop",          "1",
                "-framerate",     "25",
                "-i",             img_path,
                "-i",             aud_path,
                "-vf",            vf,
                "-map",           "0:v",
                "-map",           "1:a",
                # vídeo
                "-c:v",           "libx264",
                "-preset",        "ultrafast",
                "-tune",          "stillimage",
                "-crf",           "26",
                "-r",             "25",
                "-g",             "50",           # keyframe a cada 2s
                "-pix_fmt",       "yuv420p",
                "-threads",       CPU_CORES,
                # áudio
                "-c:a",           "copy" if audio_copy else "aac",
                "-b:a",           "128k",
                "-ar",            "44100",
                "-ac",            "2",
                # saída
                "-shortest",
                "-movflags",      "+faststart",
                # buffers para arquivos grandes
                "-bufsize",       "8M",
                "-maxrate",       "4M",
                out_path,
            ]
            if not audio_copy:
                # insere b:a depois de c:a na posição certa (já está na lista)
                pass

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode != 0:
            app.logger.error(f"FFmpeg stderr: {result.stderr[-3000:]}")
            return f"Erro FFmpeg:\n{result.stderr[-2000:]}", 500

        @after_this_request
        def _cleanup(response):
            try: os.unlink(out_path)
            except Exception: pass
            return response

        return send_file(
            out_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="tron_clipe.mp4"
        )

    except subprocess.TimeoutExpired:
        return "Tempo limite excedido (60 min).", 504
    except Exception:
        return f"Erro interno:\n{traceback.format_exc()}", 500

# ─────────────────────────────────────────────
@app.route("/healthz")
def healthz():
    return "OK", 200

@app.errorhandler(Exception)
def handle_exception(e):
    return f"<pre>{traceback.format_exc()}</pre>", 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
