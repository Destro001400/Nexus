import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Cria o diretório 'templates' se não existir
if not os.path.exists("chatbot/templates"):
    os.makedirs("chatbot/templates")

# Cria o diretório 'static' se não existir
if not os.path.exists("chatbot/static"):
    os.makedirs("chatbot/static")

app = FastAPI(
    title="Nexus Chatbot",
    description="Interface de chat para o LLM Nexus.",
    version="0.1.0"
)

# Monta o diretório 'static' para servir arquivos como CSS e JS
app.mount("/static", StaticFiles(directory="chatbot/static"), name="static")

# Carrega o conteúdo do arquivo HTML
html_path = "chatbot/templates/index.html"

# Verifica se o arquivo HTML existe antes de ler
if not os.path.exists(html_path):
    # Cria um arquivo HTML placeholder se não existir
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("""<!DOCTYPE html>
<html>
    <head>
        <title>Nexus Chat</title>
        <link href="/static/styles.css" rel="stylesheet">
    </head>
    <body>
        <h1>Nexus Chat</h1>
        <div id="chat-box"></div>
        <input type="text" id="messageText" autocomplete="off"/>
        <button onclick="sendMessage()">Send</button>
        <script src="/static/scripts.js"></script>
    </body>
</html>""")

with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Lógica de mock para a resposta do chatbot
            response = f"Nexus responde: {data}"
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("Cliente desconectado")

# Para executar o chatbot localmente:
# uvicorn chatbot.main:app --reload
