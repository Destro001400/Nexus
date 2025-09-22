from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Nexus",
    description="API e Chatbot para o Large Language Model Nexus.",
    version="0.1.0",
)

# Monta o diretório 'static' para servir arquivos como CSS e JS
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# Carrega o conteúdo do arquivo HTML
html_path = "api/templates/index.html"
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Modelo de dados para a requisição da API
class Query(BaseModel):
    prompt: str
    mode: str = "geral"  # Modos: geral, pesquisa, criativo, desenvolvimento
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

# Modelo de dados para a resposta da API
class Response(BaseModel):
    text: str
    mode: str
    tokens_used: int

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

@app.post("/generate", response_model=Response, summary="Geração de Texto", description="Gera texto com base em um prompt.")
async def generate_text(query: Query):
    """
    Recebe um prompt e retorna a resposta gerada pelo modelo Nexus.

    - **prompt**: O texto de entrada para o modelo.
    - **mode**: O modo de operação (geral, pesquisa, criativo, desenvolvimento).
    - **max_tokens**: O número máximo de tokens a serem gerados.
    - **temperature**: A criatividade da resposta (0.0 a 1.0).
    """
    # Validação do modo
    valid_modes = ["geral", "pesquisa", "criativo", "desenvolvimento"]
    if query.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Modo inválido. Use um dos seguintes: {valid_modes}")

    # --- Lógica de Mock (simulação) ---
    # Aqui entraria a chamada real para o modelo Nexus.
    # Por enquanto, vamos simular uma resposta.
    simulated_response = f"Resposta simulada para o prompt '{query.prompt}' no modo '{query.mode}'."
    tokens_used = len(simulated_response.split())
    # --- Fim da Lógica de Mock ---

    return Response(
        text=simulated_response,
        mode=query.mode,
        tokens_used=tokens_used
    )

# Para executar a API localmente:
# uvicorn api.main:app --reload
