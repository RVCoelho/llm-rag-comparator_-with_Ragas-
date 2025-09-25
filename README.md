# RAG FII - Sistema RAG Completo

Sistema RAG (Retrieval-Augmented Generation) completo com avaliação RAGAS para análise de Fundos Imobiliários.

## 🚀 Funcionalidades

- **3 Endpoints principais:**
  - `/llm` - LLM puro (pode alucinar)
  - `/rag` - RAG com citações/tags
  - `/evaluate` - RAG + avaliação RAGAS

- **Recursos:**
  - Processamento automático de PDFs
  - Sistema de citações com numeração
  - Avaliação de qualidade com RAGAS
  - Logging detalhado
  - Arquitetura limpa (Clean Architecture)

## 📁 Estrutura do Projeto

```
tcc/
├── assets/                    # Pasta para PDFs
├── controllers/               # Controladores da API
├── domain/services/          # Serviços de domínio
├── data/repositories/local/  # Repositórios de dados
├── main.py                   # Aplicação principal
├── requirements.txt          # Dependências
├── .env                      # Variáveis de ambiente
└── README.md                 # Este arquivo
```

## 🛠️ Instalação

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Configurar API Key:**
   - O arquivo `.env` já está configurado com a API key
   - Se necessário, edite o arquivo `.env` com sua própria chave

3. **Adicionar PDFs:**
   - Coloque seus PDFs na pasta `assets/`
   - O sistema processará automaticamente na primeira execução

## 🚀 Execução

```bash
python main.py
```

O servidor Flask será iniciado em `http://localhost:5000`

## 📡 Endpoints da API

### 1. LLM Puro
```bash
POST /llm
Content-Type: application/json

{
  "question": "Sua pergunta aqui"
}
```

### 2. RAG com Citações
```bash
POST /rag
Content-Type: application/json

{
  "question": "Sua pergunta aqui"
}
```

### 3. RAG + Avaliação RAGAS
```bash
POST /evaluate
Content-Type: application/json

{
  "question": "Sua pergunta aqui"
}
```

### 4. Health Check
```bash
GET /health
```

## 📊 Exemplo de Uso

### Teste com curl:

```bash
# LLM puro
curl -X POST http://localhost:5000/llm \
  -H "Content-Type: application/json" \
  -d '{"question": "O que são fundos imobiliários?"}'

# RAG com citações
curl -X POST http://localhost:5000/rag \
  -H "Content-Type: application/json" \
  -d '{"question": "Quais são os principais tipos de FIIs?"}'

# RAG + avaliação
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"question": "Como funciona a distribuição de dividendos em FIIs?"}'
```

## 🔧 Configurações

- **PDFs Directory:** `assets/`
- **FAISS Index:** `pdf_faiss_index/`
- **Chunk Size:** 1000 caracteres
- **Chunk Overlap:** 200 caracteres
- **Log File:** `rag_fii.log`

## 📝 Logs

O sistema gera logs detalhados em:
- Console (tempo real)
- Arquivo `rag_fii.log`

## 🏗️ Arquitetura

- **Clean Architecture** com separação clara de responsabilidades
- **Domain Services** para lógica de negócio
- **Repositories** para acesso a dados
- **Controllers** para endpoints da API

## ⚠️ Notas Importantes

1. **Primeira execução:** O sistema criará o índice FAISS automaticamente
2. **PDFs:** Coloque apenas PDFs na pasta `assets/`
3. **API Key:** Mantenha sua chave OpenAI segura
4. **Performance:** A primeira consulta pode ser mais lenta devido ao carregamento do índice

## 🐛 Troubleshooting

- **Erro de importação:** Verifique se todas as dependências estão instaladas
- **PDFs não processados:** Verifique se os PDFs estão na pasta `assets/`
- **Erro de API:** Verifique se a chave OpenAI está correta no `.env`
