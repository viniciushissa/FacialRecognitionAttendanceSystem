# 🎓 Facial Recognition Attendance System

Sistema de registro de presença automático baseado em reconhecimento facial, desenvolvido em Python. Utiliza **YOLOv8** para detecção de rostos e **SFace** para cadastro e reconhecimento, com persistência em **PostgreSQL** e interface de terminal colorida.

---

## ✨ Funcionalidades

- 📷 **Registro automático via webcam** — cadastra, detecta e reconhece rostos em tempo real, gravando a presença com data e horário exato
- 📅 **Consulta por dia** — visualize todos os alunos presentes em qualquer aula já registrada
- 📊 **Resumo de frequência** — relatório completo com total de presenças e barra de progresso por aluno
- 🛡️ **Anti-duplicata** — cada aluno é registrado apenas uma vez por dia, independentemente de quantas vezes aparecer na câmera
- 🎨 **Interface de terminal colorida** — menu interativo com cores ANSI

---

## 🗂️ Estrutura do Projeto

```
FacialRecognitionAttendanceSystem/
├── attendance_system/
│   ├── faces_db/
│   │    ├── nome_do_aluno_1/     # Uma pasta por aluno com fotos de referência
│   │    │   │   ├── foto1.jpg
│   │    │   │   └── foto2.jpg
│   │    │   └── nome_do_aluno_2/
│   ├── presenca.py          # Ponto de entrada e interface de terminal
│   ├── utils.py             # Cores do terminal
│   └── postgres.py          # Camada de banco de dados
├── docker-compose.yaml      # PostgreSQL via Docker
├── requirementes.txt        # Dependências Python
└── .gitignore
```

> Os modelos `yolov8n-face.pt` e `face_recognition_sface_2021dec.onnx` devem estar na mesma pasta do script.

---

## 🗄️ Banco de Dados

Duas tabelas são criadas automaticamente na primeira execução:

```sql
students   -- id, name, registration (único), registered_at
attendance -- id, student_id, session_date, attended_at
```

A constraint `UNIQUE(student_id, session_date)` garante que uma presença não seja duplicada.

---

## 🚀 Como Usar

### 1. Clone o repositório

```bash
git clone https://github.com/viniciushissa/FacialRecognitionAttendanceSystem.git
cd FacialRecognitionAttendanceSystem
```

### 2. Instale as dependências

```bash
pip install -r requirementes.txt
```

### 3. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
POSTGRES_DB=attendance_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=sua_senha
POSTGRES_PORT=5432
```

### 4. Suba o banco de dados com Docker

```bash
docker compose up -d
```

### 5. Adicione os alunos ao banco de faces ou Cadastre novos alunos

Se quiser adicionar manualmente, crie uma pasta dentro de `faces_db/` com o nome do aluno e coloque algumas fotos dele:

```
faces_db/
└── João Silva/
    ├── foto1.jpg
    └── foto2.jpg
```

### 6. Execute o sistema

```bash
cd attendance_system
python presenca.py
```

---

## 📋 Menu Principal

```
╔══════════════════════════════════════════════════════╗
║              SISTEMA DE PRESENÇAS                    ║
║                  Menu Principal                      ║
╠══════════════════════════════════════════════════════╣
║  [1]  📝  Registrar presenças de hoje (câmera)       ║
║  [2]  📅  Ver presenças por dia                      ║
║  [3]  📊  Resumo geral de frequência                 ║
║  [4]  📷  Cadastrar novo aluno                       ║
║  [0]  ❌  Sair                                       ║
╚══════════════════════════════════════════════════════╝
```

---

## 🛠️ Tecnologias

| Tecnologia | Uso |
|---|---|
| Python 3.12+ | Linguagem principal |
| YOLOv8 (Ultralytics) | Detecção de rostos |
| SFace (OpenCV) | Extração de embeddings e reconhecimento |
| PostgreSQL 17 | Persistência dos dados |
| psycopg2 | Driver Python para PostgreSQL |
| Docker / Docker Compose | Banco de dados em container |
| python-dotenv | Gerenciamento de variáveis de ambiente |

---

## ⚙️ Requisitos

- Python 3.12+
- Docker e Docker Compose
- Webcam
- GPU com CUDA (opcional, melhora a performance)
- Modelos pré-treinados:
  - [`yolov8n-face.pt`](https://github.com/akanametov/yolo-face)
  - [`face_recognition_sface_2021dec.onnx`](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
