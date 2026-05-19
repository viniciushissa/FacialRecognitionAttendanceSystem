import os
import cv2
import numpy as np
from postgres import record_attendance, get_all_session_dates, get_attendance_by_date, get_attendance_summary, init_db
from datetime import date
from ultralytics import YOLO
from utils import AnsiColors

yolo_model = YOLO("yolov8n-face.pt")

recognizer = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", ""
)


def yolo_detect_faces(image, conf=0.5):
    results = yolo_model(image, conf=conf, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            detections.append((x1, y1, x2, y2, score))
    return detections


def build_face_box_for_sface(x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    pts = [
        float(x1), float(y1), float(w), float(h),
        float(x1 + int(w * 0.30)), float(y1 + int(h * 0.38)),
        float(x1 + int(w * 0.70)), float(y1 + int(h * 0.38)),
        float(x1 + int(w * 0.50)), float(y1 + int(h * 0.58)),
        float(x1 + int(w * 0.35)), float(y1 + int(h * 0.75)),
        float(x1 + int(w * 0.65)), float(y1 + int(h * 0.75)),
    ]
    return np.array([pts], dtype=np.float32)


def extract_embedding(image, bbox):
    x1, y1, x2, y2, _ = bbox
    face_box = build_face_box_for_sface(x1, y1, x2, y2)
    aligned = recognizer.alignCrop(image, face_box)
    return recognizer.feature(aligned)


def recognize_face(face_feature, database, cosine_threshold=0.363):
    best_name, best_score = "Desconhecido", -1.0
    for name, features in database.items():
        for ref_feat in features:
            score = recognizer.match(
                face_feature, ref_feat, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score = score
                best_name = name
    if best_score < cosine_threshold:
        return "Desconhecido", best_score
    return best_name, best_score


def build_database(db_path="faces_db"):
    database = {}
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        database[person_name] = []
        for file_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, file_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            detections = yolo_detect_faces(img, conf=0.4)
            if not detections:
                continue
            detections.sort(key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
            feat = extract_embedding(img, detections[0])
            database[person_name].append(feat)
    return database


LINE = f"{AnsiColors.DIM}{'─' * 58}{AnsiColors.RESET}"

def header(title: str):
    print()
    print(f"{AnsiColors.BOLD}{AnsiColors.BG_BLUE}  {'SISTEMA DE PRESENÇAS':^54}  {AnsiColors.RESET}")
    print(f"{AnsiColors.BOLD}{AnsiColors.CYAN}  {title:^54}  {AnsiColors.RESET}")
    print(LINE)

def pause():
    input(f"\n{AnsiColors.DIM}  [Enter para voltar ao menu]{AnsiColors.RESET} ")


def run_attendance_session(database: dict):
    today = date.today()
    header(f"Registrando presenças — {today.strftime('%d/%m/%Y')}")
    print(f"  {AnsiColors.YELLOW}Pressione [ESC] para encerrar a sessão.{AnsiColors.RESET}")
    print(LINE)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cv2.namedWindow("Reconhecimento Facial — ESC para sair", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reconhecimento Facial — ESC para sair", 900, 700)
    if not cap.isOpened():
        print(f"  {AnsiColors.RED}✖ Não foi possível abrir a webcam.{AnsiColors.RESET}")
        pause()
        return

    registered_today: set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_detect_faces(frame, conf=0.5)

        for det in detections:
            x1, y1, x2, y2, conf = det
            try:
                feat = extract_embedding(frame, det)
                name, score = recognize_face(feat, database)
            except Exception:
                name, score = "Erro", 0.0

            if name != "Desconhecido" and name not in registered_today:
                new, ts = record_attendance(name, today)
                registered_today.add(name)
                if new:
                    ts_str = ts.strftime("%H:%M:%S")
                    print(f"  {AnsiColors.GREEN}✔ Presença registrada:{AnsiColors.RESET} "
                          f"{AnsiColors.BOLD}{name}{AnsiColors.RESET} às {ts_str}")

            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{name} ({score:.2f})",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2,
            )

        cv2.imshow("Reconhecimento Facial — ESC para sair", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(LINE)
    print(f"  {AnsiColors.CYAN}Sessão encerrada. "
          f"{AnsiColors.BOLD}{len(registered_today)}{AnsiColors.RESET}{AnsiColors.CYAN} "
          f"aluno(s) registrado(s) hoje.{AnsiColors.RESET}")
    pause()

def view_attendance_by_day():
    dates = get_all_session_dates()

    if not dates:
        header("Consultar Presenças por Dia")
        print(f"  {AnsiColors.YELLOW}Nenhuma presença registrada ainda.{AnsiColors.RESET}")
        pause()
        return

    while True:
        header("Consultar Presenças por Dia")
        print(f"  {AnsiColors.WHITE}Dias com presença registrada:{AnsiColors.RESET}\n")
        for i, d in enumerate(dates, 1):
            day_names = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}
            wd = day_names[d.weekday()]
            print(f"    {AnsiColors.CYAN}[{i:>2}]{AnsiColors.RESET}  {d.strftime('%d/%m/%Y')} "
                  f"{AnsiColors.DIM}({wd}){AnsiColors.RESET}")
        print()
        print(f"    {AnsiColors.DIM}[0]{AnsiColors.RESET}  Voltar")
        print(LINE)

        choice = input(f"  Escolha um dia: ").strip()
        if choice == "0":
            return

        if not choice.isdigit() or not (1 <= int(choice) <= len(dates)):
            print(f"  {AnsiColors.RED}Opção inválida.{AnsiColors.RESET}")
            continue

        selected = dates[int(choice) - 1]
        records = get_attendance_by_date(selected)

        header(f"Presenças em {selected.strftime('%d/%m/%Y')}")
        if not records:
            print(f"  {AnsiColors.YELLOW}Nenhum aluno registrado nesse dia.{AnsiColors.RESET}")
        else:
            print(f"  {'#':<4}  {'Nome':<30}  Horário")
            print(f"  {'─'*4}  {'─'*30}  {'─'*8}")
            for i, r in enumerate(records, 1):
                ts = r["attended_at"].strftime("%H:%M:%S")
                print(f"  {AnsiColors.DIM}{i:<4}{AnsiColors.RESET}  "
                      f"{AnsiColors.BOLD}{r['name']:<30}{AnsiColors.RESET}  {AnsiColors.GREEN}{ts}{AnsiColors.RESET}")
            print(LINE)
            print(f"  Total: {AnsiColors.BOLD}{len(records)}{AnsiColors.RESET} aluno(s)")
        pause()
        return


def view_summary():
    header("Resumo Geral de Frequência")
    rows = get_attendance_summary()
    total_sessions = len(get_all_session_dates())

    if not rows:
        print(f"  {AnsiColors.YELLOW}Nenhum dado disponível.{AnsiColors.RESET}")
    else:
        print(f"  Total de aulas realizadas: {AnsiColors.BOLD}{total_sessions}{AnsiColors.RESET}\n")
        print(f"  {'Nome':<30}  {'Aulas':>5}  {'Última vez':<12}  Frequência")
        print(f"  {'─'*30}  {'─'*5}  {'─'*12}  {'─'*26}")
        for r in rows:
            last = r["last_seen"].strftime("%d/%m/%Y") if r["last_seen"] else "—"
            pct  = (r["total_days"] / total_sessions * 100) if total_sessions else 0
            bar_len = int(pct / 5)
            bar = f"{AnsiColors.GREEN}{'█' * bar_len}{AnsiColors.DIM}{'░' * (20 - bar_len)}{AnsiColors.RESET}"
            print(f"  {AnsiColors.BOLD}{r['name']:<30}{AnsiColors.RESET}  "
                  f"{r['total_days']:>5}  {last:<12}  {bar} {pct:5.1f}%")
    pause()


def main_menu(database: dict):
    while True:
        header("Menu Principal")
        print(f"  {AnsiColors.CYAN}[1]{AnsiColors.RESET}  📝  Registrar presenças de hoje (câmera)")
        print(f"  {AnsiColors.CYAN}[2]{AnsiColors.RESET}  📅  Ver presenças por dia")
        print(f"  {AnsiColors.CYAN}[3]{AnsiColors.RESET}  📊  Resumo geral de frequência")
        print(f"  {AnsiColors.CYAN}[4]{AnsiColors.RESET}  📸  Cadastrar novo aluno")
        print(f"  {AnsiColors.CYAN}[0]{AnsiColors.RESET}  ❌  Sair")
        print(LINE)

        choice = input(f"  {AnsiColors.BOLD}Escolha uma opção:{AnsiColors.RESET} ").strip()

        if choice == "1":
            run_attendance_session(database)
        elif choice == "2":
            view_attendance_by_day()
        elif choice == "3":
            view_summary()
        elif choice == "4":
            database = registernewface(database)
        elif choice == "0":
            print(f"\n  {AnsiColors.DIM}Até logo!{AnsiColors.RESET}\n")
            break
        else:
            print(f"  {AnsiColors.RED}Opção inválida. Tente novamente.{AnsiColors.RESET}")

def registernewface(database: dict, samples=8):
    name = input("Nome do aluno: ").strip()
    if not name:
        print("Nome inválido.")
        return database

    person_dir = os.path.join("faces_db", name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Não foi possível abrir a webcam.")
        return database

    saved = 0
    frame_count = 0

    print("Olhe para a câmera. Pressione ESC para cancelar.")

    while saved < samples:
        ret, frame = cap.read()
        if not ret:
            break

        detections = yolo_detect_faces(frame, conf=0.5)

        if detections:
            detections.sort(key=lambda d: (d[2]-d[0]) * (d[3]-d[1]), reverse=True)
            x1, y1, x2, y2, conf = detections[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Amostras: {saved}/{samples}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            frame_count += 1
            if frame_count % 10 == 0:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    img_path = os.path.join(person_dir, f"{saved+1}.jpg")
                    cv2.imwrite(img_path, face)
                    saved += 1

        cv2.imshow("Cadastro facial", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if saved == 0:
        print("Nenhuma face cadastrada.")
        return database

    database[name] = []
    for file_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        detections = yolo_detect_faces(img, conf=0.4)
        if not detections:
            continue

        detections.sort(key=lambda d: (d[2]-d[0]) * (d[3]-d[1]), reverse=True)
        feat = extract_embedding(img, detections[0])
        database[name].append(feat)

    from postgres import upsert_student
    upsert_student(name)

    print(f"{name} cadastrado com {len(database[name])} embeddings.")
    return database


if __name__ == "__main__":
    print(f"\n{AnsiColors.BOLD}{AnsiColors.CYAN}  Inicializando...{AnsiColors.RESET}")

    init_db()

    print(f"  Carregando banco de faces em ./faces_db ...")
    os.makedirs("faces_db", exist_ok=True)
    database = build_database("faces_db")
    names = list(database.keys())
    print(f"  {AnsiColors.GREEN}✔ {len(names)} aluno(s) carregado(s):{AnsiColors.RESET} "
          f"{', '.join(names) if names else '(nenhum)'}")

    main_menu(database)