import os
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from groq import Groq
from datetime import datetime
from flask import flash


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# -------------------
# App Config
# -------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///C:/ai_edu_platform/instance/database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads", "pdfs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# RAG globals
model = SentenceTransformer('all-MiniLM-L6-v2')
pdf_chunks = []
faiss_index = None
active_pdf = None  # currently selected pdf name

# -------------------
# Models
# -------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20))  # admin or student

class Material(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(100))
    title = db.Column(db.String(200))
    filename = db.Column(db.String(200), nullable=True)
    type = db.Column(db.String(50))  # pdf, note, video
    content = db.Column(db.Text, nullable=True)  # note text / video link


class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(100))
    unit = db.Column(db.String(20)) 
    question = db.Column(db.String(300))
    option_a = db.Column(db.String(200))
    option_b = db.Column(db.String(200))
    option_c = db.Column(db.String(200))
    option_d = db.Column(db.String(200))
    correct = db.Column(db.String(1))  # A/B/C/D


class QuizResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    subject = db.Column(db.String(100))
    score = db.Column(db.Integer)
    total = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    action = db.Column(db.String(50))       # e.g. view_pdf, ai_tutor, quiz_start
    details = db.Column(db.String(200))     # short description
    ref_id = db.Column(db.Integer, nullable=True)  # material id, quiz result id, etc.
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    message = db.Column(db.String(300))
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# -------------------
# Helpers
# -------------------

def extract_text_from_pdf(file_path):
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_vector_db(chunks):
    if not chunks:
        return None
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def search_pdf(query):
    global pdf_chunks, faiss_index
    if not faiss_index or not pdf_chunks:
        return "No PDF is indexed."
    q_embed = model.encode([query])
    D, I = faiss_index.search(q_embed, k=3)
    results = [pdf_chunks[i] for i in I[0]]
    return "\n\n---\n\n".join(results)

def current_user():
    if "user_id" in session:
        return User.query.get(session["user_id"])
    return None

def login_required(role=None):
    def decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = current_user()
            if not user:
                return redirect(url_for("login"))
            if role and user.role != role:
                return "Access Denied"
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_activity(action, details="", ref_id=None):
    try:
        user = current_user()
        if not user:
            return
        log = ActivityLog(
            user_id=user.id,
            action=action,
            details=details[:180],
            ref_id=ref_id
        )
        db.session.add(log)
        db.session.commit()
    except Exception:
        # avoid breaking main flow due to logging error
        db.session.rollback()

def create_notification(user_id, message):
    n = Notification(user_id=user_id, message=message)
    db.session.add(n)
    db.session.commit()

def notify_admins(message):
    admins = User.query.filter_by(role="admin").all()
    for a in admins:
        create_notification(a.id, message)

def calculate_performance(uid):
    # QUIZ PERFORMANCE
    quiz_results = QuizResult.query.filter_by(user_id=uid).all()
    if quiz_results:
        quiz_percentage = sum((q.score / q.total) * 100 for q in quiz_results) / len(quiz_results)
    else:
        quiz_percentage = 0

    # MATERIAL USAGE
    activity_count = ActivityLog.query.filter_by(user_id=uid, action="view_material").count()
    material_usage_score = min(activity_count * 5, 100)

    # ACTIVITY FREQUENCY
    login_count = ActivityLog.query.filter_by(user_id=uid, action="login").count()
    activity_score = min(login_count * 10, 100)

    # AI USAGE
    ai_usage = ActivityLog.query.filter_by(user_id=uid, action="ai_tutor").count()
    ai_score = min(ai_usage * 5, 100)

    # COMPLETION RATE
    total_materials = Material.query.count()
    viewed_materials = ActivityLog.query.filter_by(user_id=uid, action="view_material").count()
    completion_rate = (viewed_materials / total_materials) * 100 if total_materials > 0 else 0

    final_score = (
        quiz_percentage * 0.40 +
        material_usage_score * 0.25 +
        activity_score * 0.15 +
        ai_score * 0.10 +
        completion_rate * 0.10
    )
    return round(final_score, 2)
def detect_language(text):
    try:
        code = detect(text)
    except:
        return "English"

    lang_map = {
        "en": "English",
        "ta": "Tamil",
        "hi": "Hindi",
        "kn": "Kannada",
        "te": "Telugu"
    }

    return lang_map.get(code, "English")
      

# -------------------
# Routes: Auth & Home
# -------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]

        if User.query.filter_by(username=username).first():
            return render_template("register.html", msg="Username already exists")

        hashed = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(username=username, password=hashed, role=role)
        db.session.add(user)
        db.session.commit()

        if role == "student":
           notify_admins(f"New student registered: {username}")
           # New student gets notifications for all past materials
           materials = Material.query.all()
           for m in materials:
              create_notification(user.id, f"Material available: {m.title}")

    # New student gets notifications for all past quizzes
           quiz_subjects = Quiz.query.with_entities(Quiz.subject).distinct().all()
           for qs in quiz_subjects:
               create_notification(user.id, f"Quiz available in {qs[0]}")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session["user_id"] = user.id
            log_activity("login", f"Logged in as {user.username}")
            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("student_dashboard"))
        return render_template("login.html", msg="Invalid credentials")
    return render_template("login.html")



@app.route("/logout")
def logout():
    log_activity("logout", "User logged out")
    session.clear()
    return redirect(url_for("home"))


# -------------------
# Dashboards
# -------------------
@app.route("/admin/dashboard")
@login_required(role="admin")
def admin_dashboard():
    materials = Material.query.order_by(Material.subject).all()
    total_students = User.query.filter_by(role="student").count()
    total_quizzes = db.session.query(Quiz.subject).distinct().count()
    results = QuizResult.query.all()
    return render_template("admin_dashboard.html",
                           materials=materials,
                           total_students=total_students,
                           total_quizzes=total_quizzes,
                           results=results)

@app.route("/admin/analytics")
@login_required(role="admin")
def admin_analytics():
    total_students = User.query.filter_by(role="student").count()
    total_materials = Material.query.count()
    total_quizzes = db.session.query(Quiz.subject).distinct().count()

    results = QuizResult.query.all()

    subject_stats = {}

    for r in results:
        subject = r.subject.split(" - ")[0].strip()

        if subject not in subject_stats:
            subject_stats[subject] = {"score": 0, "total": 0, "attempts": 0}

        subject_stats[subject]["score"] += r.score
        subject_stats[subject]["total"] += r.total
        subject_stats[subject]["attempts"] += 1

    for subject, stats in subject_stats.items():
        if stats["total"] == 0:
            stats["average"] = 0
        else:
            stats["average"] = round((stats["score"] / stats["total"]) * 100, 2)

    # STUDENT PERFORMANCE SCORE
    students = User.query.filter_by(role="student").all()
    student_scores = {}

    for s in students:
        student_scores[s.username] = calculate_performance(s.id)

    # STUDENT QUIZ ATTEMPT DATA (for charts)
    student_stats = []
    for s in students:
        s_results = QuizResult.query.filter_by(user_id=s.id).all()
        total_score = sum(r.score for r in s_results)
        total_total = sum(r.total for r in s_results)

        avg_score = round((total_score / total_total) * 100, 2) if total_total > 0 else 0

        student_stats.append({
            "username": s.username,
            "attempts": len(s_results),
            "avg_score": avg_score
        })

    return render_template(
        "admin_analytics.html",
        total_students=total_students,
        total_materials=total_materials,
        total_quizzes=total_quizzes,
        subject_stats=subject_stats,
        student_stats=student_stats,
        student_scores=student_scores
    )


@app.route("/student/dashboard")
@login_required(role="student")
def student_dashboard():
    materials = Material.query.order_by(Material.subject).all()
    return render_template("student_dashboard.html", materials=materials)


# -------------------
# Materials (Admin)
# -------------------
@app.route("/admin/materials/upload", methods=["GET", "POST"])
@login_required(role="admin")
def upload_material():
    if request.method == "POST":
        
        subject = request.form.get("subject")
        new_subject = request.form.get("new_subject")
        if new_subject.strip():
            subject = new_subject.strip()
        title = request.form["title"]
        mat_type = request.form["type"]

        filename = None
        content = None

        if mat_type == "pdf":
            file = request.files["file"]
            filename = file.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
        elif mat_type == "note":
            content = request.form["content_note"]
        elif mat_type == "video":
            content = request.form["content_video"]  # store link

        m = Material(subject=subject,title=title, filename=filename, type=mat_type, content=content)
        db.session.add(m)
        db.session.commit()

        students = User.query.filter_by(role="student").all()
        for s in students:
            create_notification(s.id, f"New material uploaded: {title}")
        return render_template("upload_material.html", msg="Material uploaded successfully!")


    return render_template("upload_material.html")

@app.route("/admin/materials/delete/<int:mid>")
@login_required(role="admin")
def delete_material(mid):
    m = Material.query.get_or_404(mid)
    if m.filename:
        path = os.path.join(UPLOAD_FOLDER, m.filename)
        if os.path.exists(path):
            os.remove(path)
    db.session.delete(m)
    db.session.commit()
    return redirect(url_for("admin_dashboard"))


# -------------------
# Materials View (Both)
# -------------------
@app.route("/materials")
@login_required()
def materials_list():
    user = current_user()
    subject = request.args.get("subject")

    if subject and subject != "All":
        materials = Material.query.filter_by(subject=subject).all()
    else:
        materials = Material.query.order_by(Material.subject, Material.title).all()
    return render_template("materials_list.html", materials=materials, role=user.role)

@app.route("/uploads/pdfs/<filename>")
@login_required()
def serve_pdf(filename):
    m = Material.query.filter_by(filename=filename).first()
    if m:
        log_activity("view_pdf", f"Viewed PDF: {m.title}", ref_id=m.id)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.context_processor
def inject_subjects():
    subjects = db.session.query(Material.subject).distinct().all()
    subjects = [s[0] for s in subjects]
    return dict(subject_list=subjects)
@app.context_processor
def inject_user():
    return dict(current_user=current_user)
@app.context_processor
def inject_notifications():
    if "user_id" in session:
        count = Notification.query.filter_by(
            user_id=session["user_id"], is_read=False
        ).count()
        return dict(notif_count=count)
    return dict(notif_count=0)



# -------------------
# RAG: Select PDF & Chat (Student)
# -------------------
@app.route("/select_pdf/<int:mid>")
@login_required(role="student")
def select_pdf(mid):
    global pdf_chunks, faiss_index, active_pdf
    m = Material.query.get_or_404(mid)
    if m.type != "pdf":
        return "Not a PDF material."

    pdf_path = os.path.join(UPLOAD_FOLDER, m.filename)
    text = extract_text_from_pdf(pdf_path)
    pdf_chunks = split_text(text)
    faiss_index = build_vector_db(pdf_chunks)
    active_pdf = m.filename

    # clear old pdf chat and log
    session.pop("pdf_chat_history", None)
    log_activity("open_pdf_tutor", f"Opened PDF tutor on: {m.title}", ref_id=m.id)

    return redirect(url_for("chat_with_pdf"))

@app.route("/view_note/<int:mid>")
@login_required()
def view_note(mid):
    material = Material.query.get_or_404(mid)
    log_activity("view_note", f"Viewed note: {material.title}", ref_id=material.id)
    return render_template("view_note.html", material=material)

@app.route("/view_video/<int:mid>", methods=["GET", "POST"])
@login_required()
def view_video(mid):
    material = Material.query.get_or_404(mid)
    url = material.content.strip()
    log_activity("view_video", f"Viewed video: {material.title}", ref_id=material.id)

    # Convert YouTube to embed URL
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[-1].split("&")[0]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
    elif "shorts/" in url:
        video_id = url.split("shorts/")[-1]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
    else:
        embed_url = url

    # AI Chat History
    if "video_chat_history" not in session:
        session["video_chat_history"] = []

    history = session["video_chat_history"]

    # If student asks a question
    if request.method == "POST":
        msg = request.form["msg"]

        history.append({"role":"user","content": msg})

        prompt = f"""
You are an AI tutor. The student is watching a video titled: {material.title}.
Answer the student's question clearly and simply.

Student Question: {msg}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role":"user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        history.append({"role":"assistant", "content": answer})

        session["video_chat_history"] = history

    return render_template("view_video.html", material=material, embed_url=embed_url, history=history)


@app.route("/test_video")
def test_video():
    url = "https://www.youtube.com/embed/kqtD5dpn9C8"
    return f"""
    <iframe width='600' height='400' src='{url}' frameborder='0' allowfullscreen></iframe>
    """
@app.route("/debug_video/<int:mid>")
def debug_video(mid):
    m = Material.query.get(mid)
    return f"RAW STORED URL:<br>{m.content}"

@app.route("/chat", methods=["GET", "POST"])
@login_required(role="student")
def chat_with_pdf():
    global active_pdf

    if not active_pdf:
        return render_template("chat.html", pdf=None, history=[], lang="English")

    if "pdf_chat_history" not in session:
        session["pdf_chat_history"] = []

    history = session["pdf_chat_history"]

    lang = request.args.get("lang") or request.form.get("lang") or "English"
    reply = None

    if request.method == "POST":
        question = request.form["msg"]

        log_activity("pdf_tutor_question", f"Q: {question}", ref_id=None)

        history.append({"role": "user", "content": question})

        pdf_context = search_pdf(question)

        prompt = f"""
You are an AI tutor answering questions based ONLY on the given PDF content.

Preferred language: {lang}

PDF Content:
{pdf_context}

Student Question:
{question}

Give a clear and helpful answer in {lang}.
Ensure your answer ends with a complete, meaningful sentence.
Do NOT stop mid-sentence.
Do NOT mention that you used PDF content.
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        reply = response.choices[0].message.content

        log_activity("pdf_tutor_reply", f"AI replied: {reply[:100]}")

        history.append({"role": "assistant", "content": reply})
        session["pdf_chat_history"] = history

    return render_template("chat.html", history=history, pdf=active_pdf, lang=lang)


@app.route("/ai_tutor", methods=["GET", "POST"])
@login_required(role="student")
def ai_tutor():
    # Initialize chat history if not present
    if "chat_history" not in session:
        session["chat_history"] = []

    history = session["chat_history"]
    reply = None

    if request.method == "POST":
        msg = request.form["msg"]
        lang = request.form.get("lang", "English")

        # Add user message to history
        history.append({"role": "user", "content": msg})

        log_activity("ai_tutor_question", f"AI tutor Q: {msg}", ref_id=None)
   

        # Prepare Groq messages
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.insert(0, {"role": "system", "content": f"You are an AI tutor. Respond in {lang}. Ensure your answer ends with a complete, grammatically correct sentence."})

        # Get model response
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=500
        )

        reply = response.choices[0].message.content
        log_activity("ai_tutor_reply", f"AI replied: {reply[:100]}")


        # Add AI response to chat history
        history.append({"role": "assistant", "content": reply})

        # Save back to session
        session["chat_history"] = history

    return render_template("ai_tutor.html", reply=reply, history=history)

@app.route("/clear_ai_chat")
@login_required(role="student")
def clear_ai_chat():
    session.pop("chat_history", None)
    return redirect(url_for("ai_tutor"))


@app.route("/clear_pdf_chat")
@login_required(role="student")
def clear_pdf_chat():
    session.pop("pdf_chat_history", None)
    return redirect(url_for("chat_with_pdf"))


# -------------------
# Quiz
# -------------------
@app.route("/admin/quiz", methods=["GET", "POST"])
@login_required(role="admin")
def admin_quiz():
    if request.method == "POST":
        subject = request.form["subject"]
        unit = request.form["unit"]
        question = request.form["question"]
        a = request.form["a"]
        b = request.form["b"]
        c = request.form["c"]
        d = request.form["d"]
        correct = request.form["correct"]

        q = Quiz(
            subject = subject.strip().title(),
            unit = unit.strip().title(),
            question=question,
            option_a=a,
            option_b=b,
            option_c=c,
            option_d=d,
            correct=correct
        )
        db.session.add(q)
        db.session.commit()

        students = User.query.filter_by(role="student").all()
        #for s in students:
         # create_notification(s.id, f"New quiz added in {subject}")

        return render_template("admin_quiz.html", msg="Question added!")

    return render_template("admin_quiz.html")

@app.route("/admin/quiz/manage") 
@login_required(role="admin")
def manage_quiz():
    selected_subject = request.args.get("subject", "all")
    selected_unit = request.args.get("unit", "all")

    subjects = db.session.query(Quiz.subject).distinct().all()
    subjects = [s[0] for s in subjects]

    units = db.session.query(Quiz.unit).distinct().all()
    units = [u[0] for u in units]

    query = Quiz.query

    if selected_subject != "all":
        query = query.filter_by(subject=selected_subject)

    if selected_unit != "all":
        query = query.filter_by(unit=selected_unit)

    questions = query.all()

    return render_template(
        "manage_quiz.html",
        questions=questions,
        subjects=subjects,
        units=units,
        selected_subject=selected_subject,
        selected_unit=selected_unit
    )


@app.route("/admin/quiz/edit/<int:qid>", methods=["GET", "POST"])
@login_required(role="admin")
def edit_question(qid):
    q = Quiz.query.get_or_404(qid)

    if request.method == "POST":
        q.subject = request.form["subject"]
        q.unit = request.form["unit"]          # <-- FIXED
        q.question = request.form["question"]
        q.option_a = request.form["a"]
        q.option_b = request.form["b"]
        q.option_c = request.form["c"]
        q.option_d = request.form["d"]
        q.correct = request.form["correct"]

        db.session.commit()
        return redirect(url_for("manage_quiz"))

    return render_template("edit_quiz.html", q=q)


@app.route("/admin/quiz/delete/<int:qid>")
@login_required(role="admin")
def delete_question(qid):
    q = Quiz.query.get_or_404(qid)
    subject = q.subject
    db.session.delete(q)
    db.session.commit()

    # Notify all students that quiz was removed
    students = User.query.filter_by(role="student").all()
    for s in students:
        create_notification(s.id, f"A quiz in {subject} has been removed by the teacher.")
    return redirect("/admin/quiz/manage")

@app.route("/quiz/subjects")
@login_required(role="student")
def quiz_subjects():
    subjects = db.session.query(Quiz.subject).distinct().all()
    subjects = [s[0] for s in subjects]
    return render_template("quiz_subjects.html", subjects=subjects)

@app.route("/quiz/<subject>/units")
@login_required(role="student")
def quiz_units(subject):
    units = db.session.query(Quiz.unit).filter_by(subject=subject).distinct().all()
    units = [u[0] for u in units]
    return render_template("quiz_units.html", subject=subject, units=units)

@app.route("/quiz/<subject>")
@login_required(role="student")
def redirect_to_units(subject):
    return redirect(url_for("quiz_units", subject=subject))

@app.route("/quiz/<subject>/<unit>", methods=["GET", "POST"])
@login_required(role="student")
def take_quiz_unitwise(subject, unit):
    questions = Quiz.query.filter_by(subject=subject, unit=unit).all()

    if request.method == "POST":
        score = 0
        total = len(questions)
        results = []

        for q in questions:
            chosen = request.form.get(str(q.id))
            correct = q.correct

            if chosen == correct:
                score += 1

            results.append({
                "question": q.question,
                "chosen": chosen,
                "correct": correct
            })

        # SAVE RESULT
        saved = QuizResult(
            user_id=session["user_id"],
            subject=f"{subject} - {unit}",   # store subject + unit
            score=score,
            total=total
        )
        db.session.add(saved)
        db.session.commit()

        student = User.query.get(session["user_id"])
        notify_admins(f"{student.username} scored {score}/{total} in {subject} quiz.")


        create_notification(
    session["user_id"],
    f"You scored {score}/{total} in {subject} quiz"
)


        log_activity(
    "quiz_submit",
    f"Submitted quiz: {subject} | Score: {score}/{total}",
    ref_id=saved.id
)

        return render_template(
            "quiz_result.html",
            score=score, total=total,
            results=results, subject=f"{subject} - {unit}"
        )
    log_activity("quiz_start", f"Started quiz: {subject}", ref_id=None)
    return render_template(
        "take_quiz.html",
        questions=questions,
        subject=f"{subject} - {unit}"
    )

@app.route("/admin/quiz/publish", methods=["POST"])
@login_required(role="admin")
def publish_quiz():
    subject = request.form.get("subject")
    unit = request.form.get("unit")

    if not subject or not unit or subject == "all" or unit == "all":
        flash("Please select a subject and unit before publishing.", "warning")
        return redirect(url_for("manage_quiz"))

    students = User.query.filter_by(role="student").all()

    for s in students:
        n = Notification(
            user_id=s.id,
            message=f"📢 New quiz published: {subject} – {unit}"
        )
        db.session.add(n)

    db.session.commit()
    flash(f"{subject} {unit} quiz published successfully!", "success")

    return redirect(url_for("manage_quiz"))



@app.route("/student/analytics")
@login_required(role="student")
def student_analytics():
    uid = session["user_id"]

    results = QuizResult.query.filter_by(user_id=uid).all()

    # CASE 1: No quiz attempts
    if not results:
        performance_score = calculate_performance(uid)
        return render_template(
            "student_analytics.html",
            subjects=[],
            scores=[],
            totals=[],
            percentages=[],
            attempt_counts={},
            performance_score=performance_score,
            study_tips=[],                  # ADDED
            learning_recommendations=[]     # ADDED
        )

    # CASE 2: Student has quiz attempts
    subject_data = {}

    for r in results:
        # Remove unit part from subject name if present
        subject = r.subject.split(" - ")[0].strip()

        if subject not in subject_data:
            subject_data[subject] = {"score": 0, "total": 0, "attempts": 0}

        subject_data[subject]["score"] += r.score
        subject_data[subject]["total"] += r.total
        subject_data[subject]["attempts"] += 1

    subjects = []
    scores = []
    totals = []          # ADDED: to show total marks per subject in table
    percentages = []
    attempt_counts = {}

    for subject, data in subject_data.items():
        subjects.append(subject)

        pct = (data["score"] / data["total"]) * 100 if data["total"] else 0
        pct = round(pct, 2)

        percentages.append(pct)
        scores.append(data["score"])
        totals.append(data["total"])          # ADDED
        attempt_counts[subject] = data["attempts"]

    # -------- PERSONALIZED STUDY TIPS (simple) --------
    study_tips = []
    for i, subject in enumerate(subjects):
        pct = percentages[i]

        if pct < 50:
            study_tips.append(
                f"Your performance in {subject} is low. Revise the notes and try the quiz again."
            )
        elif pct < 80:
            study_tips.append(
                f"You're doing fairly well in {subject}. Practice a few more quizzes to improve."
            )
        else:
            study_tips.append(
                f"Excellent performance in {subject}! Maintain your practice."
            )

    # -------- SMART LEARNING RECOMMENDATIONS --------
    learning_recommendations = []

    for subject, data in subject_data.items():
        pct = (data["score"] / data["total"]) * 100 if data["total"] > 0 else 0
        attempts = data["attempts"]

        # Rule 1: Weak subjects (<50%)
        if pct < 50:
            learning_recommendations.append(
                f"Focus on {subject}: start from basics (Unit 1 & Unit 2) and then retake the quiz."
            )

        # Rule 2: Average performance (50–80%)
        elif pct < 80:
            learning_recommendations.append(
                f"In {subject}, you have a moderate score. Strengthen by practicing more medium-level MCQs."
            )

        # Rule 3: Strong subjects (>80%)
        else:
            learning_recommendations.append(
                f"You are strong in {subject}. Try higher difficulty questions to master this subject."
            )

        # Rule 4: Only one quiz attempt
        if attempts == 1:
            learning_recommendations.append(
                f"You attempted only one quiz in {subject}. Retaking it after revision will improve retention."
            )

    # Rule 5: Suggest untouched subjects/materials (very simple version)
    opened_subjects = set(subject_data.keys())
    all_materials = Material.query.all()

    for m in all_materials:
        if m.subject not in opened_subjects:
            learning_recommendations.append(
                f"You haven't explored materials for {m.subject} yet. Go through them to build a foundation."
            )

    # Always calculate performance score at the end
    performance_score = calculate_performance(uid)

    return render_template(
        "student_analytics.html",
        subjects=subjects,
        scores=scores,
        totals=totals,                      # now filled correctly
        percentages=percentages,
        attempt_counts=attempt_counts,
        performance_score=performance_score,
        study_tips=study_tips,              # ADDED
        learning_recommendations=learning_recommendations  # ADDED
    )


@app.route("/admin/students/monitor")
@login_required(role="admin")
def monitor_students():
    students = User.query.filter_by(role="student").all()
    overview = []

    for s in students:
        total_quizzes = QuizResult.query.filter_by(user_id=s.id).count()
        total_activities = ActivityLog.query.filter_by(user_id=s.id).count()
        last_log = ActivityLog.query.filter_by(user_id=s.id).order_by(ActivityLog.timestamp.desc()).first()

        overview.append({
            "id": s.id,
            "username": s.username,
            "total_quizzes": total_quizzes,
            "total_activities": total_activities,
            "last_activity": last_log.timestamp.strftime("%Y-%m-%d %H:%M") if last_log else "No activity"
        })

    return render_template("monitor_students.html", overview=overview)

@app.route("/admin/student/<int:uid>/activity")
@login_required(role="admin")
def student_activity(uid):
    user = User.query.get_or_404(uid)
    logs = ActivityLog.query.filter_by(user_id=uid).order_by(ActivityLog.timestamp.desc()).limit(100).all()
    quiz_results = QuizResult.query.filter_by(user_id=uid).order_by(QuizResult.timestamp.desc()).all()

    return render_template(
        "student_activity.html",
        student=user,
        logs=logs,
        quiz_results=quiz_results
    )

@app.route("/admin/delete_log/<int:log_id>/<int:uid>")
@login_required(role="admin")
def delete_log(log_id, uid):
    log = ActivityLog.query.get_or_404(log_id)
    db.session.delete(log)
    db.session.commit()
    return redirect(url_for('student_activity', uid=uid))

@app.route("/notifications")
@login_required()
def notifications():
    user_id = session["user_id"]
    notes = Notification.query.filter_by(user_id=user_id).order_by(Notification.timestamp.desc()).all()

    # Mark all as read when viewed
    for n in notes:
        n.is_read = True
    db.session.commit()

    return render_template("notifications.html", notes=notes)

@app.route("/notifications/clear")
@login_required()
def clear_notifications():
    user_id = session["user_id"]
    Notification.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return redirect(url_for("notifications"))




# -------------------
# Init
# -------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
