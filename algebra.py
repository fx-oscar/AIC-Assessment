import streamlit as st
import sqlite3
import os
import time
import random
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional

from owlready2 import get_ontology, sync_reasoner_pellet

# =========================================================
# CONFIG
# =========================================================

DB_PATH = "its_streamlit.db"

# IMPORTANT: make sure this filename matches your actual OWL file
# Put the OWL file in the same folder as this .py file.
OWL_PATH = os.path.join(os.path.dirname(__file__), "its_algebra_example.owl")


# =========================================================
# DB HELPERS
# =========================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


@st.cache_resource
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            total_questions INTEGER DEFAULT 0,
            correct_questions INTEGER DEFAULT 0,
            avg_response_time REAL DEFAULT 0.0,
            created_at TEXT NOT NULL
        )
        """
    )

    # Attempts table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question_type TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            question_text TEXT NOT NULL,
            correct_answer TEXT NOT NULL,
            student_answer TEXT NOT NULL,
            is_correct INTEGER NOT NULL,
            error_type TEXT,
            response_time REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    conn.commit()


# Add email normalization to prevent case issues
def normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(full_name, email, password):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    email = normalize_email(email)  # Normalize email
    cur.execute(
        """
        INSERT INTO users (full_name, email, password_hash, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (full_name, email, hash_password(password), now),
    )
    conn.commit()


def get_user_by_email(email):
    conn = get_connection()
    cur = conn.cursor()
    email = normalize_email(email)  # Normalize email
    cur.execute("SELECT * FROM users WHERE email = ?", (email,))
    return cur.fetchone()


def get_user_by_id(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cur.fetchone()


def update_user_stats(user_id, is_correct, response_time):
    conn = get_connection()
    cur = conn.cursor()

    user = get_user_by_id(user_id)
    if not user:
        return

    total = user["total_questions"] + 1
    correct = user["correct_questions"] + (1 if is_correct else 0)
    old_avg = user["avg_response_time"] or 0.0

    if user["total_questions"] == 0:
        new_avg = response_time
    else:
        new_avg = ((old_avg * user["total_questions"]) + response_time) / total

    cur.execute(
        """
        UPDATE users
        SET total_questions = ?, correct_questions = ?, avg_response_time = ?
        WHERE id = ?
        """,
        (total, correct, new_avg, user_id),
    )
    conn.commit()


def log_attempt(user_id, q_type, difficulty, question_text, correct_answer,
                student_answer, is_correct, error_type, response_time):
    conn = get_connection()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT INTO attempts (
            user_id, question_type, difficulty, question_text,
            correct_answer, student_answer, is_correct,
            error_type, response_time, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id, q_type, difficulty, question_text, correct_answer,
            student_answer, 1 if is_correct else 0,
            error_type, response_time, now,
        ),
    )
    conn.commit()


def get_recent_attempts(user_id, limit=10):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM attempts
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )
    return cur.fetchall()


# =========================================================
# AUTH SESSION
# =========================================================

def login_user_session(user_row):
    st.session_state["user_id"] = user_row["id"]
    st.session_state["user_email"] = user_row["email"]
    st.session_state["user_name"] = user_row["full_name"]


def logout_user_session():
    for key in ["user_id", "user_email", "user_name",
                "current_question", "last_result"]:
        st.session_state.pop(key, None)


def is_logged_in():
    return "user_id" in st.session_state


# =========================================================
# OWL / ITS REASONER – USING YOUR ALGEBRA ONTOLOGY
# =========================================================

# Map Python error types to your misconception individual
ERROR_TYPE_TO_MISCONCEPTION = {
    "algebraic_manipulation_error": "one_sided_op_example",
}

# Human-readable texts for Hint individuals
HINT_TEXT_BY_INDIVIDUAL = {
    "hint_subtract_both_sides": (
        "Try subtracting the same value from both sides of the equation. "
        "That keeps the equation balanced and helps you isolate x."
    ),
}


class ITSReasoner:
    def __init__(self, owl_path: str):
        self.owl_path = owl_path
        self.onto = None
        self.load_ontology()

    def load_ontology(self):
        if not os.path.exists(self.owl_path):
            print(f"[WARNING] OWL file not found at {self.owl_path}")
            print(f"[INFO] App will continue with generic hints only.")
            self.onto = None
            return
        
        try:
            print(f"[INFO] Loading ontology from {self.owl_path}")
            self.onto = get_ontology(f"file:///{self.owl_path}").load()
            try:
                with self.onto:
                    sync_reasoner_pellet(
                        infer_property_values=True,
                        infer_data_property_values=True,
                    )
                print("[INFO] ✅ Ontology loaded and reasoner initialized successfully.")
            except Exception as e:
                print(f"[WARNING] Pellet reasoner warning: {e}")
                print("[INFO] Ontology loaded without reasoner - hints will still work.")
        except Exception as e:
            print(f"[ERROR] Failed to load ontology: {e}")
            self.onto = None

    def _get_individual_by_name(self, name: Optional[str]):
        if not self.onto or not name:
            return None
        try:
            return self.onto[name]
        except KeyError:
            return None

    def get_misconception_individual(self, error_type: Optional[str]):
        owl_name = ERROR_TYPE_TO_MISCONCEPTION.get(error_type) if error_type else None
        return self._get_individual_by_name(owl_name)

    def _get_hints_from_misconception(self, misc_ind):
        if misc_ind is None or not self.onto:
            return []
        hints = []
        has_hint_attr = getattr(misc_ind, "hasHint", None)
        if has_hint_attr:
            for h in has_hint_attr:
                name = h.name
                if name in HINT_TEXT_BY_INDIVIDUAL:
                    hints.append(HINT_TEXT_BY_INDIVIDUAL[name])
        return hints

    def suggest_hint(self, question_type: str, error_type: Optional[str]):
        misc_ind = self.get_misconception_individual(error_type)
        hints = self._get_hints_from_misconception(misc_ind)

        if hints:
            return hints[0]

        # Generic fallbacks
        if error_type == "algebraic_manipulation_error":
            return (
                "Think about how to keep the equation balanced: "
                "whatever you do to one side, do to the other side as well."
            )
        elif error_type == "vector_addition_error":
            return "Add the x-components together and the y-components together."
        elif error_type == "matrix_addition_error":
            return "Add matrices element by element: C[i][j] = A[i][j] + B[i][j]."

        return "Go through your steps carefully and check each calculation."


@st.cache_resource
def get_reasoner():
    return ITSReasoner(OWL_PATH)


# =========================================================
# QUESTION GENERATION & EVALUATION
# =========================================================

QUESTION_TYPES = [
    "scalar_equation",
    "vector_addition",
    "matrix_addition_2x2",
]

DIFFICULTY_LEVELS = {
    "easy": "🟢 Easy",
    "medium": "🟡 Medium",
    "hard": "🔴 Hard",
}

TOPICS = {
    "scalar_equation": "Linear Equations (ax + b = c)",
    "vector_addition": "Vector Addition",
    "matrix_addition_2x2": "Matrix Addition (2×2)",
}

STEP_BY_STEP_SOLUTIONS = {
    "scalar_equation": [
        "Step 1: Identify the equation format (ax + b = c)",
        "Step 2: Subtract b from both sides to isolate the x term",
        "Step 3: Divide both sides by a to solve for x",
        "Step 4: Verify your answer by substituting back into the original equation",
    ],
    "vector_addition": [
        "Step 1: Write the vectors in component form (x, y)",
        "Step 2: Add the x-components together",
        "Step 3: Add the y-components together",
        "Step 4: Write the result as (sum_x, sum_y)",
    ],
    "matrix_addition_2x2": [
        "Step 1: Identify each matrix element position [i][j]",
        "Step 2: Add corresponding elements: C[0][0] = A[0][0] + B[0][0]",
        "Step 3: Add all remaining elements following the same pattern",
        "Step 4: Write the resulting 2×2 matrix",
    ],
}


def generate_question(user_row, difficulty_override=None, topic_override=None):
    total = user_row["total_questions"]
    correct = user_row["correct_questions"]
    avg_time = user_row["avg_response_time"] or 0.0

    accuracy = (correct / total) if total > 0 else 0.0

    difficulty = difficulty_override or (
        "easy" if total < 5 else
        "hard" if accuracy > 0.8 and avg_time < 20 else
        "medium" if accuracy > 0.6 else
        "easy"
    )

    q_type = topic_override or (
        "scalar_equation" if difficulty == "easy" else
        "matrix_addition_2x2" if difficulty == "hard" else
        "vector_addition"
    )

    if q_type == "scalar_equation":
        a = random.randint(1, 9)
        x_val = random.randint(1, 10)
        b = random.randint(0, 9)
        c = a * x_val + b
        question_text = f"Solve for x: {a}x + {b} = {c}"
        correct_answer = str(x_val)

    elif q_type == "vector_addition":
        v1 = (random.randint(-5, 5), random.randint(-5, 5))
        v2 = (random.randint(-5, 5), random.randint(-5, 5))
        result = (v1[0] + v2[0], v1[1] + v2[1])
        question_text = (
            f"Compute v₁ + v₂ where v₁ = {v1} and v₂ = {v2}.\n"
            f"Answer as (x, y)."
        )
        correct_answer = f"({result[0]}, {result[1]})"

    elif q_type == "matrix_addition_2x2":
        A = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        B = [[random.randint(-5, 5) for _ in range(2)] for _ in range(2)]
        C = [
            [A[0][0] + B[0][0], A[0][1] + B[0][1]],
            [A[1][0] + B[1][0], A[1][1] + B[1][1]],
        ]
        question_text = (
            f"Compute A + B for 2x2 matrices.\n"
            f"A = {A}\nB = {B}\n"
            f"Answer as [[a,b],[c,d]]."
        )
        correct_answer = str(C)

    else:
        question_text = "Error: Unknown question type."
        correct_answer = ""

    st.session_state["current_question"] = {
        "type": q_type,
        "difficulty": difficulty,
        "text": question_text,
        "correct_answer": correct_answer,
        "start_time": time.time(),
    }

    return question_text, difficulty, q_type


def evaluate_answer(student_answer: str):
    q = st.session_state.get("current_question")
    if not q:
        return (None,) * 8

    correct = q["correct_answer"]
    q_type = q["type"]
    difficulty = q["difficulty"]
    question_text = q["text"]

    start_time = q.get("start_time")
    response_time = time.time() - start_time if start_time else 0.0

    sa = student_answer.strip()
    ca = correct.strip()

    # SMART COMPARISON
    if q_type == "vector_addition":
        def parse_vec(s: str):
            nums = re.findall(r"-?\d+", s)
            if len(nums) != 2:
                return None
            return (int(nums[0]), int(nums[1]))

        stu_vec = parse_vec(sa)
        cor_vec = parse_vec(ca)
        if stu_vec is not None and cor_vec is not None:
            is_correct = (stu_vec == cor_vec)
        else:
            is_correct = (sa == ca)

    elif q_type == "scalar_equation":
        try:
            is_correct = float(sa) == float(ca)
        except ValueError:
            is_correct = (sa == ca)

    else:
        # matrix / others – simple comparison for now
        is_correct = (sa == ca)

    # ERROR TYPE
    if is_correct:
        error_type = None
    else:
        if q_type == "scalar_equation":
            error_type = "algebraic_manipulation_error"
        elif q_type == "vector_addition":
            error_type = "vector_addition_error"
        elif q_type == "matrix_addition_2x2":
            error_type = "matrix_addition_error"
        else:
            error_type = "unknown_error"

    reasoner = get_reasoner()
    hint = reasoner.suggest_hint(q_type, error_type)

    return (
        is_correct,
        error_type,
        hint,
        question_text,
        correct,
        response_time,
        q_type,
        difficulty,
    )


# =========================================================
# UI PAGES
# =========================================================

def page_home():
    st.title("Intelligent Algebra / Linear Algebra Tutor")

    st.write(
        """
This Intelligent Tutoring System lets you practice **Algebra / Linear Algebra**:

- Linear equations, vectors, and 2×2 matrices  
- Adaptive difficulty based on your performance  
- Ontology-based hints using an OWL domain model  
        """
    )
    if not is_logged_in():
        st.info("Create an account or log in to track your progress.")
    else:
        st.success(f"Welcome, {st.session_state['user_name']}!")
        if st.button("Go to Practice"):
            st.session_state["page"] = "Practice"
            st.rerun()


def page_register():
    st.title("Register")

    with st.form("register_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")

    if submitted:
        if not full_name or not email or not password or not confirm:
            st.error("Please fill in all fields.")
            return
        if password != confirm:
            st.error("Passwords do not match.")
            return
        if get_user_by_email(email):
            st.error("Email already registered.")
            return

        create_user(full_name, email, password)
        st.success("Registration successful. You can now log in.")
        st.session_state["page"] = "Login"
        st.rerun()


def page_login():
    st.title("Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        user = get_user_by_email(email)
        if user and user["password_hash"] == hash_password(password):
            login_user_session(user)
            st.success("Logged in successfully.")
            st.session_state["page"] = "Dashboard"
            st.rerun()
        else:
            st.error("Invalid email or password.")


def page_dashboard():
    st.title("Dashboard")
    user = get_user_by_id(st.session_state["user_id"])

    total = user["total_questions"]
    correct = user["correct_questions"]
    accuracy = (correct / total * 100) if total > 0 else 0.0
    avg_time = user["avg_response_time"] or 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total questions", total)
    col2.metric("Correct answers", correct)
    col3.metric("Accuracy (%)", f"{accuracy:.1f}")

    st.metric("Average response time (s)", f"{avg_time:.1f}")

    st.subheader("Recent attempts")
    attempts = get_recent_attempts(user["id"], limit=10)
    if not attempts:
        st.info("No attempts yet. Go to Practice to start.")
    else:
        rows = []
        for a in attempts:
            rows.append({
                "Question type": a["question_type"],
                "Difficulty": a["difficulty"],
                "Correct?": "✅" if a["is_correct"] else "❌",
                "Error type": a["error_type"] or "-",
                "Time (s)": f"{(a['response_time'] or 0.0):.1f}",
                "When": a["created_at"].replace("T", " ")[:16],
            })
        st.table(rows)


def page_profile():
    st.title("👤 Profile")
    user = get_user_by_id(st.session_state["user_id"])
    created_at = datetime.fromisoformat(user["created_at"])
    
    # Make created_at timezone-aware if it isn't already
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    
    days_on_platform = (datetime.now(timezone.utc) - created_at).days

    # Display current profile info
    st.subheader("📋 Current Profile Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {user['full_name']}")
        st.write(f"**Email:** {user['email']}")
    with col2:
        st.write(f"**Member since:** {created_at.strftime('%Y-%m-%d')}")
        st.write(f"**Days on platform:** {days_on_platform} 🔥")

    st.write("---")

    # Edit Profile Section
    st.subheader("✏️ Edit Profile")
    
    with st.form("edit_profile_form"):
        new_full_name = st.text_input(
            "Full Name",
            value=user['full_name'],
            help="Update your full name"
        )
        
        new_password = st.text_input(
            "New Password (leave blank to keep current)",
            type="password",
            help="Enter a new password or leave blank"
        )
        
        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
            help="Confirm your new password"
        )
        
        submitted = st.form_submit_button("💾 Save Changes")

    if submitted:
        errors = []
        
        # Validate full name
        if not new_full_name.strip():
            errors.append("Full name cannot be empty.")
        
        # Validate password if provided
        if new_password or confirm_password:
            if new_password != confirm_password:
                errors.append("Passwords do not match.")
            elif len(new_password) < 6:
                errors.append("Password must be at least 6 characters long.")
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        else:
            # Update profile in database
            conn = get_connection()
            cur = conn.cursor()
            
            if new_password:
                # Update both name and password
                cur.execute(
                    """
                    UPDATE users
                    SET full_name = ?, password_hash = ?
                    WHERE id = ?
                    """,
                    (new_full_name, hash_password(new_password), user["id"]),
                )
            else:
                # Update only name
                cur.execute(
                    """
                    UPDATE users
                    SET full_name = ?
                    WHERE id = ?
                    """,
                    (new_full_name, user["id"]),
                )
            
            conn.commit()
            
            # Update session state
            st.session_state["user_name"] = new_full_name
            
            st.success("✅ Profile updated successfully!")
            st.balloons()
            time.sleep(1)
            st.rerun()

    st.write("---")

    # Learning Statistics
    st.subheader("📊 Learning Statistics")
    total = user["total_questions"]
    correct = user["correct_questions"]
    accuracy = (correct / total * 100) if total > 0 else 0.0
    avg_time = user["avg_response_time"] or 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Total Questions", total)
    with col2:
        st.metric("✅ Correct Answers", correct)
    with col3:
        st.metric("📊 Accuracy", f"{accuracy:.1f}%")
    with col4:
        st.metric("⏱️ Avg Response Time", f"{avg_time:.1f}s")

    st.write("---")

    # Recent Attempts
    st.subheader("📈 Recent Attempts")
    attempts = get_recent_attempts(user["id"], limit=5)
    if not attempts:
        st.info("No attempts yet. Go to Practice to start!")
    else:
        rows = []
        for a in attempts:
            rows.append({
                "Question Type": a["question_type"],
                "Difficulty": a["difficulty"],
                "Result": "✅ Correct" if a["is_correct"] else "❌ Incorrect",
                "Time (s)": f"{(a['response_time'] or 0.0):.1f}",
                "When": a["created_at"].replace("T", " ")[:16],
            })
        st.table(rows)

    st.write("---")

    # Danger Zone - Delete Account
    st.subheader("⚠️ Danger Zone")
    if st.button("🗑️ Delete Account", help="This action cannot be undone"):
        st.warning("⚠️ Are you sure you want to delete your account? This action cannot be undone!")
        if st.button("Yes, delete my account permanently"):
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM attempts WHERE user_id = ?", (user["id"],))
            cur.execute("DELETE FROM users WHERE id = ?", (user["id"],))
            conn.commit()
            
            st.error("❌ Account deleted successfully.")
            logout_user_session()
            st.session_state["page"] = "Home"
            time.sleep(1)
            st.rerun()


def page_practice():
    st.title("📚 Practice – Linear Algebra / Equations")

    user = get_user_by_id(st.session_state["user_id"])

    # Initialize session state for selections
    if "selected_difficulty" not in st.session_state:
        st.session_state["selected_difficulty"] = "easy"
    if "selected_question_type" not in st.session_state:
        st.session_state["selected_question_type"] = "scalar_equation"

    # TOP SECTION: Dropdowns for difficulty and question type selection
    st.subheader("🎯 Customize Your Practice")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_diff = st.selectbox(
            "📊 Select Difficulty Level",
            options=list(DIFFICULTY_LEVELS.keys()),
            format_func=lambda x: DIFFICULTY_LEVELS[x],
            key="diff_select",
            index=list(DIFFICULTY_LEVELS.keys()).index(st.session_state["selected_difficulty"])
        )
        if selected_diff != st.session_state["selected_difficulty"]:
            st.session_state["selected_difficulty"] = selected_diff
            # Clear current question to force regeneration
            st.session_state.pop("current_question", None)
            st.session_state.pop("last_result", None)
            st.rerun()

    with col2:
        selected_qtype = st.selectbox(
            "📝 Select Question Type",
            options=list(TOPICS.keys()),
            format_func=lambda x: TOPICS[x],
            key="qtype_select",
            index=list(TOPICS.keys()).index(st.session_state["selected_question_type"])
        )
        if selected_qtype != st.session_state["selected_question_type"]:
            st.session_state["selected_question_type"] = selected_qtype
            # Clear current question to force regeneration
            st.session_state.pop("current_question", None)
            st.session_state.pop("last_result", None)
            st.rerun()

    st.write("---")

    # STEP-BY-STEP GUIDE (expandable section)
    if st.session_state["selected_question_type"] in STEP_BY_STEP_SOLUTIONS:
        with st.expander("📖 How to solve this type of problem", expanded=False):
            for step in STEP_BY_STEP_SOLUTIONS[st.session_state["selected_question_type"]]:
                st.write(f"**{step}**")

    st.write("---")

    # Generate or retrieve question - RESPECTS DROPDOWN SELECTIONS
    if "current_question" not in st.session_state:
        question_text, difficulty, q_type = generate_question(
            user,
            difficulty_override=st.session_state["selected_difficulty"],
            topic_override=st.session_state["selected_question_type"]
        )
    else:
        q = st.session_state["current_question"]
        question_text = q["text"]
        difficulty = q["difficulty"]
        q_type = q["type"]

    st.markdown(f"**Difficulty:** {difficulty.capitalize()}  |  **Question Type:** {TOPICS.get(q_type, q_type)}")
    st.write("### 🎯 Question")
    st.code(question_text)

    st.write("### ✍️ Your Answer")
    
    # Initialize answer input key in session state to force form reset
    if "answer_input_key" not in st.session_state:
        st.session_state["answer_input_key"] = 0
    
    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_input(
            "Type your answer here",
            key=f"answer_input_{st.session_state['answer_input_key']}"
        )
        submitted = st.form_submit_button("✅ Submit Answer")

    if submitted:
        if not answer.strip():
            st.error("❌ Please type an answer.")
        else:
            (
                is_correct,
                error_type,
                hint,
                q_text,
                correct_answer,
                response_time,
                q_type,
                difficulty,
            ) = evaluate_answer(answer)

            log_attempt(
                user_id=user["id"],
                q_type=q_type,
                difficulty=difficulty,
                question_text=q_text,
                correct_answer=correct_answer,
                student_answer=answer,
                is_correct=is_correct,
                error_type=error_type,
                response_time=response_time,
            )
            update_user_stats(user["id"], bool(is_correct), response_time)

            feedback_msg = "✅ Correct! Well done." if is_correct else "❌ Not quite. Keep trying."
            st.session_state["last_result"] = {
                "feedback": feedback_msg,
                "is_correct": is_correct,
                "correct_answer": correct_answer,
                "hint": hint,
                "error_type": error_type,
                "response_time": response_time,
            }
            st.rerun()

    # FEEDBACK SECTION (appears after submission)
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        st.write("---")
        st.subheader("📊 Feedback & Results")
        
        # Color-coded feedback with celebration
        if res["is_correct"]:
            st.success("🎉 " + res["feedback"])
            st.balloons()
        else:
            st.error(res["feedback"])
        
        # Detailed feedback in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Response Time", f"{res['response_time']:.1f}s")
        with col2:
            st.metric("📊 Difficulty", difficulty.capitalize())
        with col3:
            accuracy = "✅ Correct" if res["is_correct"] else "❌ Incorrect"
            st.metric("Result", accuracy)

        if not res["is_correct"]:
            st.warning(f"**Correct answer:** `{res['correct_answer']}`")
        
        if res["hint"]:
            st.info(f"💡 **Hint for next time:** {res['hint']}")
        
        if res["error_type"]:
            st.caption(f"🔍 Error detected: `{res['error_type']}`")

        # NEXT BUTTON - PROMINENTLY DISPLAYED
        st.write("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Next Question", key="next_btn", use_container_width=True):
                st.session_state.pop("current_question", None)
                st.session_state.pop("last_result", None)
                # Increment the key to force form reset
                st.session_state["answer_input_key"] += 1
                st.rerun()


# =========================================================
# MAIN
# =========================================================

def main():
    st.set_page_config(
        page_title="ITS – Algebra / Linear Algebra",
        layout="wide",
    )

    init_db()
    get_reasoner()

    with st.sidebar:
        st.title("ITS Tutor")
        
        if is_logged_in():
            st.write(f"👋 {st.session_state['user_name']}")
            page = st.radio(
                "Navigate",
                ["Home", "Dashboard", "Practice", "Profile"],
                index=["Home", "Dashboard", "Practice", "Profile"].index(
                    st.session_state.get("page", "Home")
                ) if st.session_state.get("page") in ["Home", "Dashboard", "Practice", "Profile"] else 0,
            )
            st.write("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout_user_session()
                st.session_state["page"] = "Home"
                st.rerun()
        else:
            page = st.radio(
                "Navigate",
                ["Home", "Login", "Register"],
                index=["Home", "Login", "Register"].index(
                    st.session_state.get("page", "Home")
                ) if st.session_state.get("page") in ["Home", "Login", "Register"] else 0,
            )

    st.session_state["page"] = page

    if page == "Home":
        page_home()
    elif page == "Login":
        page_login()
    elif page == "Register":
        page_register()
    elif page == "Dashboard":
        if is_logged_in():
            page_dashboard()
        else:
            st.warning("Please log in first.")
            page_login()
    elif page == "Practice":
        if is_logged_in():
            page_practice()
        else:
            st.warning("Please log in first.")
            page_login()
    elif page == "Profile":
        if is_logged_in():
            page_profile()
        else:
            st.warning("Please log in first.")
            page_login()


if __name__ == "__main__":
    main()
