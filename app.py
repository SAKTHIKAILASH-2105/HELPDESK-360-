"""
HelpDesk 360 - Production Flask Web Application
Employee IT Support Ticket Management System
"""

import os
import sqlite3
import joblib
from datetime import datetime, timezone
from functools import wraps
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    g,
    send_from_directory,
)
from werkzeug.security import check_password_hash, generate_password_hash

# Directory configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "helpdesk.db")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "priority_model.pkl")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Configuration
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "helpdesk360-dev-insecure-key-2026-prod")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["JSON_SORT_KEYS"] = False

# Fixed Enums
VALID_CATEGORIES = {"hardware", "network", "software", "account-access"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_STATUSES = {"open", "in-progress", "resolved", "closed"}

# Allowed status transitions: open -> in-progress -> resolved -> closed
VALID_STATUS_TRANSITIONS = {
    "open": {"in-progress"},
    "in-progress": {"resolved"},
    "resolved": {"closed"},
    "closed": set(),  # terminal state
}

# Priority ranking for sorting queue (urgent first)
PRIORITY_WEIGHTS = {
    "urgent": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

# Global ML Pipeline handle
ml_pipeline = None


def load_ml_model():
    """Loads the trained ML priority classifier if available."""
    global ml_pipeline
    if os.path.exists(MODEL_PATH):
        try:
            ml_pipeline = joblib.load(MODEL_PATH)
            print(f"[+] Loaded ML Priority Model from {MODEL_PATH}")
        except Exception as e:
            print(f"[!] Warning: Failed to load ML model ({e}). Fallback heuristic active.")
            ml_pipeline = None
    else:
        print("[!] Notice: Priority model not found at ml/priority_model.pkl. Run python ml/train_model.py")
        ml_pipeline = None


def predict_ticket_priority(title, description):
    """Predicts priority class and confidence score from title and description."""
    text = f"{title or ''} . {description or ''}".strip()
    if not text:
        return "medium", 0.50, {"low": 0.1, "medium": 0.5, "high": 0.2, "urgent": 0.2}

    if ml_pipeline is not None:
        try:
            pred = ml_pipeline.predict([text])[0]
            probs = ml_pipeline.predict_proba([text])[0]
            classes = ml_pipeline.classes_
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
            confidence = round(float(prob_dict.get(pred, 0.5)), 2)
            return pred, confidence, prob_dict
        except Exception as e:
            print(f"[!] Inference error ({e}), falling back to heuristic.")

    # Graceful heuristic fallback if model file is missing
    lower_text = text.lower()
    urgent_keywords = ["outage", "ransomware", "down", "critical", "breach", "emergency", "fire", "overheating", "500 error", "offline", "locked out"]
    high_keywords = ["broken", "fail", "cannot deploy", "crash", "blocked", "bsod", "pipeline", "vpn", "dead", "fatal"]
    low_keywords = ["mouse pad", "wrist rest", "clean", "cable", "sticker", "font", "question", "inquire", "how to", "ergonomic", "wallpaper"]

    for kw in urgent_keywords:
        if kw in lower_text:
            return "urgent", 0.85, {"urgent": 0.85, "high": 0.10, "medium": 0.03, "low": 0.02}
    for kw in high_keywords:
        if kw in lower_text:
            return "high", 0.80, {"urgent": 0.10, "high": 0.80, "medium": 0.07, "low": 0.03}
    for kw in low_keywords:
        if kw in lower_text:
            return "low", 0.85, {"urgent": 0.02, "high": 0.05, "medium": 0.08, "low": 0.85}

    return "medium", 0.65, {"urgent": 0.05, "high": 0.15, "medium": 0.65, "low": 0.15}


# -----------------------------------------------------------------------------
# Database Helpers
# -----------------------------------------------------------------------------

def get_db():
    """Opens a new database connection per request if none exists."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db


@app.teardown_appcontext
def close_db(error):
    """Closes the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def get_auto_assigned_technician(db):
    """
    Selects a technician automatically using workload balancing:
    finds the technician with the fewest active tickets ('open' or 'in-progress').
    Returns (technician_id, technician_name) or (None, None).
    """
    row = db.execute("""
        SELECT u.id, u.name, COUNT(t.id) as active_tickets
        FROM users u
        LEFT JOIN tickets t ON u.id = t.assignee_id AND t.status IN ('open', 'in-progress')
        WHERE u.role = 'technician'
        GROUP BY u.id
        ORDER BY active_tickets ASC, u.id ASC
        LIMIT 1
    """).fetchone()

    if row:
        return row["id"], row["name"]
    return None, None


def ensure_db_initialized():
    """Checks if the database exists and creates schema if missing."""
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        # Import seed script to auto-initialize
        try:
            from database.seed import seed_database
            print("[*] Database not detected. Initializing database with seed data...")
            seed_database()
        except Exception as e:
            print(f"[!] Warning during DB auto-seed: {e}")


def get_current_user():
    """Retrieves current authenticated user from session or None."""
    user_id = session.get("user_id")
    if not user_id:
        return None
    db = get_db()
    user = db.execute(
        "SELECT id, name, email, role, avatar, created_at FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()
    return user


# -----------------------------------------------------------------------------
# Auth & RBAC Decorators
# -----------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required. Please log in.", "code": "UNAUTHENTICATED"}), 401
        g.user = user
        return f(*args, **kwargs)
    return decorated_function


def technician_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({"error": "Authentication required. Please log in.", "code": "UNAUTHENTICATED"}), 401
        if user["role"] != "technician":
            return jsonify({
                "error": "Access denied. Technician privileges are required for this action.",
                "code": "FORBIDDEN"
            }), 403
        g.user = user
        return f(*args, **kwargs)
    return decorated_function


# -----------------------------------------------------------------------------
# Frontend Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Renders the single-page application dashboard."""
    return render_template("index.html")


# -----------------------------------------------------------------------------
# Authentication Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/login", methods=["POST"])
def api_login():
    """Authenticates user via email and password."""
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    db = get_db()
    user = db.execute(
        "SELECT id, name, email, password_hash, role, avatar, created_at FROM users WHERE email = ?",
        (email,),
    ).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    return jsonify({
        "message": "Login successful.",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "avatar": user["avatar"],
            "created_at": user["created_at"],
        },
    }), 200


@app.route("/api/me", methods=["GET"])
@login_required
def api_me():
    """Returns the current authenticated user's profile."""
    user = g.user
    return jsonify({
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "avatar": user["avatar"],
            "created_at": user["created_at"],
        }
    }), 200


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Clears the session."""
    session.clear()
    return jsonify({"message": "Logged out successfully."}), 200


# -----------------------------------------------------------------------------
# Technician Directory
# -----------------------------------------------------------------------------

@app.route("/api/technicians", methods=["GET"])
@login_required
def api_technicians():
    """Returns list of all technicians for assignment."""
    db = get_db()
    techs = db.execute(
        "SELECT id, name, email, avatar FROM users WHERE role = 'technician' ORDER BY name ASC"
    ).fetchall()
    return jsonify({
        "technicians": [
            {
                "id": t["id"],
                "name": t["name"],
                "email": t["email"],
                "avatar": t["avatar"],
            }
            for t in techs
        ]
    }), 200


# -----------------------------------------------------------------------------
# Machine Learning Priority Prediction Endpoint
# -----------------------------------------------------------------------------

@app.route("/api/predict-priority", methods=["POST"])
@login_required
def api_predict_priority():
    """Predicts ticket priority based on title and description."""
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()

    if not title and not description:
        return jsonify({"error": "Title or description required for priority prediction."}), 400

    pred_priority, confidence, prob_dict = predict_ticket_priority(title, description)

    return jsonify({
        "predicted_priority": pred_priority,
        "confidence": confidence,
        "probabilities": prob_dict,
    }), 200


# -----------------------------------------------------------------------------
# Dashboard Statistics Endpoint
# -----------------------------------------------------------------------------

@app.route("/api/dashboard", methods=["GET"])
@login_required
def api_dashboard():
    """Returns real-time dashboard metrics calculated directly from SQLite."""
    user = g.user
    db = get_db()

    if user["role"] == "technician":
        # Technician global metrics
        total_tickets = db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        open_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'").fetchone()[0]
        in_progress_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'in-progress'").fetchone()[0]
        resolved_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'resolved'").fetchone()[0]
        closed_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'closed'").fetchone()[0]
        urgent_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE priority = 'urgent' AND status IN ('open', 'in-progress')").fetchone()[0]
        unassigned_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE assignee_id IS NULL AND status IN ('open', 'in-progress')").fetchone()[0]
        assigned_to_me = db.execute("SELECT COUNT(*) FROM tickets WHERE assignee_id = ? AND status IN ('open', 'in-progress')", (user["id"],)).fetchone()[0]

        # Category Breakdown
        cat_rows = db.execute("""
            SELECT category, COUNT(*) as count 
            FROM tickets 
            GROUP BY category
        """).fetchall()
        categories = {c: 0 for c in VALID_CATEGORIES}
        for r in cat_rows:
            categories[r["category"]] = r["count"]

        # Priority Breakdown
        prio_rows = db.execute("""
            SELECT priority, COUNT(*) as count 
            FROM tickets 
            GROUP BY priority
        """).fetchall()
        priorities = {p: 0 for p in VALID_PRIORITIES}
        for r in prio_rows:
            priorities[r["priority"]] = r["count"]

        # Recent Activity (last 6 history changes)
        recent_activity_rows = db.execute("""
            SELECT h.id, h.ticket_id, h.old_status, h.new_status, h.note, h.created_at,
                   u.name as changed_by_name, u.role as changed_by_role,
                   t.title as ticket_title
            FROM status_history h
            JOIN users u ON h.changed_by = u.id
            JOIN tickets t ON h.ticket_id = t.id
            ORDER BY h.created_at DESC
            LIMIT 6
        """).fetchall()

        recent_activity = [
            {
                "id": r["id"],
                "ticket_id": r["ticket_id"],
                "ticket_title": r["ticket_title"],
                "old_status": r["old_status"],
                "new_status": r["new_status"],
                "note": r["note"],
                "created_at": r["created_at"],
                "changed_by": r["changed_by_name"],
                "changed_by_role": r["changed_by_role"],
            }
            for r in recent_activity_rows
        ]

        # Resolution rate percentage
        resolved_total = resolved_tickets + closed_tickets
        resolution_rate = round((resolved_total / total_tickets * 100), 1) if total_tickets > 0 else 0.0

        return jsonify({
            "role": "technician",
            "stats": {
                "total_tickets": total_tickets,
                "open_tickets": open_tickets,
                "in_progress_tickets": in_progress_tickets,
                "urgent_tickets": urgent_tickets,
                "assigned_tickets": total_tickets - unassigned_tickets,
                "resolved_tickets": resolved_tickets,
                "closed_tickets": closed_tickets,
                "unassigned_tickets": unassigned_tickets,
                "assigned_to_me": assigned_to_me,
                "resolution_rate": resolution_rate,
            },
            "categories": categories,
            "priorities": priorities,
            "recent_activity": recent_activity,
        }), 200

    else:
        # Employee scoped metrics (only employee's own tickets)
        emp_id = user["id"]
        total_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ?", (emp_id,)).fetchone()[0]
        open_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'open'", (emp_id,)).fetchone()[0]
        in_progress_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'in-progress'", (emp_id,)).fetchone()[0]
        resolved_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'resolved'", (emp_id,)).fetchone()[0]
        closed_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'closed'", (emp_id,)).fetchone()[0]
        urgent_tickets = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND priority = 'urgent'", (emp_id,)).fetchone()[0]

        cat_rows = db.execute("""
            SELECT category, COUNT(*) as count 
            FROM tickets 
            WHERE reporter_id = ?
            GROUP BY category
        """, (emp_id,)).fetchall()
        categories = {c: 0 for c in VALID_CATEGORIES}
        for r in cat_rows:
            categories[r["category"]] = r["count"]

        # Recent status history for employee's tickets
        recent_activity_rows = db.execute("""
            SELECT h.id, h.ticket_id, h.old_status, h.new_status, h.note, h.created_at,
                   u.name as changed_by_name, u.role as changed_by_role,
                   t.title as ticket_title
            FROM status_history h
            JOIN users u ON h.changed_by = u.id
            JOIN tickets t ON h.ticket_id = t.id
            WHERE t.reporter_id = ?
            ORDER BY h.created_at DESC
            LIMIT 5
        """, (emp_id,)).fetchall()

        recent_activity = [
            {
                "id": r["id"],
                "ticket_id": r["ticket_id"],
                "ticket_title": r["ticket_title"],
                "old_status": r["old_status"],
                "new_status": r["new_status"],
                "note": r["note"],
                "created_at": r["created_at"],
                "changed_by": r["changed_by_name"],
                "changed_by_role": r["changed_by_role"],
            }
            for r in recent_activity_rows
        ]

        return jsonify({
            "role": "employee",
            "stats": {
                "total_tickets": total_tickets,
                "open_tickets": open_tickets,
                "in_progress_tickets": in_progress_tickets,
                "urgent_tickets": urgent_tickets,
                "resolved_tickets": resolved_tickets,
                "closed_tickets": closed_tickets,
            },
            "categories": categories,
            "recent_activity": recent_activity,
        }), 200


# -----------------------------------------------------------------------------
# Ticket Management Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/tickets", methods=["GET"])
@login_required
def api_get_tickets():
    """
    Returns ticket list filtered by role and query params.
    Employees receive ONLY tickets they created.
    Technicians can view all tickets and filter by status, priority, category, assignee.
    Urgent tickets can be ordered first.
    """
    user = g.user
    db = get_db()

    status_filter = request.args.get("status", "").strip().lower()
    priority_filter = request.args.get("priority", "").strip().lower()
    category_filter = request.args.get("category", "").strip().lower()
    assignee_filter = request.args.get("assignee_id", "").strip()
    search_query = request.args.get("search", "").strip()
    queue_mode = request.args.get("queue", "").strip().lower()  # 'priority_queue' orders urgent first

    query = """
        SELECT 
            t.id, t.title, t.description, t.category, t.priority,
            t.predicted_priority, t.prediction_confidence,
            t.status, t.reporter_id, t.assignee_id, t.created_at, t.updated_at,
            rep.name as reporter_name, rep.email as reporter_email, rep.avatar as reporter_avatar,
            ass.name as assignee_name, ass.email as assignee_email, ass.avatar as assignee_avatar
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE 1=1
    """
    params = []

    # ENFORCE ROLE PERMISSION: Employee only sees own tickets
    if user["role"] == "employee":
        query += " AND t.reporter_id = ?"
        params.append(user["id"])
    elif assignee_filter:
        if assignee_filter == "unassigned":
            query += " AND t.assignee_id IS NULL"
        else:
            try:
                tech_id = int(assignee_filter)
                query += " AND t.assignee_id = ?"
                params.append(tech_id)
            except ValueError:
                return jsonify({"error": "Invalid assignee_id filter."}), 400

    # Filters
    if status_filter:
        if status_filter not in VALID_STATUSES:
            return jsonify({"error": f"Invalid status filter. Allowed: {list(VALID_STATUSES)}"}), 400
        query += " AND t.status = ?"
        params.append(status_filter)

    if priority_filter:
        if priority_filter not in VALID_PRIORITIES:
            return jsonify({"error": f"Invalid priority filter. Allowed: {list(VALID_PRIORITIES)}"}), 400
        query += " AND t.priority = ?"
        params.append(priority_filter)

    if category_filter:
        if category_filter not in VALID_CATEGORIES:
            return jsonify({"error": f"Invalid category filter. Allowed: {list(VALID_CATEGORIES)}"}), 400
        query += " AND t.category = ?"
        params.append(category_filter)

    if search_query:
        query += " AND (t.title LIKE ? OR t.description LIKE ? OR rep.name LIKE ?)"
        like_term = f"%{search_query}%"
        params.extend([like_term, like_term, like_term])

    # Sorting: If technician priority queue, urgent tickets first, followed by high, medium, low
    if user["role"] == "technician" or queue_mode == "priority":
        query += """
            ORDER BY 
                CASE t.priority 
                    WHEN 'urgent' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    WHEN 'low' THEN 4 
                    ELSE 5 
                END ASC,
                t.updated_at DESC
        """
    else:
        query += " ORDER BY t.created_at DESC"

    cursor = db.execute(query, params)
    rows = cursor.fetchall()

    tickets = [
        {
            "id": r["id"],
            "title": r["title"],
            "description": r["description"],
            "category": r["category"],
            "priority": r["priority"],
            "predicted_priority": r["predicted_priority"],
            "prediction_confidence": r["prediction_confidence"],
            "status": r["status"],
            "reporter_id": r["reporter_id"],
            "reporter_name": r["reporter_name"],
            "reporter_email": r["reporter_email"],
            "reporter_avatar": r["reporter_avatar"],
            "assignee_id": r["assignee_id"],
            "assignee_name": r["assignee_name"],
            "assignee_email": r["assignee_email"],
            "assignee_avatar": r["assignee_avatar"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]

    return jsonify({"tickets": tickets, "count": len(tickets)}), 200


@app.route("/api/tickets", methods=["POST"])
@login_required
def api_create_ticket():
    """Creates a new support ticket. Restricted to employees only."""
    user = g.user

    # RBAC: Technicians cannot raise tickets; only employees can submit support requests
    if user["role"] != "employee":
        return jsonify({
            "error": "Access denied. Only employees can submit support tickets. Technicians handle ticket triage, assignment, and resolution.",
            "code": "FORBIDDEN"
        }), 403

    data = request.get_json(silent=True) or {}

    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    category = str(data.get("category", "")).strip().lower()
    priority = str(data.get("priority", "")).strip().lower()
    predicted_priority = data.get("predicted_priority")
    prediction_confidence = data.get("prediction_confidence")

    # VALIDATION
    if not title:
        return jsonify({"error": "Ticket title cannot be empty."}), 400
    if len(title) > 140:
        return jsonify({"error": "Ticket title cannot exceed 140 characters."}), 400

    if not description:
        return jsonify({"error": "Ticket description cannot be empty."}), 400
    if len(description) > 5000:
        return jsonify({"error": "Ticket description cannot exceed 5,000 characters."}), 400

    if not category:
        return jsonify({"error": "Category is required."}), 400
    if category not in VALID_CATEGORIES:
        return jsonify({"error": f"Invalid category. Must be one of: {sorted(list(VALID_CATEGORIES))}"}), 400

    # If priority is not provided or invalid, or if client requested prediction
    if not priority or priority not in VALID_PRIORITIES:
        # Run ML model to auto-assign priority
        ml_pred, ml_conf, _ = predict_ticket_priority(title, description)
        priority = ml_pred
        predicted_priority = ml_pred
        prediction_confidence = ml_conf
    else:
        # Validate priority enum
        if priority not in VALID_PRIORITIES:
            return jsonify({"error": f"Invalid priority. Must be one of: {sorted(list(VALID_PRIORITIES))}"}), 400
        # If client didn't supply ML prediction, run it in background to record accuracy
        if not predicted_priority or predicted_priority not in VALID_PRIORITIES:
            ml_pred, ml_conf, _ = predict_ticket_priority(title, description)
            predicted_priority = ml_pred
            prediction_confidence = ml_conf

    now_utc = datetime.now(timezone.utc).isoformat()
    db = get_db()

    # Automatically dispatch/assign ticket to technician with lowest active workload
    auto_tech_id, auto_tech_name = get_auto_assigned_technician(db)

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO tickets (
            title, description, category, priority,
            predicted_priority, prediction_confidence,
            status, reporter_id, assignee_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
    """, (
        title,
        description,
        category,
        priority,
        predicted_priority,
        float(prediction_confidence) if prediction_confidence is not None else None,
        user["id"],
        auto_tech_id,
        now_utc,
        now_utc,
    ))
    ticket_id = cursor.lastrowid

    # Create initial status history entry with auto-assignment notification
    if auto_tech_name:
        history_note = f"Ticket created and automatically assigned to {auto_tech_name} via workload auto-dispatch."
    else:
        history_note = "Ticket created by employee."

    cursor.execute("""
        INSERT INTO status_history (
            ticket_id, changed_by, old_status, new_status, note, created_at
        ) VALUES (?, ?, NULL, 'open', ?, ?)
    """, (ticket_id, user["id"], history_note, now_utc))

    db.commit()

    # Fetch created ticket record with assignee details
    created = db.execute("""
        SELECT 
            t.id, t.title, t.description, t.category, t.priority,
            t.predicted_priority, t.prediction_confidence,
            t.status, t.reporter_id, t.assignee_id, t.created_at, t.updated_at,
            rep.name as reporter_name, rep.email as reporter_email, rep.avatar as reporter_avatar,
            ass.name as assignee_name, ass.email as assignee_email, ass.avatar as assignee_avatar
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE t.id = ?
    """, (ticket_id,)).fetchone()

    success_msg = f"Ticket #{created['id']} created and auto-assigned to {created['assignee_name']}." if created["assignee_name"] else f"Ticket #{created['id']} created successfully."

    return jsonify({
        "message": success_msg,
        "ticket": {
            "id": created["id"],
            "title": created["title"],
            "description": created["description"],
            "category": created["category"],
            "priority": created["priority"],
            "predicted_priority": created["predicted_priority"],
            "prediction_confidence": created["prediction_confidence"],
            "status": created["status"],
            "reporter_id": created["reporter_id"],
            "reporter_name": created["reporter_name"],
            "reporter_email": created["reporter_email"],
            "reporter_avatar": created["reporter_avatar"],
            "assignee_id": created["assignee_id"],
            "assignee_name": created["assignee_name"],
            "assignee_email": created["assignee_email"],
            "assignee_avatar": created["assignee_avatar"],
            "created_at": created["created_at"],
            "updated_at": created["updated_at"],
        }
    }), 201


@app.route("/api/tickets/<int:ticket_id>", methods=["GET"])
@login_required
def api_get_ticket(ticket_id):
    """Retrieves full ticket details. Enforces RBAC permissions."""
    user = g.user
    db = get_db()

    ticket = db.execute("""
        SELECT 
            t.id, t.title, t.description, t.category, t.priority,
            t.predicted_priority, t.prediction_confidence,
            t.status, t.reporter_id, t.assignee_id, t.created_at, t.updated_at,
            rep.name as reporter_name, rep.email as reporter_email, rep.avatar as reporter_avatar,
            ass.name as assignee_name, ass.email as assignee_email, ass.avatar as assignee_avatar
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE t.id = ?
    """, (ticket_id,)).fetchone()

    if not ticket:
        return jsonify({"error": f"Ticket #{ticket_id} not found."}), 404

    # RBAC: Employee cannot access other employees' tickets
    if user["role"] == "employee" and ticket["reporter_id"] != user["id"]:
        return jsonify({
            "error": "Access denied. You may only view tickets you created.",
            "code": "FORBIDDEN"
        }), 403

    return jsonify({
        "ticket": {
            "id": ticket["id"],
            "title": ticket["title"],
            "description": ticket["description"],
            "category": ticket["category"],
            "priority": ticket["priority"],
            "predicted_priority": ticket["predicted_priority"],
            "prediction_confidence": ticket["prediction_confidence"],
            "status": ticket["status"],
            "reporter_id": ticket["reporter_id"],
            "reporter_name": ticket["reporter_name"],
            "reporter_email": ticket["reporter_email"],
            "reporter_avatar": ticket["reporter_avatar"],
            "assignee_id": ticket["assignee_id"],
            "assignee_name": ticket["assignee_name"],
            "assignee_email": ticket["assignee_email"],
            "assignee_avatar": ticket["assignee_avatar"],
            "created_at": ticket["created_at"],
            "updated_at": ticket["updated_at"],
        }
    }), 200


@app.route("/api/tickets/<int:ticket_id>", methods=["PATCH"])
@login_required
def api_update_ticket(ticket_id):
    """
    Updates ticket status, priority, or assigned technician.
    Technician ONLY. Rejects invalid or backwards status transitions.
    """
    user = g.user

    # RBAC: Employee cannot update ticket status or assign technicians
    if user["role"] != "technician":
        return jsonify({
            "error": "Employees cannot update ticket status, assign technicians, or modify priority.",
            "code": "FORBIDDEN"
        }), 403

    db = get_db()
    ticket = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        return jsonify({"error": f"Ticket #{ticket_id} not found."}), 404

    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    new_assignee_id = data.get("assignee_id")
    new_priority = data.get("priority")
    note = str(data.get("note", "")).strip()

    updates = []
    params = []
    current_status = ticket["status"]
    status_changed = False
    assignee_changed = False

    # Status update validation
    if new_status is not None:
        new_status = str(new_status).strip().lower()
        if new_status not in VALID_STATUSES:
            return jsonify({"error": f"Invalid status '{new_status}'. Allowed: {sorted(list(VALID_STATUSES))}"}), 400

        if new_status != current_status:
            # Reject invalid or backwards transitions
            allowed = VALID_STATUS_TRANSITIONS.get(current_status, set())
            if new_status not in allowed:
                return jsonify({
                    "error": f"Invalid status transition from '{current_status}' to '{new_status}'. "
                             f"Allowed flow: open -> in-progress -> resolved -> closed.",
                    "code": "INVALID_STATUS_TRANSITION"
                }), 400

            updates.append("status = ?")
            params.append(new_status)
            status_changed = True

    # Assignee validation
    if "assignee_id" in data:
        if new_assignee_id is None or str(new_assignee_id).strip().lower() in ("", "none", "null", "0"):
            target_assignee = None
        else:
            try:
                target_assignee = int(new_assignee_id)
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid technician ID format."}), 400

            # Verify assignee exists and has role 'technician'
            tech = db.execute("SELECT id, role, name FROM users WHERE id = ?", (target_assignee,)).fetchone()
            if not tech:
                return jsonify({"error": f"Technician ID #{target_assignee} does not exist."}), 400
            if tech["role"] != "technician":
                return jsonify({"error": f"User '{tech['name']}' is not a technician. Only technicians may be assigned tickets."}), 400

        if target_assignee != ticket["assignee_id"]:
            updates.append("assignee_id = ?")
            params.append(target_assignee)
            assignee_changed = True

    # Priority update validation
    if new_priority is not None:
        new_priority = str(new_priority).strip().lower()
        if new_priority not in VALID_PRIORITIES:
            return jsonify({"error": f"Invalid priority '{new_priority}'. Allowed: {sorted(list(VALID_PRIORITIES))}"}), 400
        if new_priority != ticket["priority"]:
            updates.append("priority = ?")
            params.append(new_priority)

    if not updates and not note:
        return jsonify({"message": "No changes requested.", "ticket_id": ticket_id}), 200

    now_utc = datetime.now(timezone.utc).isoformat()
    updates.append("updated_at = ?")
    params.append(now_utc)
    params.append(ticket_id)

    cursor = db.cursor()
    cursor.execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?", params)

    # Record status change or assignment note in status_history
    if status_changed:
        history_note = note if note else f"Status advanced from {current_status} to {new_status}."
        cursor.execute("""
            INSERT INTO status_history (
                ticket_id, changed_by, old_status, new_status, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (ticket_id, user["id"], current_status, new_status, history_note, now_utc))
    elif assignee_changed:
        history_note = note if note else "Technician assignment updated."
        cursor.execute("""
            INSERT INTO status_history (
                ticket_id, changed_by, old_status, new_status, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (ticket_id, user["id"], current_status, current_status, history_note, now_utc))
    elif note:
        cursor.execute("""
            INSERT INTO status_history (
                ticket_id, changed_by, old_status, new_status, note, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (ticket_id, user["id"], current_status, current_status, note, now_utc))

    db.commit()

    # Return refreshed ticket
    refreshed = db.execute("""
        SELECT 
            t.id, t.title, t.description, t.category, t.priority,
            t.predicted_priority, t.prediction_confidence,
            t.status, t.reporter_id, t.assignee_id, t.created_at, t.updated_at,
            rep.name as reporter_name, rep.email as reporter_email, rep.avatar as reporter_avatar,
            ass.name as assignee_name, ass.email as assignee_email, ass.avatar as assignee_avatar
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE t.id = ?
    """, (ticket_id,)).fetchone()

    return jsonify({
        "message": "Ticket updated successfully.",
        "ticket": {
            "id": refreshed["id"],
            "title": refreshed["title"],
            "description": refreshed["description"],
            "category": refreshed["category"],
            "priority": refreshed["priority"],
            "predicted_priority": refreshed["predicted_priority"],
            "prediction_confidence": refreshed["prediction_confidence"],
            "status": refreshed["status"],
            "reporter_id": refreshed["reporter_id"],
            "reporter_name": refreshed["reporter_name"],
            "reporter_email": refreshed["reporter_email"],
            "assignee_id": refreshed["assignee_id"],
            "assignee_name": refreshed["assignee_name"],
            "assignee_email": refreshed["assignee_email"],
            "created_at": refreshed["created_at"],
            "updated_at": refreshed["updated_at"],
        }
    }), 200


@app.route("/api/tickets/<int:ticket_id>/history", methods=["GET"])
@login_required
def api_ticket_history(ticket_id):
    """Retrieves full status and audit history for a ticket."""
    user = g.user
    db = get_db()

    ticket = db.execute("SELECT id, reporter_id FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        return jsonify({"error": f"Ticket #{ticket_id} not found."}), 404

    # RBAC: Employee can only view history of their own ticket
    if user["role"] == "employee" and ticket["reporter_id"] != user["id"]:
        return jsonify({
            "error": "Access denied. You may only view history for your own tickets.",
            "code": "FORBIDDEN"
        }), 403

    history_rows = db.execute("""
        SELECT 
            h.id, h.ticket_id, h.old_status, h.new_status, h.note, h.created_at,
            u.id as changed_by_id, u.name as changed_by_name, u.role as changed_by_role, u.avatar as changed_by_avatar
        FROM status_history h
        JOIN users u ON h.changed_by = u.id
        WHERE h.ticket_id = ?
        ORDER BY h.created_at ASC
    """, (ticket_id,)).fetchall()

    history = [
        {
            "id": r["id"],
            "ticket_id": r["ticket_id"],
            "old_status": r["old_status"],
            "new_status": r["new_status"],
            "note": r["note"],
            "created_at": r["created_at"],
            "changed_by": {
                "id": r["changed_by_id"],
                "name": r["changed_by_name"],
                "role": r["changed_by_role"],
                "avatar": r["changed_by_avatar"],
            },
        }
        for r in history_rows
    ]

    return jsonify({"ticket_id": ticket_id, "history": history, "count": len(history)}), 200


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({"error": "Bad Request. Invalid payload format or values.", "code": "BAD_REQUEST"}), 400


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Resource not found.", "code": "NOT_FOUND"}), 404


@app.errorhandler(405)
def handle_method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this route.", "code": "METHOD_NOT_ALLOWED"}), 405


@app.errorhandler(500)
def handle_server_error(e):
    return jsonify({"error": "An internal server error occurred.", "code": "INTERNAL_SERVER_ERROR"}), 500


# -----------------------------------------------------------------------------
# Application Startup
# -----------------------------------------------------------------------------

# Pre-load ML model and ensure database exists upon module load
ensure_db_initialized()
load_ml_model()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print("  HelpDesk 360 - IT Ticket Management System")
    print(f"  Local URL: http://localhost:{port}")
    print("  Demo Accounts:")
    print("    Technician: arjun@northstar.com / arjun123")
    print("    Employee:   priya@northstar.com / priya123")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=port, debug=True)
