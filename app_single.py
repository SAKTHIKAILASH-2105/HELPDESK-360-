"""
HelpDesk 360 - Standalone Single-File Fallback (app_single.py)
Contains self-contained SQLite initialization, seed data, TF-IDF / heuristic triage,
Flask API backend, and embedded HTML/CSS/JS single-page application.
Run directly with:
    python app_single.py
"""

import os
import sqlite3
import joblib
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import Flask, request, jsonify, session, g, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "helpdesk.db")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "priority_model.pkl")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "helpdesk360-singlefile-secret-key-2026")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

VALID_CATEGORIES = {"hardware", "network", "software", "account-access"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_STATUSES = {"open", "in-progress", "resolved", "closed"}

VALID_STATUS_TRANSITIONS = {
    "open": {"in-progress"},
    "in-progress": {"resolved"},
    "resolved": {"closed"},
    "closed": set(),
}

# In-memory ML pipeline handle
_pipeline = None


def get_ml_pipeline():
    global _pipeline
    if _pipeline is None and os.path.exists(MODEL_PATH):
        try:
            _pipeline = joblib.load(MODEL_PATH)
        except Exception:
            _pipeline = None
    return _pipeline


def predict_priority_logic(title, description):
    text = f"{title or ''} . {description or ''}".strip()
    pipeline = get_ml_pipeline()
    if pipeline is not None and text:
        try:
            pred = pipeline.predict([text])[0]
            probs = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}
            return pred, round(float(prob_dict.get(pred, 0.5)), 2), prob_dict
        except Exception:
            pass

    # Heuristic fallback
    lower = text.lower()
    if any(k in lower for k in ["outage", "ransomware", "down", "critical", "breach", "emergency", "fire", "offline"]):
        return "urgent", 0.90, {"urgent": 0.9, "high": 0.06, "medium": 0.02, "low": 0.02}
    if any(k in lower for k in ["fail", "crash", "blocked", "bsod", "pipeline", "cannot", "broken"]):
        return "high", 0.82, {"urgent": 0.08, "high": 0.82, "medium": 0.06, "low": 0.04}
    if any(k in lower for k in ["mouse pad", "wrist rest", "clean", "cable", "sticker", "how to", "ergonomic"]):
        return "low", 0.88, {"urgent": 0.02, "high": 0.03, "medium": 0.07, "low": 0.88}
    return "medium", 0.65, {"urgent": 0.05, "high": 0.15, "medium": 0.65, "low": 0.15}


# -----------------------------------------------------------------------------
# Database Setup
# -----------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db


@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def get_auto_assigned_technician(db):
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


def init_and_seed_if_empty():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('technician', 'employee')),
        avatar TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        category TEXT NOT NULL CHECK(category IN ('hardware', 'network', 'software', 'account-access')),
        priority TEXT NOT NULL CHECK(priority IN ('low', 'medium', 'high', 'urgent')),
        predicted_priority TEXT,
        prediction_confidence REAL,
        status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'in-progress', 'resolved', 'closed')),
        reporter_id INTEGER NOT NULL,
        assignee_id INTEGER,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (reporter_id) REFERENCES users(id),
        FOREIGN KEY (assignee_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS status_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticket_id INTEGER NOT NULL,
        changed_by INTEGER NOT NULL,
        old_status TEXT,
        new_status TEXT NOT NULL CHECK(new_status IN ('open', 'in-progress', 'resolved', 'closed')),
        note TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (ticket_id) REFERENCES tickets(id),
        FOREIGN KEY (changed_by) REFERENCES users(id)
    );
    """)
    conn.commit()

    # Check if users already seeded
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        now = datetime.now(timezone.utc)
        t_minus = lambda h: (now - timedelta(hours=h)).isoformat()
        users_data = [
            ("Arjun Mehta", "arjun@northstar.com", generate_password_hash("arjun123"), "technician", "AM", t_minus(720)),
            ("Priya Sharma", "priya@northstar.com", generate_password_hash("priya123"), "employee", "PS", t_minus(700)),
            ("Rahul Verma", "rahul@northstar.com", generate_password_hash("rahul123"), "employee", "RV", t_minus(680)),
            ("Nisha Patel", "nisha@northstar.com", generate_password_hash("nisha123"), "employee", "NP", t_minus(650)),
            ("Dev Anand", "dev@northstar.com", generate_password_hash("dev123"), "employee", "DA", t_minus(600)),
        ]
        cursor.executemany("INSERT INTO users (name, email, password_hash, role, avatar, created_at) VALUES (?, ?, ?, ?, ?, ?)", users_data)
        conn.commit()

        cursor.execute("SELECT id, email FROM users")
        umap = {r[1]: r[0] for r in cursor.fetchall()}

        tickets = [
            ("Core production VPN gateway handshake timeout - Engineering blocked", "Cisco AnyConnect VPN gateway is unresponsive to authentication requests.", "network", "urgent", "urgent", 0.94, "open", umap["priya@northstar.com"], None, t_minus(4), t_minus(4)),
            ("Account lockout due to multiple 2FA hardware key sync failures", "Okta account locked after YubiKey 5C NFC firmware handshake failed.", "account-access", "urgent", "urgent", 0.91, "in-progress", umap["nisha@northstar.com"], umap["arjun@northstar.com"], t_minus(6), t_minus(2)),
            ("Docker Desktop daemon crashing on container restart after Windows update", "WSL2 backend kernel hangs when booting docker service.", "software", "high", "high", 0.88, "in-progress", umap["rahul@northstar.com"], umap["arjun@northstar.com"], t_minus(14), t_minus(8)),
            ("MacBook Pro M2 battery draining rapidly and heating up", "Battery health drops from 100% to 20% in under 90 minutes.", "hardware", "medium", "medium", 0.82, "in-progress", umap["dev@northstar.com"], umap["arjun@northstar.com"], t_minus(24), t_minus(12)),
            ("Figma enterprise license seat provisioning for design sprint", "Requesting assignment of an editor license under corporate Figma organization.", "software", "low", "low", 0.92, "resolved", umap["priya@northstar.com"], umap["arjun@northstar.com"], t_minus(48), t_minus(16)),
        ]
        for t in tickets:
            cursor.execute("""
            INSERT INTO tickets (title, description, category, priority, predicted_priority, prediction_confidence, status, reporter_id, assignee_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, t)
            tid = cursor.lastrowid
            cursor.execute("INSERT INTO status_history (ticket_id, changed_by, old_status, new_status, note, created_at) VALUES (?, ?, NULL, ?, 'Seeded entry', ?)", (tid, t[7], t[6], t[9]))
        conn.commit()
    conn.close()


def get_current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    db = get_db()
    return db.execute("SELECT id, name, email, role, avatar, created_at FROM users WHERE id = ?", (uid,)).fetchone()


def login_required(f):
    @wraps(f)
    def dec(*args, **kwargs):
        u = get_current_user()
        if not u:
            return jsonify({"error": "Authentication required.", "code": "UNAUTHENTICATED"}), 401
        g.user = u
        return f(*args, **kwargs)
    return dec


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return jsonify({"error": "Email and password required."}), 400

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    return jsonify({
        "message": "Login successful.",
        "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"], "avatar": user["avatar"]}
    }), 200


@app.route("/api/me", methods=["GET"])
@login_required
def api_me():
    u = g.user
    return jsonify({"user": {"id": u["id"], "name": u["name"], "email": u["email"], "role": u["role"], "avatar": u["avatar"]}}), 200


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"message": "Logged out."}), 200


@app.route("/api/technicians", methods=["GET"])
@login_required
def api_technicians():
    db = get_db()
    techs = db.execute("SELECT id, name, email, avatar FROM users WHERE role = 'technician' ORDER BY name ASC").fetchall()
    return jsonify({"technicians": [{"id": t["id"], "name": t["name"], "email": t["email"], "avatar": t["avatar"]} for t in techs]}), 200


@app.route("/api/predict-priority", methods=["POST"])
@login_required
def api_predict_priority():
    data = request.get_json(silent=True) or {}
    t, d = str(data.get("title", "")).strip(), str(data.get("description", "")).strip()
    if not t and not d:
        return jsonify({"error": "Title or description required."}), 400
    pred, conf, probs = predict_priority_logic(t, d)
    return jsonify({"predicted_priority": pred, "confidence": conf, "probabilities": probs}), 200


@app.route("/api/dashboard", methods=["GET"])
@login_required
def api_dashboard():
    u = g.user
    db = get_db()
    if u["role"] == "technician":
        total = db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        opened = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'open'").fetchone()[0]
        in_prog = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'in-progress'").fetchone()[0]
        resolved = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'resolved'").fetchone()[0]
        closed = db.execute("SELECT COUNT(*) FROM tickets WHERE status = 'closed'").fetchone()[0]
        urgent = db.execute("SELECT COUNT(*) FROM tickets WHERE priority = 'urgent' AND status IN ('open', 'in-progress')").fetchone()[0]
        unassigned = db.execute("SELECT COUNT(*) FROM tickets WHERE assignee_id IS NULL AND status IN ('open', 'in-progress')").fetchone()[0]
        assigned_me = db.execute("SELECT COUNT(*) FROM tickets WHERE assignee_id = ? AND status IN ('open', 'in-progress')", (u["id"],)).fetchone()[0]

        cat_rows = db.execute("SELECT category, COUNT(*) as c FROM tickets GROUP BY category").fetchall()
        categories = {k: 0 for k in VALID_CATEGORIES}
        for r in cat_rows:
            categories[r["category"]] = r["c"]

        recent_rows = db.execute("""
            SELECT h.ticket_id, h.old_status, h.new_status, h.note, h.created_at, u.name as changed_by, t.title as ticket_title
            FROM status_history h
            JOIN users u ON h.changed_by = u.id
            JOIN tickets t ON h.ticket_id = t.id
            ORDER BY h.created_at DESC LIMIT 6
        """).fetchall()

        res_total = resolved + closed
        res_rate = round((res_total / total * 100), 1) if total > 0 else 0.0

        return jsonify({
            "role": "technician",
            "stats": {"total_tickets": total, "open_tickets": opened, "in_progress_tickets": in_prog, "urgent_tickets": urgent, "resolved_tickets": resolved, "closed_tickets": closed, "unassigned_tickets": unassigned, "assigned_to_me": assigned_me, "resolution_rate": res_rate},
            "categories": categories,
            "recent_activity": [dict(r) for r in recent_rows]
        }), 200
    else:
        eid = u["id"]
        total = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ?", (eid,)).fetchone()[0]
        opened = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'open'", (eid,)).fetchone()[0]
        in_prog = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'in-progress'", (eid,)).fetchone()[0]
        resolved = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'resolved'", (eid,)).fetchone()[0]
        closed = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND status = 'closed'", (eid,)).fetchone()[0]
        urgent = db.execute("SELECT COUNT(*) FROM tickets WHERE reporter_id = ? AND priority = 'urgent'", (eid,)).fetchone()[0]

        cat_rows = db.execute("SELECT category, COUNT(*) as c FROM tickets WHERE reporter_id = ? GROUP BY category", (eid,)).fetchall()
        categories = {k: 0 for k in VALID_CATEGORIES}
        for r in cat_rows:
            categories[r["category"]] = r["c"]

        recent_rows = db.execute("""
            SELECT h.ticket_id, h.old_status, h.new_status, h.note, h.created_at, u.name as changed_by, t.title as ticket_title
            FROM status_history h
            JOIN users u ON h.changed_by = u.id
            JOIN tickets t ON h.ticket_id = t.id
            WHERE t.reporter_id = ?
            ORDER BY h.created_at DESC LIMIT 5
        """, (eid,)).fetchall()

        return jsonify({
            "role": "employee",
            "stats": {"total_tickets": total, "open_tickets": opened, "in_progress_tickets": in_prog, "urgent_tickets": urgent, "resolved_tickets": resolved, "closed_tickets": closed},
            "categories": categories,
            "recent_activity": [dict(r) for r in recent_rows]
        }), 200


@app.route("/api/tickets", methods=["GET"])
@login_required
def api_get_tickets():
    u = g.user
    db = get_db()
    st = request.args.get("status", "").strip().lower()
    pr = request.args.get("priority", "").strip().lower()
    cat = request.args.get("category", "").strip().lower()
    asg = request.args.get("assignee_id", "").strip()
    sch = request.args.get("search", "").strip()

    sql = """
        SELECT t.*, rep.name as reporter_name, rep.email as reporter_email,
               ass.name as assignee_name, ass.email as assignee_email
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE 1=1
    """
    params = []
    if u["role"] == "employee":
        sql += " AND t.reporter_id = ?"
        params.append(u["id"])
    elif asg:
        if asg == "unassigned":
            sql += " AND t.assignee_id IS NULL"
        else:
            try:
                sql += " AND t.assignee_id = ?"
                params.append(int(asg))
            except ValueError:
                return jsonify({"error": "Invalid assignee filter"}), 400

    if st:
        if st not in VALID_STATUSES:
            return jsonify({"error": "Invalid status"}), 400
        sql += " AND t.status = ?"
        params.append(st)
    if pr:
        if pr not in VALID_PRIORITIES:
            return jsonify({"error": "Invalid priority"}), 400
        sql += " AND t.priority = ?"
        params.append(pr)
    if cat:
        if cat not in VALID_CATEGORIES:
            return jsonify({"error": "Invalid category"}), 400
        sql += " AND t.category = ?"
        params.append(cat)
    if sch:
        sql += " AND (t.title LIKE ? OR t.description LIKE ? OR rep.name LIKE ?)"
        s = f"%{sch}%"
        params.extend([s, s, s])

    sql += """
        ORDER BY 
            CASE t.priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END ASC,
            t.updated_at DESC
    """
    rows = db.execute(sql, params).fetchall()
    return jsonify({"tickets": [dict(r) for r in rows], "count": len(rows)}), 200


@app.route("/api/tickets", methods=["POST"])
@login_required
def api_create_ticket():
    u = g.user
    if u["role"] != "employee":
        return jsonify({"error": "Only employees can submit support tickets.", "code": "FORBIDDEN"}), 403
    data = request.get_json(silent=True) or {}
    title = str(data.get("title", "")).strip()
    description = str(data.get("description", "")).strip()
    category = str(data.get("category", "")).strip().lower()
    priority = str(data.get("priority", "")).strip().lower()
    pred_pr = data.get("predicted_priority")
    conf = data.get("prediction_confidence")

    if not title or len(title) > 140:
        return jsonify({"error": "Title must be between 1 and 140 characters."}), 400
    if not description or len(description) > 5000:
        return jsonify({"error": "Description must be between 1 and 5,000 characters."}), 400
    if category not in VALID_CATEGORIES:
        return jsonify({"error": f"Invalid category. Allowed: {list(VALID_CATEGORIES)}"}), 400

    if not priority or priority not in VALID_PRIORITIES:
        pred, cf, _ = predict_priority_logic(title, description)
        priority, pred_pr, conf = pred, pred, cf
    elif not pred_pr or pred_pr not in VALID_PRIORITIES:
        pred, cf, _ = predict_priority_logic(title, description)
        pred_pr, conf = pred, cf

    now_utc = datetime.now(timezone.utc).isoformat()
    db = get_db()
    auto_id, auto_name = get_auto_assigned_technician(db)
    cur = db.cursor()
    cur.execute("""
        INSERT INTO tickets (title, description, category, priority, predicted_priority, prediction_confidence, status, reporter_id, assignee_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
    """, (title, description, category, priority, pred_pr, float(conf) if conf else None, u["id"], auto_id, now_utc, now_utc))
    tid = cur.lastrowid
    note = f"Ticket created and automatically assigned to {auto_name}." if auto_name else "Ticket created."
    cur.execute("INSERT INTO status_history (ticket_id, changed_by, old_status, new_status, note, created_at) VALUES (?, ?, NULL, 'open', ?, ?)", (tid, u["id"], note, now_utc))
    db.commit()

    created = db.execute("SELECT t.*, rep.name as reporter_name, ass.name as assignee_name FROM tickets t JOIN users rep ON t.reporter_id = rep.id LEFT JOIN users ass ON t.assignee_id = ass.id WHERE t.id = ?", (tid,)).fetchone()
    return jsonify({"message": f"Ticket #{tid} created and auto-assigned to {created['assignee_name']}." if created['assignee_name'] else "Ticket created.", "ticket": dict(created)}), 201


@app.route("/api/tickets/<int:ticket_id>", methods=["GET"])
@login_required
def api_get_ticket(ticket_id):
    u = g.user
    db = get_db()
    ticket = db.execute("""
        SELECT t.*, rep.name as reporter_name, rep.email as reporter_email, ass.name as assignee_name, ass.email as assignee_email
        FROM tickets t
        JOIN users rep ON t.reporter_id = rep.id
        LEFT JOIN users ass ON t.assignee_id = ass.id
        WHERE t.id = ?
    """, (ticket_id,)).fetchone()
    if not ticket:
        return jsonify({"error": f"Ticket #{ticket_id} not found."}), 404
    if u["role"] == "employee" and ticket["reporter_id"] != u["id"]:
        return jsonify({"error": "Access denied.", "code": "FORBIDDEN"}), 403
    return jsonify({"ticket": dict(ticket)}), 200


@app.route("/api/tickets/<int:ticket_id>", methods=["PATCH"])
@login_required
def api_update_ticket(ticket_id):
    u = g.user
    if u["role"] != "technician":
        return jsonify({"error": "Only technicians can update ticket status or assignment.", "code": "FORBIDDEN"}), 403

    db = get_db()
    ticket = db.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not ticket:
        return jsonify({"error": f"Ticket #{ticket_id} not found."}), 404

    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    new_assignee = data.get("assignee_id")
    new_priority = data.get("priority")
    note = str(data.get("note", "")).strip()

    updates, params = [], []
    curr_status = ticket["status"]
    status_changed = False

    if new_status:
        new_status = str(new_status).strip().lower()
        if new_status not in VALID_STATUSES:
            return jsonify({"error": "Invalid status."}), 400
        if new_status != curr_status:
            if new_status not in VALID_STATUS_TRANSITIONS.get(curr_status, set()):
                return jsonify({"error": f"Invalid transition from {curr_status} to {new_status}."}), 400
            updates.append("status = ?")
            params.append(new_status)
            status_changed = True

    if "assignee_id" in data:
        target_asg = None if new_assignee in (None, "", "null", 0) else int(new_assignee)
        if target_asg:
            tech = db.execute("SELECT id, role FROM users WHERE id = ?", (target_asg,)).fetchone()
            if not tech or tech["role"] != "technician":
                return jsonify({"error": "Assignee must be a registered technician."}), 400
        if target_asg != ticket["assignee_id"]:
            updates.append("assignee_id = ?")
            params.append(target_asg)

    if new_priority:
        new_priority = str(new_priority).strip().lower()
        if new_priority in VALID_PRIORITIES and new_priority != ticket["priority"]:
            updates.append("priority = ?")
            params.append(new_priority)

    now_utc = datetime.now(timezone.utc).isoformat()
    if updates:
        updates.append("updated_at = ?")
        params.extend([now_utc, ticket_id])
        db.execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?", params)

    if status_changed:
        db.execute("INSERT INTO status_history (ticket_id, changed_by, old_status, new_status, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (ticket_id, u["id"], curr_status, new_status, note or f"Status changed to {new_status}", now_utc))
    elif note:
        db.execute("INSERT INTO status_history (ticket_id, changed_by, old_status, new_status, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (ticket_id, u["id"], curr_status, curr_status, note, now_utc))

    db.commit()
    refreshed = db.execute("SELECT t.*, rep.name as reporter_name, ass.name as assignee_name FROM tickets t JOIN users rep ON t.reporter_id = rep.id LEFT JOIN users ass ON t.assignee_id = ass.id WHERE t.id = ?", (ticket_id,)).fetchone()
    return jsonify({"message": "Ticket updated.", "ticket": dict(refreshed)}), 200


@app.route("/api/tickets/<int:ticket_id>/history", methods=["GET"])
@login_required
def api_ticket_history(ticket_id):
    u = g.user
    db = get_db()
    t = db.execute("SELECT reporter_id FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    if not t:
        return jsonify({"error": "Ticket not found."}), 404
    if u["role"] == "employee" and t["reporter_id"] != u["id"]:
        return jsonify({"error": "Access denied.", "code": "FORBIDDEN"}), 403

    rows = db.execute("""
        SELECT h.*, u.name as changed_by_name, u.role as changed_by_role
        FROM status_history h JOIN users u ON h.changed_by = u.id
        WHERE h.ticket_id = ? ORDER BY h.created_at ASC
    """, (ticket_id,)).fetchall()
    return jsonify({"history": [
        {"id": r["id"], "old_status": r["old_status"], "new_status": r["new_status"], "note": r["note"], "created_at": r["created_at"], "changed_by": {"name": r["changed_by_name"], "role": r["changed_by_role"]}}
        for r in rows
    ]}), 200


# Fallback HTML Template serving frontend files if available, or direct minimal fallback
@app.route("/")
def index():
    tpl_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(tpl_path):
        with open(tpl_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>HelpDesk 360 Fallback Running</h1><p>Please run the modular version with <code>python app.py</code></p>"


init_and_seed_if_empty()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] HelpDesk 360 Single-File Fallback started at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
