# HelpDesk 360 - Enterprise IT Support Ticket Management System

**HelpDesk 360** is an enterprise-grade, full-stack IT operations command center and support ticket management application. Designed for speed, strict role-based access control (RBAC), and automated triage, HelpDesk 360 incorporates a machine learning classifier to predict ticket urgency upon submission, enforces state transition flows, and provides real-time operational metrics powered entirely by a local SQLite persistence layer.

---

## 1. Project Overview

In an enterprise IT organization, incident response time is critical. HelpDesk 360 streamlines the lifecycle from **Employee Report ➔ Automated ML Triage ➔ Technician Assignment ➔ Active Resolution ➔ Audit Logging**.

The application runs 100% locally on Windows (`localhost`), without any external cloud providers, Firebase, Supabase, third-party APIs, or paid subscriptions.

---

## 2. Key Features

- **Automated ML Priority Advisor:** Real-time text classification using scikit-learn (`TF-IDF Vectorizer + Multinomial Logistic Regression`) trained on realistic IT incident vocabulary. Employees can accept or manually adjust recommendations.
- **Strict Role-Based Access Control (RBAC):**
  - **Employee:** Can submit tickets, view *only* their own tickets, monitor assigned technicians, and track status history. Cannot modify status, assign tickets, or view other employees' tickets (enforced via HTTP 403 Forbidden on backend).
  - **Technician:** Full visibility into all company tickets, access to the Urgent Priority Queue, ability to assign technicians, advance status, and view real-time operations dashboards.
- **Enforced Lifecycle State Machine:**
  - Strict transition path: `open` ➔ `in-progress` ➔ `resolved` ➔ `closed`.
  - Rejects skipping steps or backwards transitions on the backend.
- **Comprehensive Audit Trail:** Every status advance and technician reassignment creates an immutable record in `status_history` with changer identity, UTC timestamp, and optional technician notes.
- **Pure Offline SVG Operations Analytics:** Interactive category breakdown and KPI metric cards calculated directly from SQLite queries—no CDN dependencies required.
- **Zero-Friction Demo Testing:** Built-in demo account switcher both on the login page and topbar for instantaneous role toggling during evaluations.
- **Standalone Fallback Mode:** Includes `app_single.py` for one-file zero-config execution.

---

## 3. Project Structure

```
helpdesk-360/
├── app.py                      # Main modular Flask backend & REST API
├── app_single.py               # Standalone single-file fallback version
├── requirements.txt            # Python dependencies
├── README.md                   # Complete architectural and operational manual
├── .env.example                # Sample environment configuration
├── .gitignore                  # Git ignore rules for venv, caches, and DB journals
├── test_app.py                 # Automated integration and RBAC test suite
├── database/
│   ├── helpdesk.db             # SQLite database file (created on seed/run)
│   └── seed.py                 # Database schema initialization and seed script
├── ml/
│   ├── training_data.csv       # IT support ticket training dataset (130+ cases)
│   ├── train_model.py          # Model training, evaluation, and export script
│   └── priority_model.pkl      # Serialized scikit-learn classification pipeline
├── templates/
│   └── index.html              # Modern IT operations command center SPA
├── static/
│   ├── styles.css              # Dark sidebar + light dashboard theme, SVG styles
│   └── app.js                  # Frontend SPA controller, state management, live ML
└── reports/
    └── model_report.md         # Classification report, confusion matrix, accuracy metrics
```

---

## 4. Python Version Requirement

- **Python 3.10+** (Tested and verified on **Python 3.11.9** on Windows)

---

## 5. Installation Steps

1. Open Windows Command Prompt (`cmd.exe`) or PowerShell.
2. Navigate to the `helpdesk-360` directory.
3. Create a dedicated virtual environment.
4. Activate the virtual environment.
5. Install Python dependencies from `requirements.txt`.

---

## 6. Windows CMD Commands (Exact Run Instructions)

Run the following commands in order:

```cmd
cd helpdesk-360
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python ml\train_model.py
python database\seed.py
python app.py
```

---

## 7. How to Train the ML Model

To train (or retrain) the scikit-learn priority classification pipeline on `ml/training_data.csv`:

```cmd
python ml\train_model.py
```

### What this does:
1. Loads 135+ realistic IT support incidents categorized into `urgent`, `high`, `medium`, and `low`.
2. Evaluates performance using an 80/20 stratified split and 5-fold cross-validation.
3. Fits a `TfidfVectorizer(ngram_range=(1,2))` + `LogisticRegression(class_weight='balanced')` pipeline.
4. Serializes the final trained model to `ml/priority_model.pkl`.
5. Generates a comprehensive evaluation report in `reports/model_report.md`.

> [!NOTE]
> The included training data in `ml/training_data.csv` represents simulated enterprise IT support scenarios. For production enterprise deployments, this file should be replaced with historical ticket resolutions exported from your internal service desk.

---

## 8. How to Seed or Reset the Database

To initialize the SQLite database schema and load fresh demo data:

```cmd
python database\seed.py
```

This creates:
- 3 Technician accounts (`Arjun Mehta`, `Kavya Iyer`, `Rohit Sen`)
- 4 Employee accounts (`Priya Sharma`, `Rahul Verma`, `Nisha Patel`, `Dev Anand`)
- 8 Sample tickets covering all 4 categories, all 4 priorities, and all 4 lifecycle statuses
- 16 Status history audit entries documenting the progression of existing tickets

---

## 9. How to Run the Backend

Start the modular Flask production server:

```cmd
python app.py
```

Or run the one-file fallback version:

```cmd
python app_single.py
```

The server binds to `http://127.0.0.1:5000` (or the port specified via `PORT` environment variable).

---

## 10. Browser URL

Open your web browser and navigate to:

```
http://localhost:5000
```

---

## 11. Demo Login Accounts

All accounts use pre-hashed passwords stored in SQLite:

| Role | Name | Email | Password | Permissions |
| :--- | :--- | :--- | :--- | :--- |
| **Technician 1** | Arjun Mehta | `arjun@northstar.com` | `arjun123` | Full queue visibility, assign techs, advance status, triage, command center |
| **Technician 2** | Kavya Iyer | `kavya@northstar.com` | `kavya123` | Full queue visibility, assign techs, advance status, triage, command center |
| **Technician 3** | Rohit Sen | `rohit@northstar.com` | `rohit123` | Full queue visibility, assign techs, advance status, triage, command center |
| **Employee 1** | Priya Sharma | `priya@northstar.com` | `priya123` | Submit tickets (auto-assigned to technician), view own tickets only |
| **Employee 2** | Rahul Verma | `rahul@northstar.com` | `rahul123` | Submit tickets (auto-assigned to technician), view own tickets only |
| **Employee 3** | Nisha Patel | `nisha@northstar.com` | `nisha123` | Submit tickets (auto-assigned to technician), view own tickets only |
| **Employee 4** | Dev Anand | `dev@northstar.com` | `dev123` | Submit tickets (auto-assigned to technician), view own tickets only |

> [!TIP]
> You can click on any of the **Quick Demo Accounts** chips on the login screen or topbar to instantly populate credentials and switch roles!

---

## 12. API Documentation

All API responses are formatted in JSON with explicit HTTP status codes.

### Authentication Endpoints
- `POST /api/login`
  - **Body:** `{ "email": "...", "password": "..." }`
  - **Returns:** `{ "message": "Login successful.", "user": { ... } }` (HTTP 200) or error (HTTP 401).
- `GET /api/me`
  - **Returns:** Current authenticated user profile (HTTP 200) or HTTP 401.
- `POST /api/logout`
  - **Clears:** Session cookie (HTTP 200).

### Dashboard & Analytics
- `GET /api/dashboard`
  - **Role-Aware:**
    - If Technician: Returns company-wide total tickets, open, in-progress, urgent, resolved, closed, unassigned, assigned to me, resolution rate, category breakdown, and recent activity.
    - If Employee: Returns metrics strictly scoped to tickets created by the authenticated user.

### Technician Directory
- `GET /api/technicians`
  - **Returns:** List of all registered technicians (`id`, `name`, `email`, `avatar`) for assignment dropdowns.

### Machine Learning Inference
- `POST /api/predict-priority`
  - **Body:** `{ "title": "...", "description": "..." }`
  - **Returns:** `{ "predicted_priority": "urgent", "confidence": 0.94, "probabilities": { ... } }`

### Ticket Management
- `GET /api/tickets`
  - **Query Params:** `status`, `priority`, `category`, `assignee_id`, `search`, `queue`
  - **Enforcement:** Employees receive *only* tickets where `reporter_id == current_user.id`.
- `POST /api/tickets`
  - **Body:** `{ "title": "...", "description": "...", "category": "...", "priority": "...", "predicted_priority": "...", "prediction_confidence": ... }`
  - **Validations:** Title (1-140 chars), Description (1-5000 chars), Category enum, Priority enum.
  - **Returns:** Created ticket with initial `status_history` entry (HTTP 201).
- `GET /api/tickets/<id>`
  - **Returns:** Full ticket details. Returns HTTP 403 Forbidden if an employee requests another employee's ticket.
- `PATCH /api/tickets/<id>`
  - **Technician Only.** Returns HTTP 403 if called by an employee.
  - **Body:** `{ "status": "...", "assignee_id": ..., "priority": "...", "note": "..." }`
  - **Status Flow Enforcement:** Only allows `open` ➔ `in-progress` ➔ `resolved` ➔ `closed`. Rejects invalid or backwards transitions with HTTP 400.
  - **Assignee Enforcement:** Target must be a valid user with role `'technician'`.
- `GET /api/tickets/<id>/history`
  - **Returns:** Ordered audit trail of all status transitions, notes, and changer identities.

---

## 13. Database Schema Explanation

Implemented using SQLite with Foreign Key constraints and indexed search fields:

```sql
-- 1. Users Table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('technician', 'employee')),
    avatar TEXT,
    created_at TEXT NOT NULL
);

-- 2. Tickets Table
CREATE TABLE tickets (
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
    FOREIGN KEY (reporter_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (assignee_id) REFERENCES users(id) ON DELETE SET NULL
);

-- 3. Status History Table (Audit Trail)
CREATE TABLE status_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    changed_by INTEGER NOT NULL,
    old_status TEXT,
    new_status TEXT NOT NULL CHECK(new_status IN ('open', 'in-progress', 'resolved', 'closed')),
    note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id) ON DELETE CASCADE,
    FOREIGN KEY (changed_by) REFERENCES users(id) ON DELETE CASCADE
);
```

---

## 14. ML Model Explanation

- **Vectorizer:** `TfidfVectorizer` configured with unigrams and bigrams (`ngram_range=(1, 2)`), sublinear term frequency scaling, and English stop words filtering.
- **Classifier:** `LogisticRegression` with `class_weight='balanced'` and `solver='lbfgs'`. Logistic Regression yields well-calibrated posterior probabilities via the softmax function, giving employees transparent confidence percentages (e.g., *94% confidence*).
- **Inference Speed:** < 5 ms on a standard single CPU core.
- **Heuristic Fallback:** If the pickle file is missing or uncompiled, a keyword-based rule heuristic guarantees zero downtime.

---

## 15. Testing Checklist

Run the automated integration test suite:

```cmd
python test_app.py
```

The test suite validates:
- [x] Technician login and password hashing verification
- [x] Rejection of invalid credentials (HTTP 401)
- [x] Machine learning prediction for critical vs minor cases
- [x] Employee ticket isolation (employee cannot view or query other employees' tickets)
- [x] Rejection of employee status modification attempts (HTTP 403)
- [x] Ticket creation validation rules (title <= 140 chars, description <= 5000 chars, valid enums)
- [x] State transition enforcement (`open` ➔ `in-progress` ➔ `resolved` ➔ `closed`)
- [x] Rejection of invalid or backwards status jumps (HTTP 400)
- [x] Technician assignment validation (cannot assign to non-technicians)
- [x] Real-time database metric accuracy

---

## 16. Known Limitations

- **Single Database File:** Uses SQLite, which is ideal for single-node local execution, desktop deployments, and small-to-medium teams. For high-concurrency multi-region deployments, migration to PostgreSQL with connection pooling is recommended.
- **Single Process In-Memory Model:** The ML pipeline is loaded into memory during worker startup. While fast and lightweight (~250 KB), model retraining requires a process reload to refresh cached weights.

---

## 17. Future Improvements

1. **Email/Webhook Notifications:** Outbound SMTP or Slack notifications when tickets are escalated to Urgent.
2. **File Attachments:** Support for uploading diagnostic screenshots and crash log `.txt` files saved to a local storage directory.
3. **SLA Breach Warnings:** Visual timer badges highlighting open urgent tickets exceeding 2-hour response windows.
4. **Active Learning Feedback Loop:** Automatic retraining triggers when technicians override ML-suggested priority.
