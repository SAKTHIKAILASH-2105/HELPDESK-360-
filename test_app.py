"""
Automated Integration and Security Test Suite for HelpDesk 360
Tests Authentication, RBAC Permissions, Status Transitions, Validations, and ML Inference.
"""

import sys
import os
import json
import unittest

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from app import app, get_db
from database.seed import seed_database


class HelpDesk360TestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fresh database seed before tests
        seed_database()
        cls.client = app.test_client()

    def test_01_login_technician_and_me(self):
        # 1. Login with technician credentials
        res = self.client.post("/api/login", json={
            "email": "arjun@northstar.com",
            "password": "arjun123"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["user"]["role"], "technician")
        self.assertEqual(data["user"]["email"], "arjun@northstar.com")
        self.assertNotIn("password_hash", data["user"])

        # 2. Check /api/me
        me_res = self.client.get("/api/me")
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.get_json()
        self.assertEqual(me_data["user"]["name"], "Arjun Mehta")

    def test_02_login_invalid_password(self):
        res = self.client.post("/api/login", json={
            "email": "arjun@northstar.com",
            "password": "wrongpassword"
        })
        self.assertEqual(res.status_code, 401)
        data = res.get_json()
        self.assertIn("error", data)

    def test_03_ml_prediction_endpoint(self):
        # Login first
        self.client.post("/api/login", json={"email": "arjun@northstar.com", "password": "arjun123"})
        
        # Test urgent prediction
        res = self.client.post("/api/predict-priority", json={
            "title": "Production PostgreSQL cluster crashed",
            "description": "All customer transactions failing with 500 error, urgent service restoration needed"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["predicted_priority"], "urgent")
        self.assertGreater(data["confidence"], 0.5)

        # Test low prediction
        res2 = self.client.post("/api/predict-priority", json={
            "title": "Request gel wrist rest mouse pad",
            "description": "Would like an ergonomic mouse pad for comfort"
        })
        self.assertEqual(res2.status_code, 200)
        data2 = res2.get_json()
        self.assertEqual(data2["predicted_priority"], "low")

    def test_04_employee_ticket_isolation_rbac(self):
        # Login as Priya Sharma (Employee)
        self.client.post("/api/login", json={"email": "priya@northstar.com", "password": "priya123"})

        # Get Priya's tickets
        res = self.client.get("/api/tickets")
        self.assertEqual(res.status_code, 200)
        priya_tickets = res.get_json()["tickets"]

        # Ensure all returned tickets belong to Priya
        for t in priya_tickets:
            self.assertEqual(t["reporter_email"], "priya@northstar.com")

        # Attempt to access another employee's ticket (Ticket #3 is Rahul's Docker issue)
        # First verify ticket #3 exists and belongs to someone else
        with app.app_context():
            db = get_db()
            rahul_ticket = db.execute("SELECT id, reporter_id FROM tickets WHERE id = 3").fetchone()
            self.assertIsNotNone(rahul_ticket)

        # Priya tries to GET ticket #3
        access_res = self.client.get("/api/tickets/3")
        self.assertEqual(access_res.status_code, 403)
        self.assertIn("Access denied", access_res.get_json()["error"])

        # Priya tries to PATCH ticket status directly (Forbidden for employee)
        patch_res = self.client.patch("/api/tickets/1", json={"status": "resolved"})
        self.assertEqual(patch_res.status_code, 403)
        self.assertIn("Employees cannot update ticket status", patch_res.get_json()["error"])

    def test_05_ticket_creation_validation(self):
        # Login as Priya
        self.client.post("/api/login", json={"email": "priya@northstar.com", "password": "priya123"})

        # Empty title rejected
        r1 = self.client.post("/api/tickets", json={
            "title": "   ",
            "description": "Some description",
            "category": "software"
        })
        self.assertEqual(r1.status_code, 400)

        # Empty description rejected
        r2 = self.client.post("/api/tickets", json={
            "title": "Valid title",
            "description": "   ",
            "category": "software"
        })
        self.assertEqual(r2.status_code, 400)

        # Invalid category rejected
        r3 = self.client.post("/api/tickets", json={
            "title": "Valid title",
            "description": "Valid description",
            "category": "invalid-category-xyz"
        })
        self.assertEqual(r3.status_code, 400)

        # Title > 140 chars rejected
        r4 = self.client.post("/api/tickets", json={
            "title": "A" * 141,
            "description": "Valid description",
            "category": "software"
        })
        self.assertEqual(r4.status_code, 400)

        # Valid creation succeeds
        r5 = self.client.post("/api/tickets", json={
            "title": "Kubernetes staging deployment timeout",
            "description": "Pods are stuck in Pending state due to volume attach timeout on node worker-02.",
            "category": "software",
            "priority": "high"
        })
        self.assertEqual(r5.status_code, 201)
        created_ticket = r5.get_json()["ticket"]
        self.assertEqual(created_ticket["status"], "open")
        self.assertEqual(created_ticket["priority"], "high")
        self.assertIsNotNone(created_ticket["predicted_priority"])
        # Verify automatic technician assignment via workload balancer
        self.assertIsNotNone(created_ticket["assignee_id"])
        self.assertEqual(created_ticket["assignee_name"], "Arjun Mehta")

    def test_06_technician_status_flow_enforcement(self):
        # 1. Login as Arjun (Technician) and verify technician CANNOT create tickets
        self.client.post("/api/login", json={"email": "arjun@northstar.com", "password": "arjun123"})
        tech_create_attempt = self.client.post("/api/tickets", json={
            "title": "Technician trying to raise ticket",
            "description": "Should be rejected because only employees raise tickets",
            "category": "network",
            "priority": "medium"
        })
        self.assertEqual(tech_create_attempt.status_code, 403)
        self.assertIn("Only employees can submit support tickets", tech_create_attempt.get_json()["error"])

        # 2. Login as Priya (Employee) to create a test ticket for technician triage
        self.client.post("/api/login", json={"email": "priya@northstar.com", "password": "priya123"})
        create_res = self.client.post("/api/tickets", json={
            "title": "Test state machine ticket",
            "description": "Testing valid and invalid transitions",
            "category": "network",
            "priority": "medium"
        })
        self.assertEqual(create_res.status_code, 201)
        ticket_id = create_res.get_json()["ticket"]["id"]

        # 3. Switch back to Arjun (Technician) to perform triage and status transitions
        self.client.post("/api/login", json={"email": "arjun@northstar.com", "password": "arjun123"})

        # Invalid jump: open -> resolved (must fail)
        jump_res = self.client.patch(f"/api/tickets/{ticket_id}", json={"status": "resolved"})
        self.assertEqual(jump_res.status_code, 400)
        self.assertIn("Invalid status transition", jump_res.get_json()["error"])

        # 2. Valid: open -> in-progress
        step1 = self.client.patch(f"/api/tickets/{ticket_id}", json={
            "status": "in-progress",
            "note": "Arjun started investigation"
        })
        self.assertEqual(step1.status_code, 200)
        self.assertEqual(step1.get_json()["ticket"]["status"], "in-progress")

        # 3. Invalid backwards: in-progress -> open (must fail)
        back_res = self.client.patch(f"/api/tickets/{ticket_id}", json={"status": "open"})
        self.assertEqual(back_res.status_code, 400)

        # 4. Valid: in-progress -> resolved
        step2 = self.client.patch(f"/api/tickets/{ticket_id}", json={
            "status": "resolved",
            "note": "Root cause identified and resolved"
        })
        self.assertEqual(step2.status_code, 200)
        self.assertEqual(step2.get_json()["ticket"]["status"], "resolved")

        # 5. Valid: resolved -> closed
        step3 = self.client.patch(f"/api/tickets/{ticket_id}", json={
            "status": "closed",
            "note": "User confirmed fix"
        })
        self.assertEqual(step3.status_code, 200)
        self.assertEqual(step3.get_json()["ticket"]["status"], "closed")

        # 6. Invalid transition out of terminal state: closed -> in-progress (must fail)
        terminal_res = self.client.patch(f"/api/tickets/{ticket_id}", json={"status": "in-progress"})
        self.assertEqual(terminal_res.status_code, 400)

        # Check status history records
        hist_res = self.client.get(f"/api/tickets/{ticket_id}/history")
        self.assertEqual(hist_res.status_code, 200)
        history = hist_res.get_json()["history"]
        self.assertGreaterEqual(len(history), 4)

    def test_07_technician_assignment_validation(self):
        # Login as Arjun
        self.client.post("/api/login", json={"email": "arjun@northstar.com", "password": "arjun123"})

        # Try to assign to Priya (who is an employee, ID 4, not a technician) -> must reject
        bad_assign = self.client.patch("/api/tickets/1", json={"assignee_id": 4})
        self.assertEqual(bad_assign.status_code, 400)
        self.assertIn("not a technician", bad_assign.get_json()["error"])

        # Assign to valid technician (Arjun, ID 1 or Kavya, ID 2) -> must succeed
        good_assign = self.client.patch("/api/tickets/1", json={"assignee_id": 1, "note": "Assigned to Arjun"})
        self.assertEqual(good_assign.status_code, 200)
        self.assertEqual(good_assign.get_json()["ticket"]["assignee_id"], 1)

    def test_08_dashboard_metrics_accuracy(self):
        # Login as Arjun
        self.client.post("/api/login", json={"email": "arjun@northstar.com", "password": "arjun123"})
        dash_res = self.client.get("/api/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        data = dash_res.get_json()
        self.assertEqual(data["role"], "technician")
        stats = data["stats"]
        self.assertIn("total_tickets", stats)
        self.assertIn("open_tickets", stats)
        self.assertIn("urgent_tickets", stats)
        self.assertIn("resolution_rate", stats)
        self.assertGreater(stats["total_tickets"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
