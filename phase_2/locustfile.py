from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)  # Thời gian chờ giữa các requests

    @task
    def make_prediction(self):
        headers = {
            "Content-Type": "application/json"
        }
        payload_file = "problem_1/data/curl/phase-2/prob-1/payload-1.json"  # Điều chỉnh đường dẫn tới file payload-1.json
        with open(payload_file, "r") as f:
            payload_data = f.read()

        self.client.post("/phase-2/prob-1/predict", headers=headers, data=payload_data)
