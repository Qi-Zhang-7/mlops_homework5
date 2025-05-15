from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

# def test_query_endpoint():
#     response = client.post("/similar_responses", json={"question": "What is the capital of France?"})
#     assert response.status_code == 200
#     assert response.json() == {"answers": ["These are test responses"]}
def test_query_endpoint():
    response = client.post("/similar_responses", json={"question": "What is the capital of France?"})
    
    # Check that the request was successful
    assert response.status_code == 200
    
    # Check the structure of the response
    json_data = response.json()
    assert "answers" in json_data
    assert isinstance(json_data["answers"], list)
    assert "text" in json_data["answers"][0]

    # Optionally check that the response makes sense
    print("Test output preview:", json_data["answers"][0])