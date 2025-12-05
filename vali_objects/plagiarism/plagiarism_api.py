import requests
#TODO this function needs to use an api key for fetching plagiarism scores
def get_plagiarism_elimination_scores(api_base_url="http://localhost:5000"):
    """
    Get elimination scores from the plagiarism API

    Args:
        api_base_url (str): Base URL of the API server

    Returns:
        list: List of elimination scores
    """
    try:
        response = requests.get(f"{api_base_url}/elimination_scores")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elimination scores: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []