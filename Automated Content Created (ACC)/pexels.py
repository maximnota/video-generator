import requests

class PexelsAPI:
    def __init__(self, API_KEY):
        self.api_key = API_KEY
        self.base_url = "https://api.pexels.com/videos/"

    def searchVideo(self, query, per_page=10):
        headers = {'Authorization': self.api_key}
        params = {'query': query, 'per_page': per_page}
        response = requests.get(self.base_url + 'search', headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            return [video['video_files'][0]['link'] for video in data['videos']]
        else:
            print("Failed to fetch videos", response.status_code)
            return []

    def downloadVideo(self, url, file_name):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Video downloaded successfully as {file_name}")
        else:
            print("Failed to download video", response.status_code)
