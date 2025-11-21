from googleapiclient.discovery import build

# Replace 'your_api_key' with your actual API key
API_KEY = 'AIzaSyCN8ssIdEY1tXyJyL7Xrg2dNrRjPCv3m-I'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

def youtube_search(query, max_results=10):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=max_results,
        type='video'  # Ensure that only videos are returned
    ).execute()
    
    videos = []
    
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            # Exclude live streams and upcoming broadcasts
            if search_result['snippet']['liveBroadcastContent'] == 'none':
                video = {
                    'title': search_result['snippet']['title'],
                    'url': f"https://www.youtube.com/watch?v={search_result['id']['videoId']}",
                    'description': search_result['snippet']['description'],
                    'channel': search_result['snippet']['channelTitle']
                }
                videos.append(video)
    
    return videos

