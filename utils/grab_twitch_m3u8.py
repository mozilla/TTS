from argparse import ArgumentParser
import requests
import json

def get_access_token(video_id):
    response = json.loads(requests.get("https://api.twitch.tv/api/vods/"+str(video_id)+"/access_token",
                            headers={'Client-ID': 'cn6tt813aqffkddemx3ice0o9urw3f'}).content)

    token = response["token"]
    sig = response["sig"]
    return token, sig

def get_m3u8(video_id):
    token, sig = get_access_token(video_id)
    token = json.dumps(token)
    token = token.replace('\\', '')[1:-1]

    response = requests.get("https://usher.ttvnw.net/vod/"+
                            str(video_id)+
                            ".m3u8?allow_source=true&p=2537257&player_backend=mediaplayer&playlist_include_framerate=true&reassignments_supported=true&sig="+
                            str(sig)+
                            "&token="+
                            token
                            )
    m3u8_str = response.content.decode('utf-8')
    m3u8_list = m3u8_str.split("\n")

    # grab the first one this is hard coded cause deal with it
    variant_playlist = m3u8_list[4]

    playlist_resp = requests.get(variant_playlist)
    playlist = playlist_resp.content.decode('utf-8')
    # hard coded cause deal with it again.... I know im gonna pay for it,
    # just look from the back of the list till you find  a '.' and a int
    playlist_size = int(playlist.split("\n")[-3].split(".")[0])
    playlist_path = "/".join(variant_playlist.split("/")[:-1])+"/"

    return playlist_path, playlist_size

def get_segment(path, segment_num):
    response = requests.get(path+str(segment_num)+".ts")
    with open("seg.ts", 'wb') as seg:
        seg.write(response.content)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('channel')
    args = parser.parse_args()

    user_id = args.channel
    response = requests.get("https://api.twitch.tv/helix/users?login="+user_id,
                       headers={'Client-ID': 'cn6tt813aqffkddemx3ice0o9urw3f'})
    resp_dict = json.loads(response.content)['data'][0]
    response = requests.get("https://api.twitch.tv/helix/videos?user_id="+resp_dict['id']+"&first=100", headers={'Client-ID': 'cn6tt813aqffkddemx3ice0o9urw3f'})
    data_list = json.loads(response.content)['data']

    video_ids = []
    for elt in data_list:
        video_ids.append(elt["id"])

    path, size = get_m3u8(video_ids[2])
    get_segment(path, 0)
