from requests import post, exceptions

access_token = "zvTeftV3K59dyTLWqsun5frELEQcsfBn"
url = "http://124.195.248.73:8000"

def send_msg(title, message, tts_emo="默认"):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    data = {
        "title": title,
        "message": message,
        "tts_emo": tts_emo
    }
    try:
        response = post(f"{url}/notify", headers=headers, json=data, timeout=2)
        return response
    except Exception as e:
        print(f"请求发生异常，已忽略该错误：{e}")
        pass

if __name__ == "__main__":
    title = "测试通知"
    message = "这是一个测试消息。\n请忽略此消息。"
    tts_emo = "默认"
    
    response = send_msg(title, message, tts_emo)