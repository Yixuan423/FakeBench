import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
import json, time, os
import google.generativeai as genai
import PIL.Image

# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
genai.configure(api_key="XX")  # replace with your own api key


class Request():
    def __init__(self) -> None:
        self.model = genai.GenerativeModel('gemini-pro-vision')

    def forward(self, prompt, image_path, server='Gemini'):
        if server == 'Gemini':
            img = PIL.Image.open(image_path)
            text = ""
            while len(text) < 3:
                try:
                    response = self.model.generate_content([prompt, img], stream=True)
                    response.resolve()
                    try:
                        text = response.text.strip()
                    except:
                        text = " "
                except Exception as error:
                    print(error)
                    print('Sleeping for 10 seconds')
                    time.sleep(10)
                    text = text + " "

        return text


if True:

    path = "FakeBench_images/fake_images"
    # path = "FakeBench_images/real_images"
    save_name = "test_FakeClue.json"
    f = open(r"Evaluation/FakeClue.json", encoding='utf-8')
    data = json.load(f)
    f.close()
    answers = {}
    gpt_request = Request()
    all_num = len(data)

    img_num = 1
    start_time = time.time()
    #####-------FakeClue--------------------------
    responses = []
    for obj in data:
        imgName = obj.get('image_id')
        print(imgName)
        img_path = os.path.join(path, imgName)
        FB_prompt_ff = obj.get("faultfinding_mode")
        FB_prompt_inf = obj.get("inference_mode")
        start = time.time()
        time.sleep(1)
        FB_message_ff = gpt_request.forward(FB_prompt_ff, img_path)
        FB_message_inf = gpt_request.forward(FB_prompt_inf, img_path)

        test_obj = {
            'image_id': imgName,
            'faultfinding_mode': FB_message_ff,
            'inference_mode': FB_message_inf
        }
        responses.append(test_obj)
        avg_time = (time.time() - start_time) / img_num
        need_time = (avg_time * (all_num - img_num)) / all_num
        print(
            "FakeClue--{}/{} finished. Using time (s):{:.1f}. Average image time (s):{:.1f}. Need time (h):{:.1f}.".format(
                img_num, all_num, time.time() - start, avg_time, need_time))
        img_num = img_num + 1

    with open(save_name, 'w', encoding='utf-8') as file:
        json.dump(responses, file, ensure_ascii=False, indent=4)


