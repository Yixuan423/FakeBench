import torch
from PIL import Image
import re
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
    save_name = "test_FakeQA.json"
    f = open(r"Evaluation/FakeQA.json", encoding='utf-8')
    data = json.load(f)
    f.close()
    answers = {}
    gpt_request = Request()
    all_num = len(data)

    img_num = 1
    start_time = time.time()
    #####-------FakeQA--------------------------
    responses = []
    for obj in data:
        q_list = []
        imgName = obj.get('image_id')
        print(imgName)
        img_path = os.path.join(path, imgName)
        for key, value in obj.items():
            if key.startswith('Q'):
                value = "Answer this question without itemizing: " + value
                q_list.append(value)
        modified_q_list = [
            question[:question.find('?')] + " regarding the image authenticity?" for
            question in q_list
        ]
        target_word_list = ['texture', 'edge', 'clarity', 'light', 'shadow', 'layout',
                            'symmetry', 'reflection', 'perspective', 'shape', 'theme',
                            'deficiency', 'distortion', 'unrealistic', 'color', 'tone']
        start = time.time()
        time.sleep(1)

        for idx, question in enumerate(modified_q_list):
            words_in_question = [word for word in target_word_list if word in question.lower()]
            FB_prompt = question
            FB_message = gpt_request.forward(FB_prompt, img_path)
            qanswer = ""
            sentences = FB_message.split(". ")
            answer_dict = {}
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(re.search(r'\b' + re.escape(target_word) + r'\w*', sentence_lower) for target_word in
                       words_in_question):
                    qanswer += sentence.replace('\n', ' ').strip() + ' '
                else:
                    qanswer = FB_message
            answer_dict[f'A{idx + 1}'] = qanswer.strip()
            answer_dict[f'A{idx + 1}_org'] = FB_message
            test_obj = {
                'image_id': imgName
            }
            test_obj.append(answer_dict)
        responses.append(test_obj)
        avg_time = (time.time() - start_time) / img_num
        need_time = (avg_time * (all_num - img_num)) / all_num
        print(
            "FakeQA--{}/{} finished. Using time (s):{:.1f}. Average image time (s):{:.1f}. Need time (h):{:.1f}.".format(
                img_num, all_num, time.time() - start, avg_time, need_time))
        img_num = img_num + 1

    with open(save_name, 'w', encoding='utf-8') as file:
        json.dump(responses, file, ensure_ascii=False, indent=4)


