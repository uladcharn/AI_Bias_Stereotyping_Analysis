import os
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

if __name__ == "__main__":
    from utils import create_prompts, collect_data
    from models_llama import SLMModelInstance
    import sys
    import json
    import time

    model_name = sys.argv[1]
    target_cat = sys.argv[2]
    print(target_cat)

    out_folder = f'./outputs/{model_name}'
    try:
        os.mkdir(out_folder)
    except:
        pass

    with open('./data/jobs.json', 'r', encoding='utf-8') as file:
        jobs_data = json.load(file)

    jobs = jobs_data['jobs']
    resume_text = jobs_data['resumes']

    cities = ["San Francisco, CA", "New York, NY", "Cambridge, MA", "Des Moines, SD", 
            "Mobile, AL", "Charlotte, NC", "Yongston, OH"]
    colleges = ["Harvard University", "Smith College", "CUNY Baruch College", 
                    "Carleston College", "Ohio State University", "Wichita State University"]
    degrees = ["Bachelor", "MBA", "MS", "PhD"]
    races = ["black","white","asian"]
    genders = ["male","female"]

    if model_name == "Phi-3.5-mini":
        model_id = "Phi-3.5-mini-instruct-Q4_K_M"
    elif model_name == "google_gemma-3":
        model_id = "google_gemma-3-4b-it-Q4_K_M"
    elif model_name == "LLama-3.2":
        model_id = "Llama-3.2-3B-Instruct-Q4_K_M"

    if target_cat == "location":
        other_cats_1 = {"college":"Harvard University", "degree":"Bachelor", "race": "white", "gender":"male"}
        other_cats_2 = {"college":"Ohio State University", "degree":"Bachelor", "race": "white", "gender":"male"}
        other_cats_3 = {"college":"Carleston College", "degree":"Bachelor", "race": "white", "gender":"male"}
        categories = cities
    elif target_cat == "college":
        other_cats_1 = {"location":"New York, NY", "degree":"Bachelor", "race": "white", "gender":"male"}
        other_cats_2 = {"location":"Mobile, AL", "degree":"Bachelor", "race": "white", "gender":"male"}
        other_cats_3 = {"location":"Des Moines, SD", "degree":"Bachelor","race": "white", "gender":"male"}
        categories = colleges
    elif target_cat == "race_gender": # develop
        other_cats_1 = {"location":"New York, NY", "degree":"Bachelor", "gender":"male"}
        other_cats_2 = {"location":"New York, NY", "degree":"Bachelor", "gender":"female"}
        other_cats_3 = {"location":"New York, NY", "degree":"Bachelor", "gender":"male"}
        categories = races

    prmpts1, prmpts1_plus = create_prompts(jobs, resume_text, categories, other_cats=other_cats_1)
    prmpts2, prmpts2_plus = create_prompts(jobs, resume_text, categories, other_cats=other_cats_2)
    prmpts3, prmpts3_plus = create_prompts(jobs, resume_text, categories, other_cats=other_cats_3)

    model = SLMModelInstance(model_path=f"./llama.cpp/models/{model_id}.gguf")
    model.start_server()

    print('Preparing Profile 1...')
    data_dict_1 = collect_data(model, prmpts1, categories)
    with open(f'./outputs/{model_name}/data_dict_{model_name}_{target_cat}_1', "w") as f:
        json.dump(data_dict_1, f, indent=4)
    print('Profile 1 is ready!')

    print('Preparing Profile 2...')
    data_dict_2 = collect_data(model, prmpts2, categories)
    with open(f'data_dict_{model_name}_{target_cat}_2', "w") as f:
        json.dump(data_dict_2, f, indent=4)
    print('Profile 2 is ready!')

    print('Preparing Profile 3...')
    data_dict_3 = collect_data(model, prmpts3, categories)
    with open(f'data_dict_{model_name}_{target_cat}_3', "w") as f:
        json.dump(data_dict_3, f, indent=4)
    print('Profile 3 is ready!')

    print('Deleting a model...')
    del(model)
