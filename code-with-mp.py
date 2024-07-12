import openai
openai.api_key = 'YOURKEY'
import time
from urllib.error import HTTPError
import pandas as pd
from tqdm import tqdm  # Import tqdm


processed_data = pd.read_csv('processed-data.csv')
processed_metaphors = set(processed_data['metaphor'])  
def is_processed(metaphor):
    return metaphor in processed_metaphors

def extract_metaphor(row_data, max_retries=5, backoff_factor=1.1, max_sleep=30):
    """
    Extracts the metaphorical elements and explanation for a given row of data
    from the dataset using the OpenAI API.
    """
    metaphor = row_data['metaphor']
    if is_processed(metaphor):
        print(f"Metaphor '{metaphor}' has already been processed.")
        return "Already processed"
    for attempt in range(1, max_retries + 1):
        try:
            prompt = (
                 f"1. Metaphor analysis: Analyze the metaphor: {row_data['metaphor']}. Focus on the metaphorical mapping involving the source and target frames: {row_data['sourceFrame']} and {row_data['targetFrame']}, including corresponding roles {row_data['sourceRoles']} and {row_data['targetRoles']}. 2: Find source roles: use frame semantics principles to identify and connect potential roles and elements from the source to the target domain that are made available. Your analysis should: 1. Enumerate any explicit roles or elements present within the source frame, which if present are here: {row_data['sourceRoles']}. If they're not specified, add them yourself  3. Role mapping with target roles: map the identified source role to its corresponding target role, if one is readily apparent here: {row_data['targetRoles']}, meaning you don't need to map them all if not relevant. If no target role is specified, add it yourself. Return the information in the format %%SOURCE_ROLE --> TARGET_ROLE%%. 4. Infer missing roles: when roles are missing or implicit, infer their presence and articulate the potential roles in the mapping, guided by semantic intuition and the principles of metaphorical transfer. 5. Highlight any concept that link these roles, marking core concepts or ideas in the format &&key concept&&, to signify their importance to the metaphor's underlying structure. 6. Reassess your initial analysis: review key concepts you have previously defined: {row_data['extracted_text']}, making sure they are not more than 10. When key concepts are missing, add them. 6. Confirm whether the key concepts explain the metaphor's emergence, focusing on the roles and concepts interplay. (Don't return this reasoning). 7. For instance, in the metaphor ACQUIRING_IDEAS_IS_EATING, critical concepts such as &&transformation&& and &&internalization&& emerge. This metaphor extends beyond mere conversion, encompassing subtleties like &&saturation&& and &&fulfillment&&, indicative of the extent to which ideas are absorbed; &&quality of consumption&&, reflecting the discernment of ideas; &&preferences&& and &&taste&&, denoting the selective nature of cognitive engagement; and the act of chewing as analogous to contemplation, where ideas are deliberated upon. Apply this level of analysis to solve the tasks I asked you to, spotlighting the nuanced interplay of its roles and concepts, but return nothing but the frame role mappings in the format %%SOURCE_ROLE --> TARGET_ROLE%% and the key concepts, either revised or not, in the format &&key concept&&."
            )

            messages = [{
                "role": "user",
                "content": prompt
            }]

            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0,
                max_tokens=4096,
                frequency_penalty=0.0
            )
            return response.choices[0].message['content'].strip()
        except HTTPError as e:
            if e.code == 503 and attempt < max_retries:
                sleep_time = min(backoff_factor ** attempt, max_sleep)
                print(f"Attempt {attempt}: Service Unavailable. Retrying in {sleep_time} seconds.")
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            print(f"An error occurred: {e}")
            return "An error occurred during API call."

data = pd.read_csv('data.csv')
data = data[300:]

tqdm.pandas(desc="Processing Metaphors")
data['gpt4_output_new'] = data.progress_apply(extract_metaphor, axis=1)

updated_data_path = 'reassessment-output.csv'
data.to_csv(updated_data_path, index=False)
print(data.head())