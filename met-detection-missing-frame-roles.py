import pandas as pd
import openai
import os
# Replace 'YOUR_API_KEY' with your actual OpenAI API key
openai.api_key = 'apikey'

def extract_metaphor(row_data):
    """
    Extracts the metaphorical elements and explanation for a given row of data
    from the dataset using the OpenAI API.
    """
    sentence = (
        f"Metaphor: {row_data['metaphor']}, "
        f"SourceFrame: {row_data['sourceFrame']}, "
        f"TargetFrame: {row_data['targetFrame']}, "
        f"SourceRoles: {row_data['sourceRoles']}, "
        f"TargetRoles: {row_data['targetRoles']}"
    )

    prompt = (
        f"Examine the following data: '{sentence}'. Based on the metaphorical mapping provided, evaluate the given source and target frames: {row_data['sourceFrame']} and {row_data['targetFrame']}. Use frame semantic principles to identify and connect potential roles and elements from the source to the target domain that are made available. Your analysis should: 1. Enumerate any explicit roles or elements present within the source frame. 2. Map the identified source role to its corresponding target role, if one is readily apparent, meaning you don't need to map them all if not relevant, in the format 'SOURCE_ROLE --> TARGET_ROLE'. 3. Highlight any conceptual themes or actions that link these roles, marking core ideas with asterisks to signify their importance to the metaphor's underlying structure. 4. Where roles are missing or implicit, infer their presence and articulate the potential roles in the mapping, guided by semantic intuition and the principles of metaphorical transfer. For instance, in the metaphor ACQUIRING_IDEAS_IS_EATING, critical concepts such as *transformation* and *internalization* emerge. This metaphor extends beyond mere conversion, encompassing subtleties like *saturation* and *fulfillment*, indicative of the extent to which ideas are absorbed; *quality of consumption*, reflecting the discernment of ideas; *preferences* and *taste*, denoting the selective nature of cognitive engagement; and the act of *chewing as analogous to contemplation*, where ideas are deliberated upon. Apply this level of analysis to elucidate the metaphor at hand, spotlighting the nuanced interplay of its roles and concepts."
    )

    messages = [{
        "role": "user",
        "content": prompt
    }]

    response = openai.ChatCompletion.create(
        model = "gpt-4-1106-preview",
        messages=messages,
        temperature=0.5,  # Adjust temperature as needed
        max_tokens=4096,
        frequency_penalty=0.0
    )

    return response.choices[0].message['content'].strip()

data = pd.read_csv('data.csv')
data['gpt4_output'] = data.apply(extract_metaphor, axis=1)

updated_data_path = 'output-finding-frame-target-roles.csv'
data.to_csv(updated_data_path, index=False)
print(data.head())
