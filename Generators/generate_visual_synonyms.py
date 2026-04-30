import json

with open('retrieval/output/tlu_states_synonyms.json', 'r') as f:
    data = json.load(f)

for k, v in data.items():
    name = v['name']
    
    # Handle naming 
    parts = name.replace('-', '_').split('_')
    fruit_type = parts[-1]
    origin = parts[0]
    
    if origin == "other":
        origin_str = "mixed variety"
    elif origin == "newzealand":
        origin_str = "new zealand"
    elif origin == "southafrica":
        origin_str = "south africa"
    else:
        origin_str = origin

    # Generate high quality visual synonyms focusing on realistic presentation
    new_synonyms = {
        f'a clear high resolution photo of a fresh {origin_str} {fruit_type} fruit': 0,
        f'close up macro image of a ripe {origin_str} {fruit_type} isolating texture': 0,
        f'professional high quality photography of {origin_str} {fruit_type}': 0,
        f'a single {origin_str} {fruit_type} with clear background': 0,
        f'fresh agricultural {origin_str} {fruit_type} in a local market': 0
    }
    v['synonyms'] = new_synonyms

with open('retrieval/output/tlu_states_synonyms_Visual_HQ.json', 'w') as f:
    json.dump(data, f, indent=4)

print('Generated visual synonyms HQ!')
