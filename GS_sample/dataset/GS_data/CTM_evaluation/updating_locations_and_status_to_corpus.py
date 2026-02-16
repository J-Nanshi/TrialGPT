#%%
import json
import requests
from tqdm import tqdm
from copy import deepcopy
#%%
# Input JSON 
input_json = json.load(open(r"GS_corpus_with_gt_pan_cancer_cases_trials_tobe_used_in_GS_algo.json"))
input_json
#%%
# Prepare Output JSONL Line
# Extracting status and locations for each trial
status_json= []
for item in tqdm(input_json):
    url = f"https://clinicaltrials.gov/api/v2/studies?filter.ids={item.get('NCT_ID')}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    a = response.text
    a_=json.loads(a)
    Overallstatus_list = []
    locations_list = []
    countrys_list = []
    # Extract 'studies' from the output json
    studies = a_.get('studies', [])

    # Iterate through studies
    for study in studies:
        # print(study)
        status = study.get('protocolSection', {}).get('statusModule', {}).get('overallStatus', '').upper()
        Overallstatus_list.append(status)
        # Check if status is one of the desired statuses
        # print(status)
            # Get the locations
        locations = study.get('protocolSection', {}).get('contactsLocationsModule', {}).get("locations", {})
        for location in locations:
            # print(location)
            country = location.get('country', '').upper()
            # print(country)
            locations_list.append(location)
            countrys_list.append(country)
        entry = {
        "NCT_ID": item.get('NCT_ID'),
        "Status": Overallstatus_list[0],
        "locations_details": locations_list,
        "Countries": list(set(countrys_list))
        }
    status_json.append(entry)
# %%
status_json
# %%
#Updating locations and status to the original trial json
for nct_entry in tqdm(input_json):
    for status_entry in status_json:
        if nct_entry['NCT_ID'] == status_entry['NCT_ID']:
            # Append the status data to the matching nct entry
            nct_entry.update(status_entry)

# Output the merged result
input_json
# %%
## Adding a key "formatted locations" which contais all the locations details as a single string for that particular trial
def format_locations(trial_data):
    try:
        formatted_locations = []
        locations = trial_data.get('locations_details', [])
        
        if not isinstance(locations, list):
            return trial_data
                
        for loc in locations:
            if not isinstance(loc, dict):
                continue
                
            try:
                location_parts = []
                
                # Add parts only if they exist and are not empty
                for field in ['facility', 'city', 'state', 'zip', 'country']:
                    value = loc.get(field, '').strip()
                    if value:
                        if field == 'zip' and location_parts and 'state' in loc:
                            location_parts[-1] = f"{location_parts[-1]} {value}"
                        else:
                            location_parts.append(value)
                
                # Join all parts with commas
                location_str = ', '.join(location_parts)
                
                # Add status if it exists
                status = loc.get('status', '').strip()
                if status:
                    location_str += f" ({status})"
                
                # Add contacts if they exist
                contacts = loc.get('contacts', [])
                if contacts and isinstance(contacts, list):
                    for contact in contacts:
                        if isinstance(contact, dict):
                            name = contact.get('name', '').strip()
                            role = contact.get('role', '').strip()
                            if name or role:
                                contact_str = "\nContact: "
                                if name and role:
                                    contact_str += f"{name}, {role}"
                                elif name:
                                    contact_str += name
                                elif role:
                                    contact_str += role
                                location_str += contact_str
                
                if location_str:
                    formatted_locations.append(location_str)
                    
            except Exception as e:
                continue
        
        # Create a deep copy of the original data
        processed_data = deepcopy(trial_data)
        
        # Add formatted locations as a new key
        if formatted_locations:
            processed_data['formatted_locations'] = '\n\n'.join(formatted_locations)
        else:
            processed_data['formatted_locations'] = "No location details available"
            
        return processed_data
        
    except Exception as e:
        return trial_data

def process_all_trials(trials_list):
    processed_trials = []
    
    for trial in trials_list:
        processed_trial = format_locations(trial)
        processed_trials.append(processed_trial)
    
    return processed_trials
# %%
input_data_formatted = process_all_trials(input_json)
input_data_formatted
# %%
### saving the GS_corpus_with_gt_pan_cancer_cases_trials_tobe_used_in_GS_algo file
with open('GS_corpus_with_gt_pan_cancer_cases_trials_tobe_used_in_GS_algo_loc_status_updated_and_formatted.json', 'w') as json_file:
    json.dump(input_data_formatted, json_file, indent=4)
# %%
