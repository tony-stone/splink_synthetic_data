def gender_gen_uncorrupted_record(formatted_master_record, record_to_modify={}):
    if not formatted_master_record["sex_or_genderLabel"] or formatted_master_record["sex_or_genderLabel"] not in ['male', 'female']:
        record_to_modify["gender"] = 'female'
    else:   
        record_to_modify["gender"] = formatted_master_record["sex_or_genderLabel"]
    return record_to_modify

def gender_corrupt(formatted_master_record, record_to_modify):
    if record_to_modify["gender"] == 'female':
        record_to_modify["gender"] = 'male'
    else:
        record_to_modify["gender"] = 'female'
    return record_to_modify