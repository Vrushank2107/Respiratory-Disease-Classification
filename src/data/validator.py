VALID_LOCATIONS = {
    "Tc", "Al", "Ar", "Pl", "Pr", "Ll", "Lr"
}

VALID_MODES = {
    "sc", "mc"
}

VALID_EQUIPMENT = {
    "AKGC417L",
    "LittC2SE",
    "Litt3200",
    "Meditron"
}


def validate_location(location):
    return location in VALID_LOCATIONS


def validate_mode(mode):
    return mode in VALID_MODES


def validate_equipment(equipment):
    return equipment in VALID_EQUIPMENT


def validate_patient_id(patient_id):
    return isinstance(patient_id, int)


def validate_parsed_record(record):
    errors = []

    if not validate_patient_id(record["patient_id"]):
        errors.append("invalid_patient_id")

    if not validate_location(record["chest_location"]):
        errors.append("invalid_location")

    if not validate_mode(record["acquisition_mode"]):
        errors.append("invalid_mode")

    if not validate_equipment(record["equipment"]):
        errors.append("invalid_equipment")

    return errors

def validate_annotation_cycle(cycle):
    """
    Validate one ICBHI respiratory cycle annotation.
    """

    errors = []

    start_time = cycle["start_time"]
    end_time = cycle["end_time"]
    crackles = cycle["crackles"]
    wheezes = cycle["wheezes"]

    if start_time < 0:
        errors.append("negative_start_time")

    if end_time <= start_time:
        errors.append("invalid_time_range")

    if crackles not in {0, 1}:
        errors.append("invalid_crackles_value")

    if wheezes not in {0, 1}:
        errors.append("invalid_wheezes_value")

    return errors